//! WMA Pro single-stream decoder core, ported from FFmpeg's wmaprodec.c
//! (LGPL 2.1+). Only the features reachable from XMA2 streams (1-2 channels,
//! 512 samples per frame, decode_flags 0x10d6) are implemented; unsupported
//! bitstream features return `DecodeError`.

use crate::bits::{BitReader, BitWriter};
use crate::imdct::Imdct;
use crate::tables::*;
use crate::vlc::Vlc;
use std::sync::OnceLock;

pub const MAX_CHANNELS: usize = 2; // per XMA stream
pub const MAX_SUBFRAMES: usize = 32;
pub const MAX_BANDS: usize = 29;
pub const MAX_FRAMESIZE: usize = 32768; // bytes, bit reservoir limit

const BLOCK_MIN_BITS: usize = 6;
const BLOCK_MAX_BITS: usize = 13;
const BLOCK_MAX_SIZE: usize = 1 << BLOCK_MAX_BITS;
const BLOCK_SIZES: usize = BLOCK_MAX_BITS - BLOCK_MIN_BITS + 1;

#[derive(Debug)]
pub struct DecodeError(pub String);

impl std::fmt::Display for DecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "XMA2 decode error: {}", self.0)
    }
}
impl std::error::Error for DecodeError {}

type Result<T> = std::result::Result<T, DecodeError>;

fn err<T>(msg: &str) -> Result<T> {
    Err(DecodeError(msg.to_string()))
}

#[inline]
fn av_log2(v: u32) -> u32 {
    if v == 0 { 0 } else { 31 - v.leading_zeros() }
}

struct StaticTables {
    sf_vlc: Vlc,
    sf_rl_vlc: Vlc,
    coef_vlc: [Vlc; 2],
    vec4_vlc: Vlc,
    vec2_vlc: Vlc,
    vec1_vlc: Vlc,
    /// sine windows, index i -> window of length 1 << (BLOCK_MIN_BITS + i)
    windows: Vec<Vec<f32>>,
}

fn static_tables() -> &'static StaticTables {
    static TABLES: OnceLock<StaticTables> = OnceLock::new();
    TABLES.get_or_init(|| {
        let pair = |t: &[(u16, u8)], off: i32| {
            Vlc::from_lengths(
                t.iter().map(|&(s, l)| (s as i32, l)).collect::<Vec<_>>().into_iter(),
                off,
            )
        };
        let windows = (0..BLOCK_SIZES)
            .map(|i| {
                let n = 1usize << (BLOCK_MIN_BITS + i);
                (0..n)
                    .map(|k| {
                        (((k as f64 + 0.5) * (std::f64::consts::PI / (2.0 * n as f64))) as f32)
                            .sin()
                    })
                    .collect()
            })
            .collect();
        StaticTables {
            sf_vlc: pair(&SCALE_TABLE, -60),
            sf_rl_vlc: pair(&SCALE_RL_TABLE, 0),
            coef_vlc: [
                Vlc::from_lengths(
                    COEF0_LENS
                        .iter()
                        .zip(COEF0_SYMS.iter())
                        .map(|(&l, &s)| (s as i32, l))
                        .collect::<Vec<_>>()
                        .into_iter(),
                    0,
                ),
                pair(&COEF1_TABLE, 0),
            ],
            vec4_vlc: Vlc::from_lengths(
                VEC4_LENS
                    .iter()
                    .zip(VEC4_SYMS.iter())
                    .map(|(&l, &s)| (s as i32, l))
                    .collect::<Vec<_>>()
                    .into_iter(),
                -1,
            ),
            vec2_vlc: pair(&VEC2_TABLE, -1),
            vec1_vlc: pair(&VEC1_TABLE, 0),
            windows,
        }
    })
}

const FVAL_TAB: [f32; 16] = [
    0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
];

#[derive(Clone)]
struct ChannelCtx {
    prev_block_len: usize,
    transmit_coefs: bool,
    num_subframes: usize,
    subframe_len: [usize; MAX_SUBFRAMES],
    subframe_offset: [usize; MAX_SUBFRAMES],
    cur_subframe: usize,
    decoded_samples: usize,
    grouped: bool,
    quant_step: i32,
    reuse_sf: bool,
    scale_factor_step: i32,
    max_scale_factor: i32,
    saved_scale_factors: [[i32; MAX_BANDS]; 2],
    scale_factor_idx: usize,
    table_idx: usize,
    num_vec_coeffs: usize,
    /// buffer in saved_scale_factors used for the current subframe
    /// (C: the `scale_factors` pointer)
    sf_active: usize,
    /// coeffs pointer == &out[coeffs_off]
    coeffs_off: usize,
    out: Vec<f32>, // BLOCK_MAX_SIZE + BLOCK_MAX_SIZE / 2
}

impl ChannelCtx {
    fn new() -> Self {
        ChannelCtx {
            prev_block_len: 0,
            transmit_coefs: false,
            num_subframes: 0,
            subframe_len: [0; MAX_SUBFRAMES],
            subframe_offset: [0; MAX_SUBFRAMES],
            cur_subframe: 0,
            decoded_samples: 0,
            grouped: false,
            quant_step: 0,
            reuse_sf: false,
            scale_factor_step: 0,
            max_scale_factor: 0,
            saved_scale_factors: [[0; MAX_BANDS]; 2],
            scale_factor_idx: 0,
            table_idx: 0,
            num_vec_coeffs: 0,
            sf_active: 0,
            coeffs_off: 0,
            out: vec![0.0; BLOCK_MAX_SIZE + BLOCK_MAX_SIZE / 2],
        }
    }
}

#[derive(Clone)]
struct ChannelGroup {
    num_channels: usize,
    transform: bool,
    transform_band: [bool; MAX_BANDS],
    decorrelation_matrix: [f32; MAX_CHANNELS * MAX_CHANNELS],
    /// channel indices whose coeffs participate, in order
    channels: [usize; MAX_CHANNELS],
}

pub struct WmaProDecoder {
    // stream config
    pub nb_channels: usize,
    bits_per_sample: u32,
    samples_per_frame: usize,
    log2_frame_size: usize,
    len_prefix: bool,
    dynamic_range_compression: bool,
    max_num_subframes: usize,
    subframe_len_bits: u32,
    max_subframe_len_bit: bool,
    min_samples_per_subframe: usize,
    num_sfb: [usize; BLOCK_SIZES],
    sfb_offsets: [[usize; MAX_BANDS + 1]; BLOCK_SIZES],
    sf_offsets: Vec<[[u8; MAX_BANDS]; BLOCK_SIZES]>, // [cur][ref][band]

    // packet decode state
    next_packet_start: usize,
    packet_offset: usize,
    num_saved_bits: usize,
    frame_offset: usize,
    pub packet_loss: bool,
    pub packet_done: bool,
    pub eof_done: bool,
    pub skip_packets: u32,
    buf_bit_size: usize,
    pb: BitWriter,

    // frame decode state
    skip_frame: bool,
    parsed_all_subframes: bool,
    drc_gain: u32,
    pub trim_start: usize,
    pub trim_end: usize,

    // subframe decode state
    subframe_len: usize,
    channels_for_cur_subframe: usize,
    channel_indexes_for_cur_subframe: [usize; MAX_CHANNELS],
    num_bands: usize,
    transmit_num_vec_coeffs: bool,
    table_idx: usize,
    esc_len: u32,
    num_chgroups: usize,
    chgroup: Vec<ChannelGroup>,

    channel: Vec<ChannelCtx>,
    imdct: Imdct,
    tmp: Vec<f32>,
}

impl WmaProDecoder {
    /// `num_stream`: index of this stream; `total_channels`: channel count of
    /// the whole file. Mirrors FFmpeg decode_init for AV_CODEC_ID_XMA2 with
    /// XMA2WAVEFORMATEX extradata.
    pub fn new_xma2(sample_rate: u32, num_stream: usize, total_channels: usize) -> Result<Self> {
        let decode_flags: u32 = 0x10d6;
        let bits_per_sample: u32 = 16;
        let nb_channels = if (num_stream + 1) * 2 > total_channels { 1 } else { 2 };

        let block_align = 2048usize;
        let log2_frame_size = av_log2(block_align as u32) as usize + 4; // 15

        let len_prefix = decode_flags & 0x40 != 0;
        let samples_per_frame = 512usize;

        let log2_max_num_subframes = ((decode_flags & 0x38) >> 3) as usize;
        let max_num_subframes = 1usize << log2_max_num_subframes;
        let max_subframe_len_bit = max_num_subframes == 16 || max_num_subframes == 4;
        let subframe_len_bits = av_log2(log2_max_num_subframes as u32) + 1;

        let num_possible_block_sizes = log2_max_num_subframes + 1;
        let min_samples_per_subframe = samples_per_frame / max_num_subframes;
        let dynamic_range_compression = decode_flags & 0x80 != 0;

        if min_samples_per_subframe < (1 << BLOCK_MIN_BITS) {
            return err("min_samples_per_subframe too small");
        }

        // rate used for scale factor band layout (XMA quantizes the rate)
        let rate = if sample_rate > 44100 {
            48000
        } else if sample_rate > 32000 {
            44100
        } else if sample_rate > 24000 {
            32000
        } else {
            24000
        } as usize;

        let mut num_sfb = [0usize; BLOCK_SIZES];
        let mut sfb_offsets = [[0usize; MAX_BANDS + 1]; BLOCK_SIZES];
        for i in 0..num_possible_block_sizes {
            let subframe_len = samples_per_frame >> i;
            let mut band = 1usize;
            sfb_offsets[i][0] = 0;
            for &cf in CRITICAL_FREQ.iter().take(MAX_BANDS - 1) {
                if sfb_offsets[i][band - 1] >= subframe_len {
                    break;
                }
                let mut offset = (subframe_len * 2 * cf as usize) / rate + 2;
                offset &= !3;
                if offset > sfb_offsets[i][band - 1] {
                    sfb_offsets[i][band] = offset;
                    band += 1;
                }
                if offset >= subframe_len {
                    break;
                }
            }
            sfb_offsets[i][band - 1] = subframe_len;
            num_sfb[i] = band - 1;
            if num_sfb[i] == 0 {
                return err("num_sfb invalid");
            }
        }

        let mut sf_offsets = vec![[[0u8; MAX_BANDS]; BLOCK_SIZES]; BLOCK_SIZES];
        for i in 0..num_possible_block_sizes {
            for b in 0..num_sfb[i] {
                let offset = ((sfb_offsets[i][b] + sfb_offsets[i][b + 1] - 1) << i) >> 1;
                for x in 0..num_possible_block_sizes {
                    let mut v = 0usize;
                    while (sfb_offsets[x][v + 1] << x) < offset {
                        v += 1;
                        assert!(v < MAX_BANDS);
                    }
                    sf_offsets[i][x][b] = v as u8;
                }
            }
        }

        let mut channel = vec![ChannelCtx::new(); nb_channels];
        for c in channel.iter_mut() {
            c.prev_block_len = samples_per_frame;
        }

        Ok(WmaProDecoder {
            nb_channels,
            bits_per_sample,
            samples_per_frame,
            log2_frame_size,
            len_prefix,
            dynamic_range_compression,
            max_num_subframes,
            subframe_len_bits,
            max_subframe_len_bit,
            min_samples_per_subframe,
            num_sfb,
            sfb_offsets,
            sf_offsets,
            next_packet_start: 0,
            packet_offset: 0,
            num_saved_bits: 0,
            frame_offset: 0,
            packet_loss: true,
            packet_done: false,
            eof_done: false,
            skip_packets: 0,
            buf_bit_size: 0,
            pb: BitWriter::new(),
            skip_frame: true,
            parsed_all_subframes: false,
            drc_gain: 0,
            trim_start: 0,
            trim_end: 0,
            subframe_len: 0,
            channels_for_cur_subframe: 0,
            channel_indexes_for_cur_subframe: [0; MAX_CHANNELS],
            num_bands: 0,
            transmit_num_vec_coeffs: false,
            table_idx: 0,
            esc_len: 0,
            num_chgroups: 0,
            chgroup: vec![
                ChannelGroup {
                    num_channels: 0,
                    transform: false,
                    transform_band: [false; MAX_BANDS],
                    decorrelation_matrix: [0.0; MAX_CHANNELS * MAX_CHANNELS],
                    channels: [0; MAX_CHANNELS],
                };
                MAX_CHANNELS
            ],
            channel,
            imdct: Imdct::new(),
            tmp: vec![0.0; BLOCK_MAX_SIZE],
        })
    }

    pub fn samples_per_frame(&self) -> usize {
        self.samples_per_frame
    }

    // ---------------------------------------------------------------- frame

    fn decode_subframe_length(&self, gb: &mut BitReader, offset: usize) -> Result<usize> {
        if offset == self.samples_per_frame - self.min_samples_per_subframe {
            return Ok(self.min_samples_per_subframe);
        }
        if gb.bits_left() < 1 {
            return err("truncated subframe length");
        }
        let frame_len_shift = if self.max_subframe_len_bit {
            if gb.get_bits1() {
                1 + gb.get_bits(self.subframe_len_bits - 1)
            } else {
                0
            }
        } else {
            gb.get_bits(self.subframe_len_bits)
        };
        let subframe_len = self.samples_per_frame >> frame_len_shift;
        if subframe_len < self.min_samples_per_subframe || subframe_len > self.samples_per_frame {
            return err("broken frame: subframe_len");
        }
        Ok(subframe_len)
    }

    fn decode_tilehdr(&mut self, gb: &mut BitReader) -> Result<()> {
        let mut num_samples = [0usize; MAX_CHANNELS];
        let mut contains_subframe = [false; MAX_CHANNELS];
        let mut channels_for_cur_subframe = self.nb_channels;
        let mut min_channel_len = 0usize;

        for c in 0..self.nb_channels {
            self.channel[c].num_subframes = 0;
        }

        let fixed_channel_layout = self.max_num_subframes == 1 || gb.get_bits1();

        loop {
            for c in 0..self.nb_channels {
                if num_samples[c] == min_channel_len {
                    if fixed_channel_layout
                        || channels_for_cur_subframe == 1
                        || min_channel_len
                            == self.samples_per_frame - self.min_samples_per_subframe
                    {
                        contains_subframe[c] = true;
                    } else {
                        contains_subframe[c] = gb.get_bits1();
                    }
                } else {
                    contains_subframe[c] = false;
                }
            }

            let subframe_len = self.decode_subframe_length(gb, min_channel_len)?;

            min_channel_len += subframe_len;
            for c in 0..self.nb_channels {
                if contains_subframe[c] {
                    let chan = &mut self.channel[c];
                    if chan.num_subframes >= MAX_SUBFRAMES {
                        return err("broken frame: num subframes > 31");
                    }
                    chan.subframe_len[chan.num_subframes] = subframe_len;
                    num_samples[c] += subframe_len;
                    chan.num_subframes += 1;
                    if num_samples[c] > self.samples_per_frame {
                        return err("broken frame: channel len > samples_per_frame");
                    }
                } else if num_samples[c] <= min_channel_len {
                    if num_samples[c] < min_channel_len {
                        channels_for_cur_subframe = 0;
                        min_channel_len = num_samples[c];
                    }
                    channels_for_cur_subframe += 1;
                }
            }

            if min_channel_len >= self.samples_per_frame {
                break;
            }
        }

        for c in 0..self.nb_channels {
            let chan = &mut self.channel[c];
            let mut offset = 0usize;
            for i in 0..chan.num_subframes {
                chan.subframe_offset[i] = offset;
                offset += chan.subframe_len[i];
            }
        }
        Ok(())
    }

    fn decode_channel_transform(&mut self, gb: &mut BitReader) -> Result<()> {
        self.num_chgroups = 0;
        if self.nb_channels > 1 {
            let mut remaining_channels = self.channels_for_cur_subframe;

            if gb.get_bits1() {
                return err("channel transform bit not supported");
            }

            while remaining_channels > 0 && self.num_chgroups < self.channels_for_cur_subframe {
                let grp_idx = self.num_chgroups;
                let mut channels = [0usize; MAX_CHANNELS];
                let mut transform = false;
                let mut matrix = [0.0f32; MAX_CHANNELS * MAX_CHANNELS];
                let mut transform_band = [false; MAX_BANDS];

                // channel mask (remaining_channels <= 2 for XMA)
                let num_channels = remaining_channels;
                let mut n = 0;
                for i in 0..self.channels_for_cur_subframe {
                    let channel_idx = self.channel_indexes_for_cur_subframe[i];
                    if !self.channel[channel_idx].grouped {
                        channels[n] = channel_idx;
                        n += 1;
                    }
                    self.channel[channel_idx].grouped = true;
                }

                if num_channels == 2 {
                    if gb.get_bits1() {
                        if gb.get_bits1() {
                            return err("unknown channel transform type");
                        }
                    } else {
                        transform = true;
                        if self.nb_channels == 2 {
                            matrix[0] = 1.0;
                            matrix[1] = -1.0;
                            matrix[2] = 1.0;
                            matrix[3] = 1.0;
                        } else {
                            matrix[0] = 0.70703125;
                            matrix[1] = -0.70703125;
                            matrix[2] = 0.70703125;
                            matrix[3] = 0.70703125;
                        }
                    }
                }

                if transform {
                    if !gb.get_bits1() {
                        for tb in transform_band.iter_mut().take(self.num_bands) {
                            *tb = gb.get_bits1();
                        }
                    } else {
                        for tb in transform_band.iter_mut().take(self.num_bands) {
                            *tb = true;
                        }
                    }
                }

                let g = &mut self.chgroup[grp_idx];
                g.num_channels = num_channels;
                g.transform = transform;
                g.transform_band = transform_band;
                g.decorrelation_matrix = matrix;
                g.channels = channels;

                remaining_channels -= num_channels;
                self.num_chgroups += 1;
            }
        }
        Ok(())
    }

    fn get_large_val(gb: &mut BitReader) -> u32 {
        let mut n_bits = 8;
        if gb.get_bits1() {
            n_bits += 8;
            if gb.get_bits1() {
                n_bits += 8;
                if gb.get_bits1() {
                    n_bits += 7;
                }
            }
        }
        gb.get_bits(n_bits)
    }

    #[allow(clippy::too_many_arguments)]
    fn run_level_decode(
        &self,
        gb: &mut BitReader,
        vlc: &Vlc,
        level_table: &[f32],
        run_table: &[u16],
        coeffs: &mut [f32],
        mut offset: usize,
        num_coefs: usize,
        block_len: usize,
        frame_len_bits: u32,
    ) -> Result<()> {
        let coef_mask = block_len - 1;
        while offset < num_coefs {
            let code = vlc.get(gb);
            if code > 1 {
                offset += run_table[code as usize] as usize;
                let sign_neg = !gb.get_bits1();
                let lvl = level_table[code as usize];
                coeffs[offset & coef_mask] = if sign_neg { -lvl } else { lvl };
            } else if code == 1 {
                break;
            } else {
                let level = Self::get_large_val(gb) as i64;
                if gb.get_bits1() {
                    if gb.get_bits1() {
                        if gb.get_bits1() {
                            return err("broken escape sequence");
                        } else {
                            offset += gb.get_bits(frame_len_bits) as usize + 4;
                        }
                    } else {
                        offset += gb.get_bits(2) as usize + 1;
                    }
                }
                let sign_neg = !gb.get_bits1();
                let v = if sign_neg { -level } else { level };
                coeffs[offset & coef_mask] = v as f32;
            }
            offset += 1;
        }
        if offset > num_coefs {
            return err("overflow in spectral RLE");
        }
        Ok(())
    }

    fn decode_coeffs(&mut self, gb: &mut BitReader, c: usize) -> Result<()> {
        let t = static_tables();
        let vlctable = gb.get_bits1() as usize;
        let vlc = &t.coef_vlc[vlctable];
        let (run, level): (&[u16], &[f32]) = if vlctable == 1 {
            (&COEF1_RUN, &COEF1_LEVEL)
        } else {
            (&COEF0_RUN, &COEF0_LEVEL)
        };

        let num_vec_coeffs = self.channel[c].num_vec_coeffs;
        let coeffs_off = self.channel[c].coeffs_off;
        let subframe_len = self.subframe_len;

        let mut rl_mode = false;
        let mut cur_coeff = 0usize;
        let mut num_zeros = 0usize;

        // borrow the channel's coeff region
        let mut out = std::mem::take(&mut self.channel[c].out);
        let coeffs = &mut out[coeffs_off..coeffs_off + subframe_len];

        let result = (|| -> Result<()> {
            while (self.transmit_num_vec_coeffs || !rl_mode) && cur_coeff + 3 < num_vec_coeffs {
                let mut vals = [0.0f32; 4];
                let idx = t.vec4_vlc.get(gb);
                if idx < 0 {
                    let mut i = 0;
                    while i < 4 {
                        let idx2 = t.vec2_vlc.get(gb);
                        if idx2 < 0 {
                            let mut v0 = t.vec1_vlc.get(gb) as u32;
                            if v0 as usize == VEC1_TABLE.len() - 1 {
                                v0 = v0.wrapping_add(Self::get_large_val(gb));
                            }
                            let mut v1 = t.vec1_vlc.get(gb) as u32;
                            if v1 as usize == VEC1_TABLE.len() - 1 {
                                v1 = v1.wrapping_add(Self::get_large_val(gb));
                            }
                            vals[i] = v0 as f32;
                            vals[i + 1] = v1 as f32;
                        } else {
                            vals[i] = FVAL_TAB[(idx2 >> 4) as usize];
                            vals[i + 1] = FVAL_TAB[(idx2 & 0xF) as usize];
                        }
                        i += 2;
                    }
                } else {
                    vals[0] = FVAL_TAB[(idx >> 12) as usize];
                    vals[1] = FVAL_TAB[((idx >> 8) & 0xF) as usize];
                    vals[2] = FVAL_TAB[((idx >> 4) & 0xF) as usize];
                    vals[3] = FVAL_TAB[(idx & 0xF) as usize];
                }

                for &v in &vals {
                    if v != 0.0 {
                        let positive = gb.get_bits1();
                        coeffs[cur_coeff] = if positive { v } else { -v };
                        num_zeros = 0;
                    } else {
                        coeffs[cur_coeff] = 0.0;
                        num_zeros += 1;
                        rl_mode |= num_zeros > (subframe_len >> 8);
                    }
                    cur_coeff += 1;
                }
            }

            if cur_coeff < subframe_len {
                for v in coeffs[cur_coeff..].iter_mut() {
                    *v = 0.0;
                }
                self.run_level_decode(
                    gb,
                    vlc,
                    level,
                    run,
                    coeffs,
                    cur_coeff,
                    subframe_len,
                    subframe_len,
                    self.esc_len,
                )?;
            }
            Ok(())
        })();

        self.channel[c].out = out;
        result
    }

    fn decode_scale_factors(&mut self, gb: &mut BitReader) -> Result<()> {
        let t = static_tables();
        for i in 0..self.channels_for_cur_subframe {
            let c = self.channel_indexes_for_cur_subframe[i];
            let num_bands = self.num_bands;

            let ch = &mut self.channel[c];
            // C: scale_factors = saved_scale_factors[!scale_factor_idx]
            let active = 1 - ch.scale_factor_idx;
            ch.sf_active = active;

            if ch.reuse_sf {
                let sf_off = &self.sf_offsets[self.table_idx][ch.table_idx];
                for (b, &off) in sf_off.iter().enumerate().take(num_bands) {
                    ch.saved_scale_factors[active][b] =
                        ch.saved_scale_factors[ch.scale_factor_idx][off as usize];
                }
            }

            if ch.cur_subframe == 0 || gb.get_bits1() {
                if !ch.reuse_sf {
                    // DPCM coded scale factors
                    ch.scale_factor_step = gb.get_bits(2) as i32 + 1;
                    let mut val = 45 / ch.scale_factor_step;
                    for b in 0..num_bands {
                        val += t.sf_vlc.get(gb);
                        ch.saved_scale_factors[active][b] = val;
                    }
                } else {
                    // run level coded deltas
                    let mut b = 0usize;
                    while b < num_bands {
                        let idx = t.sf_rl_vlc.get(gb);
                        let (val, sign_neg, skip);
                        if idx == 0 {
                            let code = gb.get_bits(14);
                            val = (code >> 6) as i32;
                            sign_neg = code & 1 == 0;
                            skip = ((code & 0x3f) >> 1) as usize;
                        } else if idx == 1 {
                            break;
                        } else {
                            skip = SCALE_RL_RUN[idx as usize] as usize;
                            val = SCALE_RL_LEVEL[idx as usize] as i32;
                            sign_neg = !gb.get_bits1();
                        }
                        b += skip;
                        if b >= num_bands {
                            return err("invalid scale factor coding");
                        }
                        ch.saved_scale_factors[active][b] += if sign_neg { -val } else { val };
                        b += 1;
                    }
                }
                ch.scale_factor_idx = active;
                ch.table_idx = self.table_idx;
                ch.reuse_sf = true;
            }

            let sf_used = &ch.saved_scale_factors[active];
            ch.max_scale_factor = sf_used[0];
            for &sf in sf_used.iter().take(num_bands).skip(1) {
                if sf > ch.max_scale_factor {
                    ch.max_scale_factor = sf;
                }
            }
        }
        Ok(())
    }

    fn inverse_channel_transform(&mut self) {
        for gi in 0..self.num_chgroups {
            let grp = self.chgroup[gi].clone();
            if !grp.transform {
                continue;
            }
            let num_channels = grp.num_channels;
            let sfb = &self.sfb_offsets[self.table_idx];
            for b in 0..self.num_bands {
                if grp.transform_band[b] {
                    let start = sfb[b];
                    let end = sfb[b + 1].min(self.subframe_len);
                    for y in start..end {
                        let mut data = [0.0f32; MAX_CHANNELS];
                        for (n, &ci) in grp.channels[..num_channels].iter().enumerate() {
                            data[n] = self.channel[ci].out[self.channel[ci].coeffs_off + y];
                        }
                        let mut mat = 0usize;
                        for &ci in grp.channels[..num_channels].iter() {
                            let mut sum = 0.0f32;
                            for d in data[..num_channels].iter() {
                                sum += *d * grp.decorrelation_matrix[mat];
                                mat += 1;
                            }
                            let off = self.channel[ci].coeffs_off + y;
                            self.channel[ci].out[off] = sum;
                        }
                    }
                } else if self.nb_channels == 2 {
                    let start = sfb[b];
                    let end = sfb[b + 1].min(self.subframe_len);
                    if end > start {
                        const S: f32 = 181.0 / 128.0;
                        for k in 0..2usize.min(num_channels) {
                            let ci = grp.channels[k];
                            let off = self.channel[ci].coeffs_off;
                            for y in start..end {
                                self.channel[ci].out[off + y] *= S;
                            }
                        }
                    }
                }
            }
        }
    }

    fn window(&mut self) {
        let t = static_tables();
        for i in 0..self.channels_for_cur_subframe {
            let c = self.channel_indexes_for_cur_subframe[i];
            let ch = &mut self.channel[c];
            let mut winlen = ch.prev_block_len;
            let mut start = ch.coeffs_off as isize - (winlen as isize >> 1);

            if self.subframe_len < winlen {
                start += ((winlen - self.subframe_len) >> 1) as isize;
                winlen = self.subframe_len;
            }

            let window = &t.windows[av_log2(winlen as u32) as usize - BLOCK_MIN_BITS];

            let half = winlen >> 1;
            let start = start as usize;
            // vector_fmul_window(start, start, start + half, window, half)
            // overlap region: buf[start .. start + 2*half], window len 2*half
            for k in 0..half {
                let s0 = ch.out[start + k];
                let s1 = ch.out[start + 2 * half - 1 - k];
                let wi = window[k];
                let wj = window[2 * half - 1 - k];
                ch.out[start + k] = s0 * wj - s1 * wi;
                ch.out[start + 2 * half - 1 - k] = s0 * wi + s1 * wj;
            }

            ch.prev_block_len = self.subframe_len;
        }
    }

    fn decode_subframe(&mut self, gb: &mut BitReader) -> Result<()> {
        let mut offset = self.samples_per_frame;
        let mut subframe_len = self.samples_per_frame;
        let mut total_samples = self.samples_per_frame * self.nb_channels;
        let mut transmit_coeffs = false;

        for i in 0..self.nb_channels {
            self.channel[i].grouped = false;
            if offset > self.channel[i].decoded_samples {
                offset = self.channel[i].decoded_samples;
                subframe_len = self.channel[i].subframe_len[self.channel[i].cur_subframe];
            }
        }

        self.channels_for_cur_subframe = 0;
        for i in 0..self.nb_channels {
            let cur_subframe = self.channel[i].cur_subframe;
            total_samples -= self.channel[i].decoded_samples;
            if offset == self.channel[i].decoded_samples
                && subframe_len == self.channel[i].subframe_len[cur_subframe]
            {
                total_samples -= self.channel[i].subframe_len[cur_subframe];
                self.channel[i].decoded_samples += self.channel[i].subframe_len[cur_subframe];
                self.channel_indexes_for_cur_subframe[self.channels_for_cur_subframe] = i;
                self.channels_for_cur_subframe += 1;
            }
        }

        if total_samples == 0 {
            self.parsed_all_subframes = true;
        }

        self.table_idx = av_log2((self.samples_per_frame / subframe_len) as u32) as usize;
        self.num_bands = self.num_sfb[self.table_idx];

        let offset = offset + (self.samples_per_frame >> 1);

        for i in 0..self.channels_for_cur_subframe {
            let c = self.channel_indexes_for_cur_subframe[i];
            self.channel[c].coeffs_off = offset;
        }

        self.subframe_len = subframe_len;
        self.esc_len = av_log2(self.subframe_len as u32 - 1) + 1;

        // skip extended header if any
        if gb.get_bits1() {
            let mut num_fill_bits = gb.get_bits(2) as usize;
            if num_fill_bits == 0 {
                let len = gb.get_bits(4);
                num_fill_bits = if len == 0 { 0 } else { gb.get_bits(len) as usize } + 1;
            }
            if gb.bits_count() + num_fill_bits > self.num_saved_bits {
                return err("invalid number of fill bits");
            }
            gb.skip(num_fill_bits);
        }

        if gb.get_bits1() {
            return err("reserved bit set");
        }

        self.decode_channel_transform(gb)?;

        for i in 0..self.channels_for_cur_subframe {
            let c = self.channel_indexes_for_cur_subframe[i];
            self.channel[c].transmit_coefs = gb.get_bits1();
            if self.channel[c].transmit_coefs {
                transmit_coeffs = true;
            }
        }

        if transmit_coeffs {
            let mut quant_step = (90 * self.bits_per_sample as i32) >> 4;

            self.transmit_num_vec_coeffs = gb.get_bits1();
            if self.transmit_num_vec_coeffs {
                let num_bits = av_log2(self.subframe_len.div_ceil(4) as u32) + 1;
                for i in 0..self.channels_for_cur_subframe {
                    let c = self.channel_indexes_for_cur_subframe[i];
                    let num_vec_coeffs = (gb.get_bits(num_bits) as usize) << 2;
                    if num_vec_coeffs > self.subframe_len {
                        return err("num_vec_coeffs too large");
                    }
                    self.channel[c].num_vec_coeffs = num_vec_coeffs;
                }
            } else {
                for i in 0..self.channels_for_cur_subframe {
                    let c = self.channel_indexes_for_cur_subframe[i];
                    self.channel[c].num_vec_coeffs = self.subframe_len;
                }
            }

            let mut step = gb.get_sbits(6);
            quant_step += step;
            if step == -32 || step == 31 {
                let sign: i32 = if step == 31 { 0 } else { -1 };
                let mut quant = 0i32;
                while gb.bits_count() + 5 < self.num_saved_bits {
                    step = gb.get_bits(5) as i32;
                    if step != 31 {
                        break;
                    }
                    quant += 31;
                }
                quant_step += ((quant + step) ^ sign) - sign;
            }

            if self.channels_for_cur_subframe == 1 {
                let c = self.channel_indexes_for_cur_subframe[0];
                self.channel[c].quant_step = quant_step;
            } else {
                let modifier_len = gb.get_bits(3);
                for i in 0..self.channels_for_cur_subframe {
                    let c = self.channel_indexes_for_cur_subframe[i];
                    self.channel[c].quant_step = quant_step;
                    if gb.get_bits1() {
                        if modifier_len != 0 {
                            self.channel[c].quant_step +=
                                gb.get_bits(modifier_len) as i32 + 1;
                        } else {
                            self.channel[c].quant_step += 1;
                        }
                    }
                }
            }

            self.decode_scale_factors(gb)?;
        }

        // parse coefficients
        for i in 0..self.channels_for_cur_subframe {
            let c = self.channel_indexes_for_cur_subframe[i];
            if self.channel[c].transmit_coefs && gb.bits_count() < self.num_saved_bits {
                self.decode_coeffs(gb, c)?;
            } else {
                let off = self.channel[c].coeffs_off;
                for v in self.channel[c].out[off..off + subframe_len].iter_mut() {
                    *v = 0.0;
                }
            }
        }

        if transmit_coeffs {
            self.inverse_channel_transform();
            for i in 0..self.channels_for_cur_subframe {
                let c = self.channel_indexes_for_cur_subframe[i];
                let num_bands = self.num_bands;
                let subframe_len = self.subframe_len;
                let sfb = self.sfb_offsets[self.table_idx];

                let (quant_step, max_sf, sf_step, sf_idx) = {
                    let ch = &self.channel[c];
                    (ch.quant_step, ch.max_scale_factor, ch.scale_factor_step, ch.sf_active)
                };

                // inverse quantization and rescaling into tmp
                for b in 0..num_bands {
                    let start = sfb[b];
                    let end = sfb[b + 1].min(subframe_len);
                    let sf = self.channel[c].saved_scale_factors[sf_idx][b];
                    let exp = quant_step - (max_sf - sf) * sf_step;
                    let quant = (10f64).powf(exp as f64 / 20.0) as f32;
                    let off = self.channel[c].coeffs_off;
                    for y in start..end {
                        self.tmp[y] = self.channel[c].out[off + y] * quant;
                    }
                }

                // IMDCT (half output)
                let scale = 1.0
                    / (1u64 << (av_log2(subframe_len as u32) as u64 - 1)) as f64
                    / (1u64 << (self.bits_per_sample - 1)) as f64;
                let off = self.channel[c].coeffs_off;
                let mut out = std::mem::take(&mut self.channel[c].out);
                self.imdct
                    .imdct_half(&self.tmp[..subframe_len], &mut out[off..off + subframe_len], scale);
                self.channel[c].out = out;
            }
        }

        // window and overlap-add
        self.window();

        for i in 0..self.channels_for_cur_subframe {
            let c = self.channel_indexes_for_cur_subframe[i];
            if self.channel[c].cur_subframe >= self.channel[c].num_subframes {
                return err("broken subframe");
            }
            self.channel[c].cur_subframe += 1;
        }
        Ok(())
    }

    /// Decode one WMA frame from the bit reservoir. Returns
    /// (more_frames, Some(samples)) where samples is per-channel data of
    /// samples_per_frame length, or None when the frame is skipped.
    fn decode_frame(&mut self, gb: &mut BitReader) -> (bool, Option<Vec<Vec<f32>>>) {
        let len = if self.len_prefix {
            gb.get_bits(self.log2_frame_size as u32) as usize
        } else {
            0
        };

        if self.decode_tilehdr(gb).is_err() {
            self.packet_loss = true;
            return (false, None);
        }

        // postproc transform
        if self.nb_channels > 1 && gb.get_bits1() && gb.get_bits1() {
            for _ in 0..self.nb_channels * self.nb_channels {
                gb.skip(4);
            }
        }

        if self.dynamic_range_compression {
            self.drc_gain = gb.get_bits(8);
        }

        if gb.get_bits1() {
            if gb.get_bits1() {
                self.trim_start = gb.get_bits(av_log2(self.samples_per_frame as u32 * 2)) as usize;
            }
            if gb.get_bits1() {
                self.trim_end = gb.get_bits(av_log2(self.samples_per_frame as u32 * 2)) as usize;
            }
        } else {
            self.trim_start = 0;
            self.trim_end = 0;
        }

        // reset subframe states
        self.parsed_all_subframes = false;
        for i in 0..self.nb_channels {
            self.channel[i].decoded_samples = 0;
            self.channel[i].cur_subframe = 0;
            self.channel[i].reuse_sf = false;
        }

        while !self.parsed_all_subframes {
            if self.decode_subframe(gb).is_err() {
                self.packet_loss = true;
                return (false, None);
            }
        }

        // copy samples to the output
        let mut samples: Option<Vec<Vec<f32>>> = None;
        if !self.skip_frame {
            let mut chans = Vec::with_capacity(self.nb_channels);
            for i in 0..self.nb_channels {
                chans.push(self.channel[i].out[..self.samples_per_frame].to_vec());
            }
            samples = Some(chans);
        } else {
            self.skip_frame = false;
        }

        for i in 0..self.nb_channels {
            let spf = self.samples_per_frame;
            let ch = &mut self.channel[i];
            // reuse second half of the IMDCT output for the next frame
            let (a, b) = ch.out.split_at_mut(spf);
            a[..spf / 2].copy_from_slice(&b[..spf / 2]);
        }

        if self.len_prefix {
            let used = gb.bits_count() - self.frame_offset;
            if len != used + 2 {
                self.packet_loss = true;
                return (false, samples);
            }
            gb.skip(len - used - 1);
        } else {
            while gb.bits_count() < self.num_saved_bits && !gb.get_bits1() {}
        }

        let more_frames = gb.get_bits1();
        (more_frames, samples)
    }

    // --------------------------------------------------------------- packet

    fn save_bits(&mut self, gb: &mut BitReader, gb_buf: &[u8], len: usize, append: bool) {
        let buflen;
        if !append {
            self.frame_offset = gb.bits_count() & 7;
            self.num_saved_bits = self.frame_offset;
            self.pb.reset();
            buflen = (self.num_saved_bits + len + 7) >> 3;
        } else {
            buflen = (self.pb.bits_count() + len + 7) >> 3;
        }

        if len == 0 || buflen > MAX_FRAMESIZE {
            self.packet_loss = true;
            return;
        }

        self.num_saved_bits += len;
        if !append {
            self.pb
                .copy_bits(gb_buf, gb.bits_count() & !7, self.num_saved_bits);
            gb.skip(len);
        } else {
            let mut align = 8 - (gb.bits_count() & 7);
            if align > len {
                align = len;
            }
            self.pb.put_bits(align as u32, gb.get_bits(align as u32));
            let rest = len - align;
            self.pb.copy_bits(gb_buf, gb.bits_count(), rest);
            gb.skip(rest);
        }
    }

    fn run_decode_frame(&mut self) -> (bool, Option<Vec<Vec<f32>>>) {
        let buf = self.pb.buf.clone();
        let mut gb = BitReader::with_bit_size(&buf, self.num_saved_bits);
        gb.skip(self.frame_offset);
        self.decode_frame(&mut gb)
    }

    /// Decode from one XMA packet (or its remainder). Returns
    /// (consumed_bytes, Option<frame samples>). `data` empty => EOF flush.
    pub fn decode_packet(&mut self, data: &[u8]) -> (usize, Option<Vec<Vec<f32>>>) {
        const BLOCK_ALIGN: usize = 2048;

        if data.is_empty() {
            self.packet_done = false;
            if self.eof_done {
                return (0, None);
            }
            // output remaining overlap samples
            let mut chans = Vec::with_capacity(self.nb_channels);
            for i in 0..self.nb_channels {
                let mut v = vec![0.0f32; self.samples_per_frame];
                v[..self.samples_per_frame / 2]
                    .copy_from_slice(&self.channel[i].out[..self.samples_per_frame / 2]);
                chans.push(v);
            }
            self.eof_done = true;
            self.packet_done = true;
            return (0, Some(chans));
        }

        let mut out_frame: Option<Vec<Vec<f32>>> = None;

        if self.packet_done || self.packet_loss {
            self.packet_done = false;

            self.next_packet_start = data.len() - data.len().min(BLOCK_ALIGN);
            let buf_size = data.len().min(BLOCK_ALIGN);
            self.buf_bit_size = buf_size * 8;

            let buf = &data[..buf_size];
            let mut gb = BitReader::new(buf);

            // XMA2 packet header
            let _num_frames = gb.get_bits(6);
            let num_bits_prev_frame = gb.get_bits(self.log2_frame_size as u32) as usize;
            gb.skip(3);
            self.skip_packets = gb.get_bits(8);

            if num_bits_prev_frame > 0 {
                let remaining_packet_bits = self.buf_bit_size - gb.bits_count();
                let mut nbpf = num_bits_prev_frame;
                if nbpf >= remaining_packet_bits {
                    nbpf = remaining_packet_bits;
                    self.packet_done = true;
                }

                self.save_bits(&mut gb, buf, nbpf, true);
                if !self.packet_loss {
                    let (_more, samples) = self.run_decode_frame();
                    if let Some(s) = samples {
                        out_frame = Some(s);
                    }
                }
            }

            if self.packet_loss {
                self.num_saved_bits = 0;
                self.packet_loss = false;
            }

            self.finish_packet_step(&mut gb, buf, &mut out_frame)
        } else {
            let size = data.len() - self.next_packet_start.min(data.len());
            self.buf_bit_size = size * 8;
            let buf = &data[..size];
            let mut gb = BitReader::new(buf);
            gb.skip(self.packet_offset);

            let remaining = self.buf_bit_size as i64 - gb.bits_count() as i64;
            let frame_size = if remaining > self.log2_frame_size as i64 {
                gb.show_bits(self.log2_frame_size as u32) as usize
            } else {
                0
            };

            if self.len_prefix
                && remaining > self.log2_frame_size as i64
                && frame_size != 0
                && frame_size as i64 <= remaining
            {
                self.save_bits(&mut gb, buf, frame_size, false);
                if !self.packet_loss {
                    let (more, samples) = self.run_decode_frame();
                    self.packet_done = !more;
                    if let Some(s) = samples {
                        out_frame = Some(s);
                    }
                }
            } else {
                self.packet_done = true;
            }

            self.finish_packet_step(&mut gb, buf, &mut out_frame)
        }
    }

    fn finish_packet_step(
        &mut self,
        gb: &mut BitReader,
        buf: &[u8],
        out_frame: &mut Option<Vec<Vec<f32>>>,
    ) -> (usize, Option<Vec<Vec<f32>>>) {
        let remaining = self.buf_bit_size as i64 - gb.bits_count() as i64;
        if remaining < 0 {
            self.packet_loss = true;
        }

        if self.packet_done && !self.packet_loss && remaining > 0 {
            self.save_bits(gb, buf, remaining as usize, false);
        }

        self.packet_offset = gb.bits_count() & 7;
        (gb.bits_count() >> 3, out_frame.take())
    }

    /// Reset output state (matches FFmpeg flush()).
    pub fn flush(&mut self) {
        for c in self.channel.iter_mut() {
            for v in c.out.iter_mut() {
                *v = 0.0;
            }
        }
        self.packet_loss = true;
        self.skip_packets = 0;
        self.eof_done = false;
        self.skip_frame = true;
    }
}
