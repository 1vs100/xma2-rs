//! WMA v1/v2 (wFormatTag 0x0160/0x0161) decoder, ported from FFmpeg's
//! wmadec.c / wma.c (LGPL 2.1+). This is the codec used inside xWMA
//! containers as consumed by XAudio2.
//!
//! The code deliberately mirrors the C structure (indexed loops, explicit
//! assignment forms) to keep it auditable against the FFmpeg source.
#![allow(clippy::needless_range_loop)]
#![allow(clippy::assign_op_pattern)]
#![allow(clippy::misrefactored_assign_op)]

use crate::bits::BitReader;
use crate::imdct::Imdct;
use crate::vlc::Vlc;
use crate::wma_tables::*;
use crate::wmapro::DecodeError;

const BLOCK_MIN_BITS: usize = 7;
const BLOCK_MAX_BITS: usize = 11;
const BLOCK_MAX_SIZE: usize = 1 << BLOCK_MAX_BITS;
const BLOCK_NB_SIZES: usize = BLOCK_MAX_BITS - BLOCK_MIN_BITS + 1;

const HIGH_BAND_MAX_SIZE: usize = 16;
const NB_LSP_COEFS: usize = 10;
const MAX_CODED_SUPERFRAME_SIZE: usize = 32768;
const MAX_CHANNELS: usize = 2;
const NOISE_TAB_SIZE: usize = 8192;
const LSP_POW_BITS: usize = 7;
/// FFmpeg's MIN_CACHE_BITS on 32-bit bitstream readers.
const MIN_CACHE_BITS: usize = 25;

type Result<T> = std::result::Result<T, DecodeError>;

fn err<T>(msg: &str) -> Result<T> {
    Err(DecodeError(msg.to_string()))
}

#[inline]
fn av_log2(v: u32) -> u32 {
    if v == 0 { 0 } else { 31 - v.leading_zeros() }
}

/// ff_wma_get_frame_len_bits from wma_common.c.
pub fn frame_len_bits_for(sample_rate: u32, version: u8, decode_flags: u32) -> usize {
    let mut bits = if sample_rate <= 16000 {
        9
    } else if sample_rate <= 22050 || (sample_rate <= 32000 && version == 1) {
        10
    } else if sample_rate <= 48000 || version < 3 {
        11
    } else if sample_rate <= 96000 {
        12
    } else {
        13
    };
    if version == 3 {
        match decode_flags & 0x6 {
            0x2 => bits += 1,
            0x4 => bits -= 1,
            0x6 => bits -= 2,
            _ => {}
        }
    }
    bits
}

/// ff_wma_total_gain_to_bits.
fn total_gain_to_bits(total_gain: i32) -> u32 {
    if total_gain < 15 {
        13
    } else if total_gain < 32 {
        12
    } else if total_gain < 40 {
        11
    } else if total_gain < 45 {
        10
    } else {
        9
    }
}

/// Per-table run/level expansion built by init_coef_vlc.
struct CoefTables {
    vlc: Vlc,
    run: Vec<u16>,
    level: Vec<f32>,
}

fn init_coef_vlc(table_idx: usize) -> CoefTables {
    let (codes, bits, levels) = COEF_VLCS[table_idx];
    let n = bits.len();
    let vlc = Vlc::from_codes(codes, bits);
    let mut run = vec![0u16; n];
    let mut level = vec![0f32; n];
    let mut i = 2usize;
    let mut lv = 1i32;
    let mut k = 0usize;
    while i < n {
        let l = levels[k] as usize;
        k += 1;
        for j in 0..l {
            run[i] = j as u16;
            level[i] = lv as f32;
            i += 1;
        }
        lv += 1;
    }
    CoefTables { vlc, run, level }
}

pub struct WmaDecoder {
    version: u8,
    channels: usize,
    block_align: usize,

    use_bit_reservoir: bool,
    use_variable_block_len: bool,
    use_exp_vlc: bool,
    use_noise_coding: bool,
    byte_offset_bits: usize,

    pub frame_len: usize,
    frame_len_bits: usize,
    nb_block_sizes: usize,

    coefs_start: usize,
    coefs_end: [usize; BLOCK_NB_SIZES],
    exponent_bands: [[u16; 25]; BLOCK_NB_SIZES],
    high_band_start: [usize; BLOCK_NB_SIZES],
    exponent_high_sizes: [usize; BLOCK_NB_SIZES],
    exponent_high_bands: [[usize; HIGH_BAND_MAX_SIZE]; BLOCK_NB_SIZES],

    exp_vlc: Option<Vlc>,
    hgain_vlc: Option<Vlc>,
    coef_tables: [CoefTables; 2],
    windows: Vec<Vec<f32>>, // per bsize, len = frame_len >> bsize

    noise_table: Vec<f32>,
    noise_index: usize,
    noise_mult: f32,

    // lsp tables (only built when !use_exp_vlc)
    lsp_cos_table: Vec<f32>,
    lsp_pow_e_table: [f32; 256],
    lsp_pow_m_table1: [f32; 1 << LSP_POW_BITS],
    lsp_pow_m_table2: [f32; 1 << LSP_POW_BITS],

    // block state
    reset_block_lengths: bool,
    block_len_bits: usize,
    next_block_len_bits: usize,
    prev_block_len_bits: usize,
    block_len: usize,
    block_pos: usize,
    ms_stereo: bool,
    channel_coded: [bool; MAX_CHANNELS],
    exponents_bsize: [usize; MAX_CHANNELS],
    exponents: [Vec<f32>; MAX_CHANNELS],
    max_exponent: [f32; MAX_CHANNELS],
    exponents_initialized: [bool; MAX_CHANNELS],
    coefs1: [Vec<f32>; MAX_CHANNELS],
    coefs: [Vec<f32>; MAX_CHANNELS],
    output: Vec<f32>,
    frame_out: [Vec<f32>; MAX_CHANNELS],

    // superframe / bit reservoir state
    last_superframe: Vec<u8>,
    last_bitoffset: usize,
    last_superframe_len: usize,
    pub eof_done: bool,

    imdct: Imdct,
    high_band_coded: [[bool; HIGH_BAND_MAX_SIZE]; MAX_CHANNELS],
    high_band_values: [[i32; HIGH_BAND_MAX_SIZE]; MAX_CHANNELS],
}

impl WmaDecoder {
    /// `version` 1 or 2 (wFormatTag 0x0160 / 0x0161). `extradata` is the
    /// codec-private data following WAVEFORMATEX (4 bytes for v1, 6 for v2;
    /// may be empty, in which case flags2 defaults to 0).
    pub fn new(
        version: u8,
        channels: usize,
        sample_rate: u32,
        bit_rate: u32,
        block_align: usize,
        extradata: &[u8],
    ) -> Result<Self> {
        if version != 1 && version != 2 {
            return err("bad WMA version");
        }
        if block_align == 0 {
            return err("block_align is not set");
        }
        if sample_rate > 50000 || channels == 0 || channels > MAX_CHANNELS || bit_rate == 0 {
            return err("unsupported channels/sample_rate/bit_rate");
        }

        let mut flags2: u32 = 0;
        if version == 1 && extradata.len() >= 4 {
            flags2 = u16::from_le_bytes([extradata[2], extradata[3]]) as u32;
        } else if version == 2 && extradata.len() >= 6 {
            flags2 = u16::from_le_bytes([extradata[4], extradata[5]]) as u32;
        }

        let use_exp_vlc = flags2 & 0x0001 != 0;
        let use_bit_reservoir = flags2 & 0x0002 != 0;
        let mut use_variable_block_len = flags2 & 0x0004 != 0;

        // FFmpeg issue1503 workaround
        if version == 2 && extradata.len() >= 8 && flags2 == 0xd && use_variable_block_len {
            use_variable_block_len = false;
        }

        let frame_len_bits = frame_len_bits_for(sample_rate, version, 0);
        let frame_len = 1usize << frame_len_bits;

        let nb_block_sizes = if use_variable_block_len {
            let mut nb = ((flags2 >> 3) & 3) as usize + 1;
            if bit_rate as usize / channels >= 32000 {
                nb += 2;
            }
            let nb_max = frame_len_bits - BLOCK_MIN_BITS;
            nb = nb.min(nb_max);
            nb + 1
        } else {
            1
        };

        let mut use_noise_coding = true;
        let mut high_freq = sample_rate as f32 * 0.5;

        let sample_rate1 = if version == 2 {
            if sample_rate >= 44100 {
                44100
            } else if sample_rate >= 22050 {
                22050
            } else if sample_rate >= 16000 {
                16000
            } else if sample_rate >= 11025 {
                11025
            } else if sample_rate >= 8000 {
                8000
            } else {
                sample_rate
            }
        } else {
            sample_rate
        };

        let bps = bit_rate as f32 / (channels as u32 * sample_rate) as f32;
        let byte_offset_bits = av_log2((bps * frame_len as f32 / 8.0 + 0.5) as u32) as usize + 2;
        if byte_offset_bits + 3 > MIN_CACHE_BITS {
            return err("byte_offset_bits too large");
        }

        let bps1 = if channels == 2 { bps * 1.6 } else { bps };
        if sample_rate1 == 44100 {
            if bps1 >= 0.61 {
                use_noise_coding = false;
            } else {
                high_freq *= 0.4;
            }
        } else if sample_rate1 == 22050 {
            if bps1 >= 1.16 {
                use_noise_coding = false;
            } else if bps1 >= 0.72 {
                high_freq *= 0.7;
            } else {
                high_freq *= 0.6;
            }
        } else if sample_rate1 == 16000 {
            if bps > 0.5 {
                high_freq *= 0.5;
            } else {
                high_freq *= 0.3;
            }
        } else if sample_rate1 == 11025 {
            high_freq *= 0.7;
        } else if sample_rate1 == 8000 {
            if bps <= 0.625 {
                high_freq *= 0.5;
            } else if bps > 0.75 {
                use_noise_coding = false;
            } else {
                high_freq *= 0.65;
            }
        } else if bps >= 0.8 {
            high_freq *= 0.75;
        } else if bps >= 0.6 {
            high_freq *= 0.6;
        } else {
            high_freq *= 0.5;
        }

        // scale factor band sizes per block size
        let coefs_start = if version == 1 { 3 } else { 0 };
        let mut exponent_sizes = [0usize; BLOCK_NB_SIZES];
        let mut exponent_bands = [[0u16; 25]; BLOCK_NB_SIZES];
        let mut coefs_end = [0usize; BLOCK_NB_SIZES];
        let mut high_band_start = [0usize; BLOCK_NB_SIZES];
        let mut exponent_high_sizes = [0usize; BLOCK_NB_SIZES];
        let mut exponent_high_bands = [[0usize; HIGH_BAND_MAX_SIZE]; BLOCK_NB_SIZES];

        for k in 0..nb_block_sizes {
            let block_len = frame_len >> k;

            if version == 1 {
                let mut lpos = 0usize;
                let mut i = 0usize;
                while i < 25 {
                    let a = WMA_CRITICAL_FREQS[i] as usize;
                    let b = sample_rate as usize;
                    let mut pos = ((block_len * 2 * a) + (b >> 1)) / b;
                    if pos > block_len {
                        pos = block_len;
                    }
                    exponent_bands[0][i] = (pos - lpos) as u16;
                    if pos >= block_len {
                        i += 1;
                        break;
                    }
                    lpos = pos;
                    i += 1;
                }
                exponent_sizes[0] = i;
            } else {
                let a = frame_len_bits - BLOCK_MIN_BITS - k;
                let table: Option<&[u8; 25]> = if a < 3 {
                    if sample_rate >= 44100 {
                        Some(&EXPONENT_BAND_44100[a])
                    } else if sample_rate >= 32000 {
                        Some(&EXPONENT_BAND_32000[a])
                    } else if sample_rate >= 22050 {
                        Some(&EXPONENT_BAND_22050[a])
                    } else {
                        None
                    }
                } else {
                    None
                };
                if let Some(t) = table {
                    let n = t[0] as usize;
                    for i in 0..n {
                        exponent_bands[k][i] = t[1 + i] as u16;
                    }
                    exponent_sizes[k] = n;
                } else {
                    let mut j = 0usize;
                    let mut lpos = 0usize;
                    for i in 0..25 {
                        let a = WMA_CRITICAL_FREQS[i] as usize;
                        let b = sample_rate as usize;
                        let mut pos = ((block_len * 2 * a) + (b << 1)) / (4 * b);
                        pos <<= 2;
                        if pos > block_len {
                            pos = block_len;
                        }
                        if pos > lpos {
                            exponent_bands[k][j] = (pos - lpos) as u16;
                            j += 1;
                        }
                        if pos >= block_len {
                            break;
                        }
                        lpos = pos;
                    }
                    exponent_sizes[k] = j;
                }
            }

            coefs_end[k] = (frame_len - ((frame_len * 9) / 100)) >> k;
            high_band_start[k] =
                ((block_len as f32 * 2.0 * high_freq) / sample_rate as f32 + 0.5) as usize;

            let n = exponent_sizes[k];
            let mut j = 0usize;
            let mut pos = 0usize;
            for i in 0..n {
                let mut start = pos;
                pos += exponent_bands[k][i] as usize;
                let mut end = pos;
                if start < high_band_start[k] {
                    start = high_band_start[k];
                }
                if end > coefs_end[k] {
                    end = coefs_end[k];
                }
                if end > start {
                    exponent_high_bands[k][j] = end - start;
                    j += 1;
                }
            }
            exponent_high_sizes[k] = j;
        }

        // MDCT windows: sine windows of length frame_len >> i
        let windows: Vec<Vec<f32>> = (0..nb_block_sizes)
            .map(|i| {
                let n = frame_len >> i;
                (0..n)
                    .map(|k| {
                        (((k as f64 + 0.5) * (std::f64::consts::PI / (2.0 * n as f64))) as f32)
                            .sin()
                    })
                    .collect()
            })
            .collect();

        // noise generator
        let noise_mult = if use_exp_vlc { 0.02f32 } else { 0.04f32 };
        let mut noise_table = vec![0f32; NOISE_TAB_SIZE];
        if use_noise_coding {
            let norm = (1.0 / (1u64 << 31) as f64) * 3f64.sqrt() * noise_mult as f64;
            let mut seed: u32 = 1;
            for v in noise_table.iter_mut() {
                seed = seed.wrapping_mul(314159).wrapping_add(1);
                *v = ((seed as i32) as f64 * norm) as f32;
            }
        }

        // coefficient VLC selection
        let mut coef_vlc_table = 2usize;
        if sample_rate >= 32000 {
            if bps1 < 0.72 {
                coef_vlc_table = 0;
            } else if bps1 < 1.16 {
                coef_vlc_table = 1;
            }
        }
        let coef_tables = [
            init_coef_vlc(coef_vlc_table * 2),
            init_coef_vlc(coef_vlc_table * 2 + 1),
        ];

        let exp_vlc = if use_exp_vlc {
            Some(Vlc::from_codes(&AAC_SCALEFACTOR_CODE, &AAC_SCALEFACTOR_BITS))
        } else {
            None
        };
        let hgain_vlc = if use_noise_coding {
            Some(Vlc::from_lengths(
                HGAIN_HUFFTAB.iter().map(|&(s, l)| (s as i32, l)),
                -18,
            ))
        } else {
            None
        };

        // lsp_to_curve tables (decode_exp_lsp path)
        let mut lsp_cos_table = vec![0f32; BLOCK_MAX_SIZE];
        let mut lsp_pow_e_table = [0f32; 256];
        let mut lsp_pow_m_table1 = [0f32; 1 << LSP_POW_BITS];
        let mut lsp_pow_m_table2 = [0f32; 1 << LSP_POW_BITS];
        if !use_exp_vlc {
            let wdel = std::f64::consts::PI / frame_len as f64;
            for (i, v) in lsp_cos_table.iter_mut().take(frame_len).enumerate() {
                *v = 2.0 * (wdel * i as f64).cos() as f32;
            }
            for (i, v) in lsp_pow_e_table.iter_mut().enumerate() {
                let e = i as i32 - 126;
                *v = (e as f32 * -0.25).exp2();
            }
            let mut b = 1.0f32;
            for i in (0..(1 << LSP_POW_BITS)).rev() {
                let m = (1 << LSP_POW_BITS) + i;
                let a = m as f32 * (0.5 / (1 << LSP_POW_BITS) as f32);
                let a = 1.0 / a.sqrt().sqrt();
                lsp_pow_m_table1[i] = 2.0 * a - b;
                lsp_pow_m_table2[i] = b - a;
                b = a;
            }
        }

        Ok(WmaDecoder {
            version,
            channels,
            block_align,
            use_bit_reservoir,
            use_variable_block_len,
            use_exp_vlc,
            use_noise_coding,
            byte_offset_bits,
            frame_len,
            frame_len_bits,
            nb_block_sizes,
            coefs_start,
            coefs_end,
            exponent_bands,
            high_band_start,
            exponent_high_sizes,
            exponent_high_bands,
            exp_vlc,
            hgain_vlc,
            coef_tables,
            windows,
            noise_table,
            noise_index: 0,
            noise_mult,
            lsp_cos_table,
            lsp_pow_e_table,
            lsp_pow_m_table1,
            lsp_pow_m_table2,
            reset_block_lengths: true,
            block_len_bits: frame_len_bits,
            next_block_len_bits: frame_len_bits,
            prev_block_len_bits: frame_len_bits,
            block_len: 0,
            block_pos: 0,
            ms_stereo: false,
            channel_coded: [false; MAX_CHANNELS],
            exponents_bsize: [0; MAX_CHANNELS],
            exponents: [vec![0.0; BLOCK_MAX_SIZE], vec![0.0; BLOCK_MAX_SIZE]],
            max_exponent: [1.0; MAX_CHANNELS],
            exponents_initialized: [false; MAX_CHANNELS],
            coefs1: [vec![0.0; BLOCK_MAX_SIZE], vec![0.0; BLOCK_MAX_SIZE]],
            coefs: [vec![0.0; BLOCK_MAX_SIZE], vec![0.0; BLOCK_MAX_SIZE]],
            output: vec![0.0; BLOCK_MAX_SIZE * 2],
            frame_out: [vec![0.0; BLOCK_MAX_SIZE * 2], vec![0.0; BLOCK_MAX_SIZE * 2]],
            last_superframe: vec![0; MAX_CODED_SUPERFRAME_SIZE + 64],
            last_bitoffset: 0,
            last_superframe_len: 0,
            eof_done: false,
            imdct: Imdct::new(),
            high_band_coded: [[false; HIGH_BAND_MAX_SIZE]; MAX_CHANNELS],
            high_band_values: [[0; HIGH_BAND_MAX_SIZE]; MAX_CHANNELS],
        })
    }

    /// pow_m1_4: x^-0.25 via exponent/mantissa tables (bit-identical to C).
    #[inline]
    fn pow_m1_4(&self, x: f32) -> f32 {
        let u = x.to_bits();
        let e = (u >> 23) as usize;
        let m = ((u >> (23 - LSP_POW_BITS)) & ((1 << LSP_POW_BITS) - 1)) as usize;
        let t = f32::from_bits(((u << LSP_POW_BITS) & ((1 << 23) - 1)) | (127 << 23));
        let a = self.lsp_pow_m_table1[m];
        let b = self.lsp_pow_m_table2[m];
        self.lsp_pow_e_table[e] * (a + b * t)
    }

    fn wma_lsp_to_curve(&mut self, ch: usize, n: usize, lsp: &[f32; NB_LSP_COEFS]) {
        let mut val_max = 0f32;
        for i in 0..n {
            let mut p = 0.5f32;
            let mut q = 0.5f32;
            let w = self.lsp_cos_table[i];
            for j in (1..NB_LSP_COEFS).step_by(2) {
                q *= w - lsp[j - 1];
                p *= w - lsp[j];
            }
            p *= p * (2.0 - w);
            q *= q * (2.0 + w);
            let v = self.pow_m1_4(p + q);
            if v > val_max {
                val_max = v;
            }
            self.exponents[ch][i] = v;
        }
        self.max_exponent[ch] = val_max;
    }

    fn decode_exp_lsp(&mut self, gb: &mut BitReader, ch: usize) {
        let mut lsp_coefs = [0f32; NB_LSP_COEFS];
        for i in 0..NB_LSP_COEFS {
            let val = if i == 0 || i >= 8 {
                gb.get_bits(3)
            } else {
                gb.get_bits(4)
            } as usize;
            lsp_coefs[i] = LSP_CODEBOOK[i][val];
        }
        self.wma_lsp_to_curve(ch, self.block_len, &lsp_coefs);
    }

    fn decode_exp_vlc(&mut self, gb: &mut BitReader, ch: usize) -> Result<()> {
        let bsize = self.frame_len_bits - self.block_len_bits;
        let bands = &self.exponent_bands[bsize];
        let vlc = self.exp_vlc.as_ref().unwrap();

        let mut q = 0usize;
        let q_end = self.block_len;
        let mut max_scale = 0f32;
        let mut band_idx = 0usize;

        let mut last_exp: i32;
        if self.version == 1 {
            last_exp = gb.get_bits(5) as i32 + 10;
            let v = POW_TAB[(last_exp + 60) as usize];
            max_scale = v;
            let n = bands[band_idx] as usize;
            band_idx += 1;
            for _ in 0..n {
                self.exponents[ch][q] = v;
                q += 1;
            }
        } else {
            last_exp = 36;
        }

        while q < q_end {
            let code = vlc.get(gb);
            last_exp += code - 60;
            if last_exp + 60 < 0 || (last_exp + 60) as usize >= POW_TAB.len() {
                return err("exponent out of range");
            }
            let v = POW_TAB[(last_exp + 60) as usize];
            if v > max_scale {
                max_scale = v;
            }
            let n = bands[band_idx] as usize;
            band_idx += 1;
            for _ in 0..n {
                if q >= self.exponents[ch].len() {
                    return err("exponent band overflow");
                }
                self.exponents[ch][q] = v;
                q += 1;
            }
        }
        self.max_exponent[ch] = max_scale;
        Ok(())
    }

    /// ff_wma_run_level_decode with version=0 (WMA v1/v2 semantics).
    #[allow(clippy::too_many_arguments)]
    fn run_level_decode(
        &self,
        gb: &mut BitReader,
        tindex: usize,
        ch: usize,
        ptr: &mut [f32],
        num_coefs: usize,
        block_len: usize,
        coef_nb_bits: u32,
    ) -> Result<()> {
        let _ = ch;
        let t = &self.coef_tables[tindex];
        let coef_mask = block_len - 1;
        let mut offset = 0usize;
        while offset < num_coefs {
            let code = t.vlc.get(gb);
            if code > 1 {
                let code = code as usize;
                offset += t.run[code] as usize;
                let positive = gb.get_bits1();
                let level = t.level[code];
                ptr[offset & coef_mask] = if positive { level } else { -level };
            } else if code == 1 {
                break; // EOB
            } else {
                // escape (version 0)
                let level = gb.get_bits(coef_nb_bits) as i32;
                offset += gb.get_bits(self.frame_len_bits as u32) as usize;
                let positive = gb.get_bits1();
                let v = if positive { level } else { -level };
                ptr[offset & coef_mask] = v as f32;
            }
            offset += 1;
        }
        // The C loop increments offset in the for-statement; replicate its
        // final overflow check (offset may legitimately equal num_coefs).
        if offset > num_coefs {
            return err("overflow in spectral RLE");
        }
        Ok(())
    }

    /// Returns Ok(true) if this was the last block of the frame.
    fn decode_block(&mut self, gb: &mut BitReader) -> Result<bool> {
        let channels = self.channels;

        // compute current block length
        if self.use_variable_block_len {
            let n = av_log2(self.nb_block_sizes as u32 - 1) + 1;
            if self.reset_block_lengths {
                self.reset_block_lengths = false;
                let v = gb.get_bits(n) as usize;
                if v >= self.nb_block_sizes {
                    return err("prev_block_len_bits out of range");
                }
                self.prev_block_len_bits = self.frame_len_bits - v;
                let v = gb.get_bits(n) as usize;
                if v >= self.nb_block_sizes {
                    return err("block_len_bits out of range");
                }
                self.block_len_bits = self.frame_len_bits - v;
            } else {
                self.prev_block_len_bits = self.block_len_bits;
                self.block_len_bits = self.next_block_len_bits;
            }
            let v = gb.get_bits(n) as usize;
            if v >= self.nb_block_sizes {
                return err("next_block_len_bits out of range");
            }
            self.next_block_len_bits = self.frame_len_bits - v;
        } else {
            self.next_block_len_bits = self.frame_len_bits;
            self.prev_block_len_bits = self.frame_len_bits;
            self.block_len_bits = self.frame_len_bits;
        }

        if self.frame_len_bits - self.block_len_bits >= self.nb_block_sizes {
            return err("block_len_bits not initialized to a valid value");
        }

        self.block_len = 1 << self.block_len_bits;
        if self.block_pos + self.block_len > self.frame_len {
            return err("frame_len overflow");
        }

        if channels == 2 {
            self.ms_stereo = gb.get_bits1();
        }
        let mut any_coded = false;
        for ch in 0..channels {
            let a = gb.get_bits1();
            self.channel_coded[ch] = a;
            any_coded |= a;
        }

        let bsize = self.frame_len_bits - self.block_len_bits;
        let mut nb_coefs = [0usize; MAX_CHANNELS];

        if any_coded {
            // total gain
            let mut total_gain = 1i32;
            loop {
                if gb.bits_left() < 7 {
                    return err("total_gain overread");
                }
                let a = gb.get_bits(7) as i32;
                total_gain += a;
                if a != 127 {
                    break;
                }
            }
            let coef_nb_bits = total_gain_to_bits(total_gain);

            let n = self.coefs_end[bsize] - self.coefs_start;
            for c in nb_coefs.iter_mut().take(channels) {
                *c = n;
            }

            // complex (noise) coding
            if self.use_noise_coding {
                for ch in 0..channels {
                    if self.channel_coded[ch] {
                        let n = self.exponent_high_sizes[bsize];
                        for i in 0..n {
                            let a = gb.get_bits1();
                            self.high_band_coded[ch][i] = a;
                            if a {
                                nb_coefs[ch] -= self.exponent_high_bands[bsize][i];
                            }
                        }
                    }
                }
                for ch in 0..channels {
                    if self.channel_coded[ch] {
                        let n = self.exponent_high_sizes[bsize];
                        let mut val = i32::MIN;
                        for i in 0..n {
                            if self.high_band_coded[ch][i] {
                                if val == i32::MIN {
                                    val = gb.get_bits(7) as i32 - 19;
                                } else {
                                    val += self.hgain_vlc.as_ref().unwrap().get(gb);
                                }
                                self.high_band_values[ch][i] = val;
                            }
                        }
                    }
                }
            }

            // exponents (can be reused in short blocks)
            if self.block_len_bits == self.frame_len_bits || gb.get_bits1() {
                for ch in 0..channels {
                    if self.channel_coded[ch] {
                        if self.use_exp_vlc {
                            self.decode_exp_vlc(gb, ch)?;
                        } else {
                            self.decode_exp_lsp(gb, ch);
                        }
                        self.exponents_bsize[ch] = bsize;
                        self.exponents_initialized[ch] = true;
                    }
                }
            }
            for ch in 0..channels {
                if self.channel_coded[ch] && !self.exponents_initialized[ch] {
                    return err("exponents not initialized");
                }
            }

            // spectral coefficients: RLE
            for ch in 0..channels {
                if self.channel_coded[ch] {
                    let tindex = (ch == 1 && self.ms_stereo) as usize;
                    self.coefs1[ch][..self.block_len].fill(0.0);
                    let mut buf = std::mem::take(&mut self.coefs1[ch]);
                    let r = self.run_level_decode(
                        gb,
                        tindex,
                        ch,
                        &mut buf,
                        nb_coefs[ch],
                        self.block_len,
                        coef_nb_bits,
                    );
                    self.coefs1[ch] = buf;
                    r?;
                }
                if self.version == 1 && channels >= 2 {
                    let n = gb.bits_count();
                    gb.skip((8 - (n & 7)) & 7);
                }
            }

            // normalize
            let n4 = self.block_len / 2;
            let mut mdct_norm = 1.0f32 / n4 as f32;
            if self.version == 1 {
                mdct_norm *= (n4 as f32).sqrt();
            }

            // build MDCT coefficients
            for ch in 0..channels {
                if !self.channel_coded[ch] {
                    continue;
                }
                let esize = self.exponents_bsize[ch];
                let mult0 = (10f64.powf(total_gain as f64 * 0.05) as f32
                    / self.max_exponent[ch])
                    * mdct_norm;

                if self.use_noise_coding {
                    // shift-based exponent indexing: exponents[i << bsize >> esize]
                    let mut coefs_i = 0usize;
                    let mut coefs1_i = 0usize;

                    let mult1_base = mult0;
                    // very low freqs: noise
                    for i in 0..self.coefs_start {
                        let e = self.exponents[ch][(i << bsize) >> esize];
                        self.coefs[ch][coefs_i] =
                            self.noise_table[self.noise_index] * e * mult1_base;
                        coefs_i += 1;
                        self.noise_index = (self.noise_index + 1) & (NOISE_TAB_SIZE - 1);
                    }

                    let n1 = self.exponent_high_sizes[bsize];

                    // compute power of high bands
                    let mut exp_power = [0f32; HIGH_BAND_MAX_SIZE];
                    let mut exp_off = (self.high_band_start[bsize] << bsize) >> esize;
                    let mut last_high_band = 0usize;
                    for j in 0..n1 {
                        let n = self.exponent_high_bands[bsize][j];
                        if self.high_band_coded[ch][j] {
                            let mut e2 = 0f32;
                            for i in 0..n {
                                let v = self.exponents[ch][exp_off + ((i << bsize) >> esize)];
                                e2 += v * v;
                            }
                            exp_power[j] = e2 / n as f32;
                            last_high_band = j;
                        }
                        exp_off += (n << bsize) >> esize;
                    }

                    // main freqs and high freqs
                    let mut exp_off = (self.coefs_start << bsize) >> esize;
                    for j in -1i32..n1 as i32 {
                        let n = if j < 0 {
                            self.high_band_start[bsize] - self.coefs_start
                        } else {
                            self.exponent_high_bands[bsize][j as usize]
                        };
                        if j >= 0 && self.high_band_coded[ch][j as usize] {
                            // noise with specified power
                            let ju = j as usize;
                            let mut mult1 =
                                (exp_power[ju] / exp_power[last_high_band]).sqrt();
                            mult1 *= 10f64.powf(self.high_band_values[ch][ju] as f64 * 0.05)
                                as f32;
                            mult1 /= self.max_exponent[ch] * self.noise_mult;
                            mult1 *= mdct_norm;
                            for i in 0..n {
                                let noise = self.noise_table[self.noise_index];
                                self.noise_index =
                                    (self.noise_index + 1) & (NOISE_TAB_SIZE - 1);
                                let e = self.exponents[ch][exp_off + ((i << bsize) >> esize)];
                                self.coefs[ch][coefs_i] = noise * e * mult1;
                                coefs_i += 1;
                            }
                            exp_off += (n << bsize) >> esize;
                        } else {
                            // coded values + small noise
                            for i in 0..n {
                                let noise = self.noise_table[self.noise_index];
                                self.noise_index =
                                    (self.noise_index + 1) & (NOISE_TAB_SIZE - 1);
                                let e = self.exponents[ch][exp_off + ((i << bsize) >> esize)];
                                self.coefs[ch][coefs_i] =
                                    (self.coefs1[ch][coefs1_i] + noise) * e * mult0;
                                coefs_i += 1;
                                coefs1_i += 1;
                            }
                            exp_off += (n << bsize) >> esize;
                        }
                    }

                    // very high freqs: noise
                    let n = self.block_len - self.coefs_end[bsize];
                    // C: exponents[(-(1 << bsize)) >> esize] relative to the
                    // advanced exponents pointer.
                    let rel = ((-(1i32 << bsize)) >> esize) + exp_off as i32;
                    let e_last = self.exponents[ch][rel as usize];
                    let mult1 = mult0 * e_last;
                    for _ in 0..n {
                        self.coefs[ch][coefs_i] = self.noise_table[self.noise_index] * mult1;
                        coefs_i += 1;
                        self.noise_index = (self.noise_index + 1) & (NOISE_TAB_SIZE - 1);
                    }
                } else {
                    let mut coefs_i = 0usize;
                    for _ in 0..self.coefs_start {
                        self.coefs[ch][coefs_i] = 0.0;
                        coefs_i += 1;
                    }
                    for i in 0..nb_coefs[ch] {
                        let e = self.exponents[ch][(i << bsize) >> esize];
                        self.coefs[ch][coefs_i] = self.coefs1[ch][i] * e * mult0;
                        coefs_i += 1;
                    }
                    for _ in 0..(self.block_len - self.coefs_end[bsize]) {
                        self.coefs[ch][coefs_i] = 0.0;
                        coefs_i += 1;
                    }
                }
            }

            if self.ms_stereo && self.channel_coded[1] {
                if !self.channel_coded[0] {
                    self.coefs[0][..self.block_len].fill(0.0);
                    self.channel_coded[0] = true;
                }
                // butterflies_float
                let (c0, c1) = self.coefs.split_at_mut(1);
                for i in 0..self.block_len {
                    let t = c0[0][i] - c1[0][i];
                    c0[0][i] += c1[0][i];
                    c1[0][i] = t;
                }
            }
        }

        // IMDCT + windowing
        for ch in 0..channels {
            let n4 = self.block_len / 2;
            if self.channel_coded[ch] {
                // full inverse MDCT (AV_TX_FULL_IMDCT): half output at
                // [n4..n4+block_len), first quarter mirrored negated.
                let mut half = vec![0f32; self.block_len];
                self.imdct.imdct_half(
                    &self.coefs[ch][..self.block_len],
                    &mut half,
                    1.0 / 32768.0,
                );
                let out = &mut self.output;
                let len = self.block_len * 2;
                out[n4..n4 + self.block_len].copy_from_slice(&half);
                for i in 0..n4 {
                    out[i] = -out[self.block_len - i - 1];
                    out[len - i - 1] = out[self.block_len + i];
                }
            } else if !(self.ms_stereo && ch == 1) {
                self.output.fill(0.0);
            }

            // multiply by the window and add in the frame
            let index = (self.frame_len / 2) + self.block_pos - n4;
            self.wma_window(ch, index);
        }

        self.block_pos += self.block_len;
        Ok(self.block_pos >= self.frame_len)
    }

    /// wma_window: apply window to self.output and overlap-add into
    /// frame_out[ch] at `index`.
    fn wma_window(&mut self, ch: usize, index: usize) {
        let block_len = self.block_len;
        let out = &mut self.frame_out[ch][index..];
        let inp = &self.output;

        // left part
        if self.block_len_bits <= self.prev_block_len_bits {
            let bsize = self.frame_len_bits - self.block_len_bits;
            let w = &self.windows[bsize];
            for i in 0..block_len {
                out[i] = inp[i] * w[i] + out[i];
            }
        } else {
            let bl = 1usize << self.prev_block_len_bits;
            let n = (block_len - bl) / 2;
            let bsize = self.frame_len_bits - self.prev_block_len_bits;
            let w = &self.windows[bsize];
            for i in 0..bl {
                out[n + i] = inp[n + i] * w[i] + out[n + i];
            }
            out[n + bl..n + bl + n].copy_from_slice(&inp[n + bl..n + bl + n]);
        }

        // right part
        let out = &mut self.frame_out[ch][index + block_len..];
        let inp = &self.output[block_len..];
        if self.block_len_bits <= self.next_block_len_bits {
            let bsize = self.frame_len_bits - self.block_len_bits;
            let w = &self.windows[bsize];
            for i in 0..block_len {
                out[i] = inp[i] * w[block_len - 1 - i];
            }
        } else {
            let bl = 1usize << self.next_block_len_bits;
            let n = (block_len - bl) / 2;
            let bsize = self.frame_len_bits - self.next_block_len_bits;
            let w = &self.windows[bsize];
            out[..n].copy_from_slice(&inp[..n]);
            for i in 0..bl {
                out[n + i] = inp[n + i] * w[bl - 1 - i];
            }
            out[n + bl..n + bl + n].fill(0.0);
        }
    }

    /// Decode one frame of frame_len samples into `samples` at `offset`.
    fn decode_frame(
        &mut self,
        gb: &mut BitReader,
        samples: &mut [Vec<f32>],
        offset: usize,
    ) -> Result<()> {
        self.block_pos = 0;
        loop {
            if self.decode_block(gb)? {
                break;
            }
        }
        for ch in 0..self.channels {
            samples[ch][offset..offset + self.frame_len]
                .copy_from_slice(&self.frame_out[ch][..self.frame_len]);
            self.frame_out[ch].copy_within(self.frame_len..self.frame_len * 2, 0);
        }
        Ok(())
    }

    /// Decode one superframe (= one block_align sized packet). Returns
    /// planar samples per channel; empty vectors when the packet only
    /// contributed to the bit reservoir. Pass an empty slice at EOF to
    /// flush the last delayed frame.
    pub fn decode_superframe(&mut self, buf: &[u8]) -> Result<Vec<Vec<f32>>> {
        if buf.is_empty() {
            if self.eof_done {
                return Ok(vec![Vec::new(); self.channels]);
            }
            let mut samples: Vec<Vec<f32>> =
                (0..self.channels).map(|_| vec![0f32; self.frame_len]).collect();
            for (ch, s) in samples.iter_mut().enumerate() {
                s.copy_from_slice(&self.frame_out[ch][..self.frame_len]);
            }
            self.last_superframe_len = 0;
            self.eof_done = true;
            return Ok(samples);
        }
        if buf.len() < self.block_align {
            return err("input packet size too small");
        }
        let buf = &buf[..self.block_align];
        let buf_size = self.block_align;

        let mut gb = BitReader::new(buf);

        let nb_frames;
        if self.use_bit_reservoir {
            gb.skip(4); // superframe index
            let raw = gb.get_bits(4) as i32;
            let adj = raw - (self.last_superframe_len == 0) as i32;
            if adj <= 0 {
                if adj < 0 || gb.bits_left() <= 8 {
                    // C returns without resetting the reservoir here
                    return err("nb_frames invalid");
                }
                // packet is pure continuation data: append to reservoir
                if self.last_superframe_len + buf_size - 1 > MAX_CODED_SUPERFRAME_SIZE {
                    self.last_superframe_len = 0;
                    return err("continuation overflows reservoir");
                }
                for q in self.last_superframe_len..self.last_superframe_len + buf_size - 1 {
                    self.last_superframe[q] = gb.get_bits(8) as u8;
                }
                // NB: FFmpeg adds a bit count to this byte count; kept
                // verbatim for parity.
                self.last_superframe_len += 8 * buf_size - 8;
                return Ok(vec![Vec::new(); self.channels]);
            }
            nb_frames = adj as usize;
        } else {
            nb_frames = 1;
        }

        let total = nb_frames * self.frame_len;
        let mut samples: Vec<Vec<f32>> =
            (0..self.channels).map(|_| vec![0f32; total]).collect();
        let mut samples_offset = 0usize;
        let mut frames_left = nb_frames;

        let res = (|| -> Result<()> {
            if self.use_bit_reservoir {
                let bit_offset = gb.get_bits(self.byte_offset_bits as u32 + 3) as usize;
                if bit_offset as i64 > gb.bits_left() {
                    return err("invalid last frame bit offset");
                }

                if self.last_superframe_len > 0 {
                    if self.last_superframe_len + bit_offset.div_ceil(8)
                        > MAX_CODED_SUPERFRAME_SIZE
                    {
                        return err("reservoir overflow");
                    }
                    let mut q = self.last_superframe_len;
                    let mut len = bit_offset;
                    while len > 7 {
                        self.last_superframe[q] = gb.get_bits(8) as u8;
                        q += 1;
                        len -= 8;
                    }
                    if len > 0 {
                        self.last_superframe[q] =
                            (gb.get_bits(len as u32) << (8 - len)) as u8;
                        q += 1;
                    }
                    let pad_end = (q + 64).min(self.last_superframe.len());
                    for pad in self.last_superframe[q..pad_end].iter_mut() {
                        *pad = 0;
                    }

                    let bitsize = self.last_superframe_len * 8 + bit_offset;
                    let resv = std::mem::take(&mut self.last_superframe);
                    let mut gb2 = BitReader::with_bit_size(&resv, bitsize);
                    if self.last_bitoffset > 0 {
                        gb2.skip(self.last_bitoffset);
                    }
                    let r = self.decode_frame(&mut gb2, &mut samples, samples_offset);
                    self.last_superframe = resv;
                    r?;
                    samples_offset += self.frame_len;
                    frames_left -= 1;
                }

                // read each frame starting from bit_offset
                let pos = bit_offset + 4 + 4 + self.byte_offset_bits + 3;
                if pos >= MAX_CODED_SUPERFRAME_SIZE * 8 || pos > buf_size * 8 {
                    return err("invalid frame position");
                }
                let mut gb3 = BitReader::new(&buf[pos >> 3..]);
                if pos & 7 > 0 {
                    gb3.skip(pos & 7);
                }

                self.reset_block_lengths = true;
                for _ in 0..frames_left {
                    self.decode_frame(&mut gb3, &mut samples, samples_offset)?;
                    samples_offset += self.frame_len;
                }

                // save the tail of the packet into the reservoir
                let end_pos = gb3.bits_count()
                    + ((bit_offset + 4 + 4 + self.byte_offset_bits + 3) & !7);
                self.last_bitoffset = end_pos & 7;
                let end_pos = end_pos >> 3;
                if end_pos > buf_size {
                    return err("len invalid");
                }
                let len = buf_size - end_pos;
                if len > MAX_CODED_SUPERFRAME_SIZE {
                    return err("len invalid");
                }
                self.last_superframe_len = len;
                self.last_superframe[..len].copy_from_slice(&buf[end_pos..end_pos + len]);
            } else {
                self.decode_frame(&mut gb, &mut samples, samples_offset)?;
            }
            Ok(())
        })();

        if let Err(e) = res {
            // on error, reset the bit reservoir
            self.last_superframe_len = 0;
            return Err(e);
        }
        Ok(samples)
    }
}
