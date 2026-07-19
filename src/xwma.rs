//! xWMA container (RIFF form `XWMA`, XAudio2) parsing and decoding.
//!
//! xWMA carries WMAv2 (0x0161), WMAv1 (0x0160) or WMA Pro (0x0162) packets of
//! `nBlockAlign` bytes, plus a `dpds` seek table of cumulative decoded byte
//! counts per packet.

use crate::formats::WaveFormatEx;
use crate::wmapro::DecodeError;
use crate::wmav2::WmaDecoder;
use crate::DecodedAudio;

pub const WMAV1_FORMAT_TAG: u16 = 0x0160;
pub const WMAV2_FORMAT_TAG: u16 = 0x0161;

type Result<T> = std::result::Result<T, DecodeError>;

/// Parsed xWMA file.
pub struct XwmaFile<'a> {
    pub wav_format: WaveFormatEx,
    /// Codec extradata following WAVEFORMATEX in the fmt chunk (usually empty
    /// in xWMA).
    pub extradata: &'a [u8],
    /// Cumulative decoded byte counts per packet (dpds chunk), if present.
    pub dpds: Vec<u32>,
    pub data: &'a [u8],
}

impl XwmaFile<'_> {
    /// Total decoded samples per channel promised by the dpds table.
    pub fn dpds_frame_count(&self) -> Option<usize> {
        let last = *self.dpds.last()?;
        let bytes_per_frame =
            self.wav_format.n_channels as usize * (self.wav_format.w_bits_per_sample as usize / 8);
        if bytes_per_frame == 0 {
            return None;
        }
        Some(last as usize / bytes_per_frame)
    }
}

fn rd32(b: &[u8], off: usize) -> Option<u32> {
    Some(u32::from_le_bytes(b.get(off..off + 4)?.try_into().ok()?))
}

/// Parse an xWMA file (RIFF form `XWMA`).
pub fn parse_xwma(buf: &[u8]) -> Result<XwmaFile<'_>> {
    if buf.len() < 12 || &buf[0..4] != b"RIFF" || &buf[8..12] != b"XWMA" {
        return Err(DecodeError("not a RIFF/XWMA file".into()));
    }
    let mut fmt: Option<&[u8]> = None;
    let mut data: Option<&[u8]> = None;
    let mut dpds = Vec::new();
    let mut off = 12usize;
    while off + 8 <= buf.len() {
        let id = &buf[off..off + 4];
        let size = rd32(buf, off + 4).ok_or_else(|| DecodeError("bad chunk".into()))? as usize;
        let body = buf
            .get(off + 8..off + 8 + size)
            .ok_or_else(|| DecodeError("truncated chunk".into()))?;
        match id {
            b"fmt " => fmt = Some(body),
            b"data" => data = Some(body),
            b"dpds" => {
                dpds = body
                    .chunks_exact(4)
                    .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
                    .collect();
            }
            _ => {} // PRIV and other game-specific chunks are ignored
        }
        off += 8 + size + (size & 1);
    }
    let fmt = fmt.ok_or_else(|| DecodeError("missing fmt chunk".into()))?;
    let data = data.ok_or_else(|| DecodeError("missing data chunk".into()))?;

    let wav_format = WaveFormatEx::from_le_bytes(fmt)
        .ok_or_else(|| DecodeError("fmt chunk too small".into()))?;
    let extradata_len = (wav_format.cb_size as usize).min(fmt.len() - WaveFormatEx::SIZE.min(fmt.len()));
    let extradata = &fmt[WaveFormatEx::SIZE.min(fmt.len())..][..extradata_len];

    Ok(XwmaFile { wav_format, extradata, dpds, data })
}

/// The xWMA encoder writes fake `nAvgBytesPerSec` values for many of its
/// (channels, sample rate) presets; the actual codec bitrate differs while the
/// packet data is identical. The decoder derives bitstream field widths from
/// the bitrate, so it must be normalized (FFmpeg does the same in xwma.c; the
/// 96 kbps mono 22050 Hz case below is seen in game rips but missing there).
fn normalize_bit_rate(channels: u16, sample_rate: u32, bit_rate: u32) -> u32 {
    match (channels, sample_rate, bit_rate) {
        (1, 22050, 48000 | 96000 | 192000) => 20000,
        (1, 32000, 48000 | 192000) => 20000,
        (1, 44100, 96000 | 192000) => 48000,
        (2, 22050, 48000 | 192000) => 32000,
        (2, 32000, 192000) => 48000,
        _ => bit_rate,
    }
}

/// Fabricated extradata for WMAv2 when the container carries none, matching
/// FFmpeg's xwma.c (flags2 = 31: exp VLC + bit reservoir + variable blocks).
const FAKE_WMAV2_EXTRADATA: [u8; 6] = [0, 0, 0, 0, 31, 0];

/// Decode a raw WMA v1/v2 packet stream described by a WAVEFORMATEX header to
/// planar f32. `frame_count` (from an xWMA dpds table or other metadata)
/// truncates the output to the declared length when given.
pub fn decode_wma_packets(
    wav_format: &WaveFormatEx,
    extradata: &[u8],
    data: &[u8],
    frame_count: Option<usize>,
) -> Result<DecodedAudio> {
    let f = wav_format;
    let version = match f.w_format_tag {
        WMAV1_FORMAT_TAG => 1u8,
        WMAV2_FORMAT_TAG => 2u8,
        tag => {
            return Err(DecodeError(format!(
                "unsupported wFormatTag {tag:#06x} (want 0x0160/0x0161 WMA)"
            )));
        }
    };
    let channels = f.n_channels as usize;
    let sample_rate = f.n_samples_per_sec;
    let block_align = f.n_block_align as usize;
    let bit_rate = normalize_bit_rate(f.n_channels, sample_rate, f.n_avg_bytes_per_sec * 8);

    let extradata: &[u8] = if extradata.is_empty() && version == 2 {
        &FAKE_WMAV2_EXTRADATA
    } else {
        extradata
    };

    let mut dec = WmaDecoder::new(
        version,
        channels,
        sample_rate,
        bit_rate,
        block_align,
        extradata,
    )?;

    let mut planes: Vec<Vec<f32>> = vec![Vec::new(); channels];
    for packet in data.chunks(block_align) {
        if packet.len() < block_align {
            break;
        }
        match dec.decode_superframe(packet) {
            Ok(samples) => {
                for (ch, s) in samples.into_iter().enumerate() {
                    planes[ch].extend_from_slice(&s);
                }
            }
            Err(_) => {
                // packet loss resilience: skip broken packets like FFmpeg's
                // CLI does (the reservoir was reset by the decoder)
            }
        }
    }
    // EOF flush: emit the delayed overlap tail
    if let Ok(samples) = dec.decode_superframe(&[]) {
        for (ch, s) in samples.into_iter().enumerate() {
            planes[ch].extend_from_slice(&s);
        }
    }

    // Decoder delay: the first 2 * frame_len output samples are leading
    // padding (FFmpeg sets avctx->delay accordingly); drop them, then honor
    // the declared length if present.
    let delay = 2 * dec.frame_len;
    for ch in planes.iter_mut() {
        if ch.len() >= delay {
            ch.drain(..delay);
        } else {
            ch.clear();
        }
    }
    if let Some(n) = frame_count {
        for ch in planes.iter_mut() {
            ch.truncate(n);
        }
    }

    Ok(DecodedAudio { sample_rate, channels, planes })
}

/// Decode a parsed xWMA file to planar f32.
pub fn decode_xwma(file: &XwmaFile) -> Result<DecodedAudio> {
    decode_wma_packets(
        &file.wav_format,
        file.extradata,
        file.data,
        file.dpds_frame_count(),
    )
}

/// Parse and decode a whole xWMA file to planar f32.
pub fn decode_xwma_file(buf: &[u8]) -> Result<DecodedAudio> {
    let file = parse_xwma(buf)?;
    decode_xwma(&file)
}
