//! Pure-Rust XMA2 and xWMA audio decoder (no FFmpeg required).
//!
//! Supports RIFF/WAVE files with wFormatTag 0x0166 (XMA2WAVEFORMATEX) as
//! produced by the Xbox 360 XMA2 encoder, raw XMA2 packet streams via
//! [`Xma2Decoder`], and xWMA containers (RIFF form `XWMA`, XAudio2) carrying
//! WMAv1/WMAv2 (wFormatTag 0x0160/0x0161) via [`decode_xwma_file`].

mod bits;
pub mod formats;
mod imdct;
mod tables;
mod vlc;
mod wma_tables;
pub mod wmapro;
pub mod wmav2;
pub mod xma;
pub mod xwma;

pub use formats::{WaveFormatEx, Xma2WaveFormatEx, XMA2_FORMAT_TAG};
pub use wmapro::DecodeError;
pub use wmav2::WmaDecoder;
pub use xma::Xma2Decoder;
pub use xwma::{decode_wma_packets, decode_xwma, decode_xwma_file, parse_xwma, XwmaFile};

/// Decoded audio.
pub struct DecodedAudio {
    pub sample_rate: u32,
    pub channels: usize,
    /// Planar f32 samples, `channels` vectors of equal length.
    pub planes: Vec<Vec<f32>>,
}

impl DecodedAudio {
    pub fn frames(&self) -> usize {
        self.planes.first().map_or(0, |p| p.len())
    }

    /// Interleave to i16 PCM.
    pub fn to_interleaved_i16(&self) -> Vec<i16> {
        let n = self.frames();
        let mut out = Vec::with_capacity(n * self.channels);
        for i in 0..n {
            for ch in &self.planes {
                let v = (ch[i] * 32768.0).round();
                out.push(v.clamp(-32768.0, 32767.0) as i16);
            }
        }
        out
    }

    /// Interleave to f32 PCM.
    pub fn to_interleaved_f32(&self) -> Vec<f32> {
        let n = self.frames();
        let mut out = Vec::with_capacity(n * self.channels);
        for i in 0..n {
            for ch in &self.planes {
                out.push(ch[i]);
            }
        }
        out
    }
}

/// Parsed XMA2 RIFF container.
pub struct Xma2File<'a> {
    pub wav_format: WaveFormatEx,
    pub xma2_format: Xma2WaveFormatEx,
    pub data: &'a [u8],
}

impl Xma2File<'_> {
    pub fn sample_rate(&self) -> u32 {
        self.wav_format.n_samples_per_sec
    }
    pub fn channels(&self) -> usize {
        self.wav_format.n_channels as usize
    }
    pub fn num_streams(&self) -> usize {
        self.xma2_format.num_streams as usize
    }
}

fn rd32(b: &[u8], off: usize) -> Option<u32> {
    Some(u32::from_le_bytes(b.get(off..off + 4)?.try_into().ok()?))
}

/// fmt and data chunk bodies of a RIFF/WAVE file.
fn parse_wave_chunks(buf: &[u8]) -> Result<(&[u8], &[u8]), DecodeError> {
    if buf.len() < 12 || &buf[0..4] != b"RIFF" || &buf[8..12] != b"WAVE" {
        return Err(DecodeError("not a RIFF/WAVE file".into()));
    }
    let mut fmt: Option<&[u8]> = None;
    let mut data: Option<&[u8]> = None;
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
            _ => {}
        }
        off += 8 + size + (size & 1);
    }
    let fmt = fmt.ok_or_else(|| DecodeError("missing fmt chunk".into()))?;
    let data = data.ok_or_else(|| DecodeError("missing data chunk".into()))?;
    Ok((fmt, data))
}

/// Parse an XMA2 RIFF/WAVE file (wFormatTag 0x0166).
pub fn parse_xma2_riff(buf: &[u8]) -> Result<Xma2File<'_>, DecodeError> {
    let (fmt, data) = parse_wave_chunks(buf)?;

    let wav_format = WaveFormatEx::from_le_bytes(fmt)
        .ok_or_else(|| DecodeError("fmt chunk too small".into()))?;
    if wav_format.w_format_tag != XMA2_FORMAT_TAG {
        let tag = wav_format.w_format_tag;
        return Err(DecodeError(format!("unsupported wFormatTag {tag:#06x} (want 0x0166 XMA2)")));
    }
    if (wav_format.cb_size as usize) < Xma2WaveFormatEx::SIZE {
        return Err(DecodeError("XMA2WAVEFORMATEX extradata missing".into()));
    }
    let xma2_format = Xma2WaveFormatEx::from_le_bytes(&fmt[WaveFormatEx::SIZE..])
        .ok_or_else(|| DecodeError("XMA2WAVEFORMATEX extradata truncated".into()))?;

    Ok(Xma2File { wav_format, xma2_format, data })
}

/// Decode raw XMA2 packet data described by wave-format headers to planar f32.
pub fn decode_xma2(
    wav_format: &WaveFormatEx,
    xma2_format: &Xma2WaveFormatEx,
    data: &[u8],
) -> Result<DecodedAudio, DecodeError> {
    let sample_rate = wav_format.n_samples_per_sec;
    let channels = wav_format.n_channels as usize;
    let num_streams = xma2_format.num_streams as usize;
    let mut dec = Xma2Decoder::new(sample_rate, num_streams, channels)?;
    let planes = dec.decode(data)?;
    Ok(DecodedAudio { sample_rate, channels, planes })
}

/// Decode raw XMA2 packet data to interleaved i16 PCM.
///
/// Float samples are converted with `(s * 32767).clamp(-32768, 32767) as i16`,
/// matching common FFmpeg-based conversion pipelines.
pub fn decode_xma2_to_pcm(
    wav_format: &WaveFormatEx,
    xma2_format: &Xma2WaveFormatEx,
    data: &[u8],
) -> Result<Vec<i16>, DecodeError> {
    let audio = decode_xma2(wav_format, xma2_format, data)?;
    let n = audio.frames();
    let mut out = Vec::with_capacity(n * audio.channels);
    for i in 0..n {
        for ch in &audio.planes {
            out.push((ch[i] * 32767.0).clamp(-32768.0, 32767.0) as i16);
        }
    }
    Ok(out)
}

/// Decode a whole XMA2 RIFF file to planar f32.
pub fn decode_xma2_file(buf: &[u8]) -> Result<DecodedAudio, DecodeError> {
    let file = parse_xma2_riff(buf)?;
    decode_xma2(&file.wav_format, &file.xma2_format, file.data)
}

/// Decode raw audio packets described by already-parsed format headers to
/// planar f32, dispatching on `wav_format.w_format_tag`. This is the
/// container-less counterpart of [`decode_audio_file`], for game assets that
/// carry a `WaveFormatEx` (and, for XMA2, an `Xma2WaveFormatEx`) in their own
/// structures rather than a RIFF file:
///
/// - tag 0x0166 (XMA2): `xma2_format` must be `Some`
/// - tag 0x0160/0x0161 (WMAv1/v2): `extradata` is the codec-private data if
///   the source has any (empty fabricates the standard xWMA layout), and the
///   fake-bitrate normalization is applied to `n_avg_bytes_per_sec`
///
/// `frame_count`, when known from asset metadata (e.g. an xWMA `dpds` table
/// or a seek table), truncates the output to the declared length.
pub fn decode_audio(
    wav_format: &WaveFormatEx,
    xma2_format: Option<&Xma2WaveFormatEx>,
    extradata: &[u8],
    data: &[u8],
    frame_count: Option<usize>,
) -> Result<DecodedAudio, DecodeError> {
    match wav_format.w_format_tag {
        XMA2_FORMAT_TAG => {
            let xma2_format = xma2_format.ok_or_else(|| {
                DecodeError("XMA2 (0x0166) requires an Xma2WaveFormatEx".into())
            })?;
            let mut audio = decode_xma2(wav_format, xma2_format, data)?;
            if let Some(n) = frame_count {
                for ch in audio.planes.iter_mut() {
                    ch.truncate(n);
                }
            }
            Ok(audio)
        }
        xwma::WMAV1_FORMAT_TAG | xwma::WMAV2_FORMAT_TAG => {
            xwma::decode_wma_packets(wav_format, extradata, data, frame_count)
        }
        tag => Err(DecodeError(format!(
            "unsupported wFormatTag {tag:#06x} (supported: 0x0160/0x0161 WMA, 0x0166 XMA2)"
        ))),
    }
}

/// Decode any supported RIFF audio file to planar f32, dispatching on the
/// container form and `wFormatTag`:
///
/// - `RIFF….XWMA` — xWMA container, WMAv1/WMAv2 (0x0160/0x0161)
/// - `RIFF….WAVE` with tag 0x0166 — XMA2
/// - `RIFF….WAVE` with tag 0x0160/0x0161 — WMA packets in a plain WAVE file
///
/// This is the "just decode it" entry point when the caller does not know
/// (or care) which codec a game asset uses.
pub fn decode_audio_file(buf: &[u8]) -> Result<DecodedAudio, DecodeError> {
    if buf.len() < 12 || &buf[0..4] != b"RIFF" {
        return Err(DecodeError("not a RIFF file".into()));
    }
    match &buf[8..12] {
        b"XWMA" => xwma::decode_xwma_file(buf),
        b"WAVE" => {
            let (fmt, data) = parse_wave_chunks(buf)?;
            // Plain PCM files may use the 16-byte pre-cbSize fmt layout;
            // pad so the tag still gets reported properly.
            let mut fmt_padded;
            let fmt = if fmt.len() >= WaveFormatEx::SIZE {
                fmt
            } else if fmt.len() >= 16 {
                fmt_padded = [0u8; WaveFormatEx::SIZE];
                fmt_padded[..fmt.len()].copy_from_slice(fmt);
                &fmt_padded[..]
            } else {
                return Err(DecodeError("fmt chunk too small".into()));
            };
            let wav_format = WaveFormatEx::from_le_bytes(fmt)
                .ok_or_else(|| DecodeError("fmt chunk too small".into()))?;
            match wav_format.w_format_tag {
                XMA2_FORMAT_TAG => decode_xma2_file(buf),
                xwma::WMAV1_FORMAT_TAG | xwma::WMAV2_FORMAT_TAG => {
                    let extradata_len =
                        (wav_format.cb_size as usize).min(fmt.len().saturating_sub(WaveFormatEx::SIZE));
                    let extradata = &fmt[WaveFormatEx::SIZE.min(fmt.len())..][..extradata_len];
                    xwma::decode_wma_packets(&wav_format, extradata, data, None)
                }
                tag => Err(DecodeError(format!(
                    "unsupported wFormatTag {tag:#06x} (supported: 0x0160/0x0161 WMA, 0x0166 XMA2)"
                ))),
            }
        }
        form => Err(DecodeError(format!(
            "unsupported RIFF form {:?} (want WAVE or XWMA)",
            String::from_utf8_lossy(form)
        ))),
    }
}
