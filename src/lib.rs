//! Pure-Rust XMA2 audio decoder (no FFmpeg required).
//!
//! Supports RIFF/WAVE files with wFormatTag 0x0166 (XMA2WAVEFORMATEX) as
//! produced by the Xbox 360 XMA2 encoder, and raw XMA2 packet streams via
//! [`Xma2Decoder`].

mod bits;
pub mod formats;
mod imdct;
mod tables;
mod vlc;
pub mod wmapro;
pub mod xma;

pub use formats::{WaveFormatEx, Xma2WaveFormatEx, XMA2_FORMAT_TAG};
pub use wmapro::DecodeError;
pub use xma::Xma2Decoder;

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

/// Parse an XMA2 RIFF/WAVE file (wFormatTag 0x0166).
pub fn parse_xma2_riff(buf: &[u8]) -> Result<Xma2File<'_>, DecodeError> {
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
