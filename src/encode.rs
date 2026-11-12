//! XMA2 encoding: WMA Pro frames from wmv9's encoder in XMA2 packets.

use crate::pack::{self, PackError, Packed};
use std::fmt;
use wmv9::wma::Format;
use wmv9::wma::pro::encode::{EncodeError as CoreError, Encoder};
pub use wmv9::wma::{ByteRate, Channels, SampleRate};

/// Samples per channel Microsoft's encoder delays its output by, and XMA
/// decoders drop.
const ENCODER_DELAY: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncodeError {
    Core(CoreError),
    Pack(PackError),
}

impl fmt::Display for EncodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EncodeError::Core(e) => e.fmt(f),
            EncodeError::Pack(e) => e.fmt(f),
        }
    }
}

impl std::error::Error for EncodeError {}

/// Encodes interleaved 16-bit samples at `byte_rate`
/// bytes per second of coded audio.
pub fn encode(
    pcm: &[i16],
    sample_rate: SampleRate,
    channels: Channels,
    byte_rate: ByteRate,
) -> Result<Packed, EncodeError> {
    let mut encoder = Encoder::xma(Format {
        sample_rate,
        channels,
        byte_rate,
    })
    .map_err(EncodeError::Core)?;
    encoder
        .push(&vec![0; ENCODER_DELAY * channels.count()])
        .map_err(EncodeError::Core)?;
    encoder.push(pcm).map_err(EncodeError::Core)?;
    let frames = encoder.finish_frames().map_err(EncodeError::Core)?;
    pack::pack(&frames).map_err(EncodeError::Pack)
}
