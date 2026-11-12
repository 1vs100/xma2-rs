//! XMA2 encoding: WMA Pro frames from wmv9's encoder in XMA2 packets.

use crate::pack::{self, PackError, Packed};
use std::fmt;
use wmv9::wma::pro::encode::{EncodeError as CoreError, Encoder};
use wmv9::wma::{ByteRate, Channels, Format, SampleRate};

/// Samples per channel Microsoft's encoder delays its output by, and XMA
/// decoders drop.
const ENCODER_DELAY: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncodeError {
    /// XMA streams carry one or two channels.
    Channels {
        count: usize,
    },
    Core(CoreError),
    Pack(PackError),
}

impl fmt::Display for EncodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EncodeError::Channels { count } => {
                write!(f, "{count} channels; XMA streams carry one or two")
            }
            EncodeError::Core(e) => e.fmt(f),
            EncodeError::Pack(e) => e.fmt(f),
        }
    }
}

impl std::error::Error for EncodeError {}

/// Encodes interleaved 16-bit samples of one or two channels at `byte_rate`
/// bytes per second of coded audio.
pub fn encode(
    pcm: &[i16],
    sample_rate: u32,
    channels: usize,
    byte_rate: u32,
) -> Result<Packed, EncodeError> {
    let channels = match channels {
        1 => Channels::Mono,
        2 => Channels::Stereo,
        count => return Err(EncodeError::Channels { count }),
    };
    let mut encoder = Encoder::xma(Format {
        sample_rate: SampleRate(sample_rate),
        channels,
        byte_rate: ByteRate(byte_rate),
    })
    .map_err(EncodeError::Core)?;
    encoder
        .push(&vec![0; ENCODER_DELAY * channels.count()])
        .map_err(EncodeError::Core)?;
    encoder.push(pcm).map_err(EncodeError::Core)?;
    let frames = encoder.finish_frames().map_err(EncodeError::Core)?;
    pack::pack(&frames).map_err(EncodeError::Pack)
}
