//! Windows wave-format structures for XMA2 (WAVEFORMATEX with
//! wFormatTag 0x0166 followed by the XMA2WAVEFORMATEX extension).

/// WAVEFORMATEX (18 bytes, little-endian on disk).
#[repr(C, packed(2))]
#[derive(Debug, Clone, Copy, Default)]
pub struct WaveFormatEx {
    pub w_format_tag: u16,
    pub n_channels: u16,
    pub n_samples_per_sec: u32,
    pub n_avg_bytes_per_sec: u32,
    pub n_block_align: u16,
    pub w_bits_per_sample: u16,
    pub cb_size: u16,
}

/// XMA2WAVEFORMATEX extension (34 bytes following WAVEFORMATEX).
#[repr(C, packed(2))]
#[derive(Debug, Clone, Copy, Default)]
pub struct Xma2WaveFormatEx {
    pub num_streams: u16,
    pub channel_mask: u32,
    pub samples_encoded: u32,
    pub bytes_per_block: u32,
    pub play_begin: u32,
    pub play_length: u32,
    pub loop_begin: u32,
    pub loop_length: u32,
    pub loop_count: u8,
    pub encoder_version: u8,
    pub block_count: u16,
}

pub const XMA2_FORMAT_TAG: u16 = 0x0166;

impl WaveFormatEx {
    pub const SIZE: usize = 18;

    pub fn from_le_bytes(b: &[u8]) -> Option<Self> {
        if b.len() < Self::SIZE {
            return None;
        }
        let u16le = |o: usize| u16::from_le_bytes([b[o], b[o + 1]]);
        let u32le = |o: usize| u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
        Some(WaveFormatEx {
            w_format_tag: u16le(0),
            n_channels: u16le(2),
            n_samples_per_sec: u32le(4),
            n_avg_bytes_per_sec: u32le(8),
            n_block_align: u16le(12),
            w_bits_per_sample: u16le(14),
            cb_size: u16le(16),
        })
    }

    pub fn swap_bytes(&self) -> Self {
        WaveFormatEx {
            w_format_tag: self.w_format_tag.swap_bytes(),
            n_channels: self.n_channels.swap_bytes(),
            n_samples_per_sec: self.n_samples_per_sec.swap_bytes(),
            n_avg_bytes_per_sec: self.n_avg_bytes_per_sec.swap_bytes(),
            n_block_align: self.n_block_align.swap_bytes(),
            w_bits_per_sample: self.w_bits_per_sample.swap_bytes(),
            cb_size: self.cb_size.swap_bytes(),
        }
    }
}

impl Xma2WaveFormatEx {
    pub const SIZE: usize = 34;

    pub fn from_le_bytes(b: &[u8]) -> Option<Self> {
        if b.len() < Self::SIZE {
            return None;
        }
        let u16le = |o: usize| u16::from_le_bytes([b[o], b[o + 1]]);
        let u32le = |o: usize| u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
        Some(Xma2WaveFormatEx {
            num_streams: u16le(0),
            channel_mask: u32le(2),
            samples_encoded: u32le(6),
            bytes_per_block: u32le(10),
            play_begin: u32le(14),
            play_length: u32le(18),
            loop_begin: u32le(22),
            loop_length: u32le(26),
            loop_count: b[30],
            encoder_version: b[31],
            block_count: u16le(32),
        })
    }

    pub fn swap_bytes(&self) -> Self {
        Xma2WaveFormatEx {
            num_streams: self.num_streams.swap_bytes(),
            channel_mask: self.channel_mask.swap_bytes(),
            samples_encoded: self.samples_encoded.swap_bytes(),
            bytes_per_block: self.bytes_per_block.swap_bytes(),
            play_begin: self.play_begin.swap_bytes(),
            play_length: self.play_length.swap_bytes(),
            loop_begin: self.loop_begin.swap_bytes(),
            loop_length: self.loop_length.swap_bytes(),
            loop_count: self.loop_count,
            encoder_version: self.encoder_version,
            block_count: self.block_count.swap_bytes(),
        }
    }

    /// Total PCM frames (samples per channel) to play, honoring
    /// PlayBegin/PlayLength.
    pub fn pcm_frame_count(&self) -> u32 {
        let play_length = self.play_length;
        if play_length != 0 {
            play_length
        } else {
            let encoded = self.samples_encoded;
            encoded.saturating_sub(self.play_begin)
        }
    }
}
