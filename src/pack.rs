//! XMA2 packets from coded WMA Pro frames, laid out as Microsoft's encoder
//! lays them out: frames stream across packets of 2048 bytes, blocks of 32
//! packets start with a frame, and padding is 1 bits.

use std::fmt;

const PACKET_BYTES: usize = 2048;
const HEADER_BITS: usize = 32;
const PAYLOAD_BITS: usize = PACKET_BYTES * 8 - HEADER_BITS;
const PACKETS_PER_BLOCK: usize = 32;
const BLOCK_BITS: usize = PACKETS_PER_BLOCK * PAYLOAD_BITS;
/// The packet header's count of frames that start in it has 6 bits.
const MAX_FRAMES_PER_PACKET: usize = 63;
/// The 3-bit metadata field of every packet header in the game's streams.
const METADATA: u32 = 1;
pub const FRAME_SAMPLES: u32 = 512;

/// Samples per channel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Samples(pub u32);

/// A packed stream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Packed {
    /// Whole packets.
    pub data: Vec<u8>,
    /// Per block, the samples of the stream up to its end.
    pub seek_table: Vec<Samples>,
    /// 512 per frame, the first frame included, as XACT declares it.
    pub samples: Samples,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PackError {
    /// A frame does not fit one packet's payload.
    FrameTooLong {
        index: usize,
        bits: usize,
    },
    NoFrames,
}

impl fmt::Display for PackError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PackError::FrameTooLong { index, bits } => write!(
                f,
                "frame {index} has {bits} bits, more than a packet's {PAYLOAD_BITS}"
            ),
            PackError::NoFrames => f.write_str("no frames to pack"),
        }
    }
}

impl std::error::Error for PackError {}

/// A block's payload bits and where its frames start.
#[derive(Default)]
struct Block {
    bits: Vec<bool>,
    starts: Vec<usize>,
}

impl Block {
    fn starts_in(&self, packet: usize) -> usize {
        self.starts
            .iter()
            .filter(|&&s| s / PAYLOAD_BITS == packet)
            .count()
    }

    /// Sets each frame's trailer bit: 1 when the next frame follows it in
    /// the same packet.
    fn set_trailers(&mut self, ends: &[usize]) {
        for (i, &end) in ends.iter().enumerate() {
            self.bits[end - 1] = end % PAYLOAD_BITS != 0 && self.starts.get(i + 1) == Some(&end);
        }
    }

    fn write(&self, out: &mut Vec<u8>) {
        let packets = self.bits.len().div_ceil(PAYLOAD_BITS);
        for p in 0..packets {
            let from = p * PAYLOAD_BITS;
            let first = self
                .starts
                .iter()
                .copied()
                .find(|&s| s >= from)
                .unwrap_or(self.bits.len());
            let previous = (first - from).min(PAYLOAD_BITS);
            let header = (self.starts_in(p) as u32) << 26 | (previous as u32) << 11 | METADATA << 8;
            let mut bits: Vec<bool> = (0..HEADER_BITS)
                .rev()
                .map(|b| header >> b & 1 == 1)
                .collect();
            bits.extend(self.bits.iter().skip(from).take(PAYLOAD_BITS));
            bits.resize(PACKET_BYTES * 8, true);
            out.extend(
                bits.chunks(8)
                    .map(|byte| byte.iter().fold(0u8, |acc, &b| acc << 1 | u8::from(b))),
            );
        }
    }
}

/// Packs frames, each from its length prefix to its trailer bit, which is
/// set here.
pub fn pack(frames: &[Vec<bool>]) -> Result<Packed, PackError> {
    if frames.is_empty() {
        return Err(PackError::NoFrames);
    }
    let mut data = Vec::new();
    let mut seek_table = Vec::new();
    let mut block = Block::default();
    let mut ends = Vec::new();
    let mut done = 0u32;
    for (index, frame) in frames.iter().enumerate() {
        if frame.len() > PAYLOAD_BITS {
            return Err(PackError::FrameTooLong {
                index,
                bits: frame.len(),
            });
        }
        let mut at = block.bits.len();
        let packet = at / PAYLOAD_BITS;
        if block.starts_in(packet) == MAX_FRAMES_PER_PACKET {
            at = (packet + 1) * PAYLOAD_BITS;
        }
        if at + frame.len() > BLOCK_BITS {
            block.set_trailers(&ends);
            block.write(&mut data);
            done += block.starts.len() as u32;
            seek_table.push(Samples(done * FRAME_SAMPLES));
            (block, ends, at) = (Block::default(), Vec::new(), 0);
        }
        block.bits.resize(at, true);
        block.starts.push(at);
        block.bits.extend_from_slice(frame);
        ends.push(block.bits.len());
    }
    block.set_trailers(&ends);
    block.write(&mut data);
    done += block.starts.len() as u32;
    seek_table.push(Samples(done * FRAME_SAMPLES));
    Ok(Packed {
        data,
        seek_table,
        samples: Samples(done * FRAME_SAMPLES),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, PartialEq, Eq)]
    struct Header {
        frames: u32,
        previous: u32,
        metadata: u32,
        skip: u32,
    }

    fn header(data: &[u8], packet: usize) -> Header {
        let h = u32::from_be_bytes(data[packet * PACKET_BYTES..][..4].try_into().unwrap());
        Header {
            frames: h >> 26,
            previous: h >> 11 & 0x7fff,
            metadata: h >> 8 & 7,
            skip: h & 0xff,
        }
    }

    fn starts(frames: u32, previous: u32) -> Header {
        Header {
            frames,
            previous,
            metadata: METADATA,
            skip: 0,
        }
    }

    fn frame(bits: usize) -> Vec<bool> {
        (0..bits).map(|i| i % 3 == 0).collect()
    }

    #[test]
    fn frames_stream_across_packets_and_the_header_counts_the_carried_bits() {
        let packed = pack(&vec![frame(5000); 5]).unwrap();
        assert_eq!(packed.data.len(), 2 * PACKET_BYTES);
        // Frames start at 0, 5000, 10000 and 15000; the fourth ends at 20000.
        assert_eq!(header(&packed.data, 0), starts(4, 0));
        assert_eq!(
            header(&packed.data, 1),
            starts(1, (20000 - PAYLOAD_BITS) as u32)
        );
        assert_eq!(packed.samples, Samples(5 * 512));
        assert_eq!(packed.seek_table, vec![Samples(5 * 512)]);
    }

    #[test]
    fn trailers_mark_a_following_frame_and_padding_is_ones() {
        let packed = pack(&vec![frame(100); 2]).unwrap();
        let bit = |i: usize| packed.data[4 + i / 8] >> (7 - i % 8) & 1 == 1;
        assert!(bit(99), "the first frame is followed by another");
        assert!(!bit(199), "the last frame is not");
        assert!(
            packed.data[4 + 200 / 8 + 1..PACKET_BYTES]
                .iter()
                .all(|&b| b == 0xff)
        );
    }

    #[test]
    fn a_frame_that_would_cross_a_block_starts_the_next_block() {
        // 47 frames of 2/3 packet fill 31.3 packets, so a frame of a packet would
        // cross from packet 31 into the next block.
        let mut frames = vec![frame(PAYLOAD_BITS * 2 / 3); 47];
        frames.push(frame(PAYLOAD_BITS));
        let packed = pack(&frames).unwrap();
        assert_eq!(packed.data.len(), (PACKETS_PER_BLOCK + 1) * PACKET_BYTES);
        assert_eq!(header(&packed.data, PACKETS_PER_BLOCK), starts(1, 0));
        assert_eq!(
            packed.seek_table,
            vec![Samples(47 * 512), Samples(48 * 512)]
        );
    }

    #[test]
    fn a_frame_ending_at_a_packet_end_has_trailer_0() {
        // Decoders take a trailer of 1 to mean the next frame follows in the
        // same packet, and would read the next header as its length.
        let packed = pack(&[frame(PAYLOAD_BITS), frame(100)]).unwrap();
        let last = PACKET_BYTES - 1;
        assert_eq!(packed.data[last] & 1, 0);
        assert_eq!(header(&packed.data, 1), starts(1, 0));
    }

    #[test]
    fn a_packet_starts_at_most_63_frames() {
        let packed = pack(&vec![frame(40); 64]).unwrap();
        assert_eq!(header(&packed.data, 0).frames, 63);
        assert_eq!(header(&packed.data, 1), starts(1, 0));
    }

    #[test]
    fn a_frame_longer_than_a_packet_is_refused() {
        assert_eq!(
            pack(&[frame(PAYLOAD_BITS + 1)]),
            Err(PackError::FrameTooLong {
                index: 0,
                bits: PAYLOAD_BITS + 1
            })
        );
    }
}
