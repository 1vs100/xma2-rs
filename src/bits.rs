//! MSB-first bit reader matching FFmpeg's GetBitContext semantics.
//!
//! Reads past the end of the buffer return zero bits; the caller is expected
//! to check `bits_left` where FFmpeg does.

pub struct BitReader<'a> {
    buf: &'a [u8],
    /// Bit position from the start of the buffer.
    pos: usize,
    /// Total number of valid bits.
    size_bits: usize,
}

impl<'a> BitReader<'a> {
    pub fn new(buf: &'a [u8]) -> Self {
        BitReader {
            buf,
            pos: 0,
            size_bits: buf.len() * 8,
        }
    }

    /// Bit reader over a buffer with an explicit bit size (may be smaller
    /// than the byte buffer).
    pub fn with_bit_size(buf: &'a [u8], size_bits: usize) -> Self {
        BitReader { buf, pos: 0, size_bits }
    }

    #[inline]
    pub fn bits_count(&self) -> usize {
        self.pos
    }

    #[inline]
    pub fn bits_left(&self) -> i64 {
        self.size_bits as i64 - self.pos as i64
    }

    /// Peek up to 32 bits without advancing. Bits past the end read as 0.
    #[inline]
    pub fn show_bits(&self, n: u32) -> u32 {
        debug_assert!(n <= 32);
        if n == 0 {
            return 0;
        }
        let byte = self.pos >> 3;
        let bit = self.pos & 7;
        // Gather 8 bytes so any (bit, n<=32) combination fits.
        let mut v: u64 = 0;
        for i in 0..8 {
            let b = self.buf.get(byte + i).copied().unwrap_or(0);
            v = (v << 8) | b as u64;
        }
        ((v << bit) >> (64 - n)) as u32
    }

    #[inline]
    pub fn skip(&mut self, n: usize) {
        self.pos += n;
    }

    #[inline]
    pub fn get_bits(&mut self, n: u32) -> u32 {
        let v = self.show_bits(n);
        self.pos += n as usize;
        v
    }

    #[inline]
    pub fn get_bits1(&mut self) -> bool {
        self.get_bits(1) != 0
    }

    /// Signed read (two's complement of n bits), n >= 1.
    #[inline]
    pub fn get_sbits(&mut self, n: u32) -> i32 {
        let v = self.get_bits(n) as i32;
        // sign extend
        (v << (32 - n)) >> (32 - n)
    }
}

/// MSB-first bit writer used for the frame bit reservoir
/// (FFmpeg PutBitContext equivalent, only what save_bits needs).
pub struct BitWriter {
    pub buf: Vec<u8>,
    bit_pos: usize,
}

impl BitWriter {
    pub fn new() -> Self {
        BitWriter { buf: Vec::new(), bit_pos: 0 }
    }

    pub fn reset(&mut self) {
        self.buf.clear();
        self.bit_pos = 0;
    }

    #[inline]
    pub fn bits_count(&self) -> usize {
        self.bit_pos
    }

    pub fn put_bits(&mut self, n: u32, value: u32) {
        for i in (0..n).rev() {
            let bit = (value >> i) & 1;
            let byte = self.bit_pos >> 3;
            if byte >= self.buf.len() {
                self.buf.push(0);
            }
            self.buf[byte] |= (bit as u8) << (7 - (self.bit_pos & 7));
            self.bit_pos += 1;
        }
    }

    /// Append `n_bits` bits read from `src` starting at bit offset `src_bit`.
    pub fn copy_bits(&mut self, src: &[u8], src_bit: usize, n_bits: usize) {
        let mut remaining = n_bits;
        let mut pos = src_bit;
        // byte-align the writer fast path is unnecessary; simple loop in
        // 24-bit chunks is plenty for 2KB packets.
        while remaining > 0 {
            let chunk = remaining.min(24) as u32;
            let byte = pos >> 3;
            let bit = pos & 7;
            let mut v: u64 = 0;
            for i in 0..5 {
                let b = src.get(byte + i).copied().unwrap_or(0);
                v = (v << 8) | b as u64;
            }
            let val = ((v << bit) >> (40 - chunk)) as u32;
            self.put_bits(chunk, val);
            pos += chunk as usize;
            remaining -= chunk as usize;
        }
    }
}
