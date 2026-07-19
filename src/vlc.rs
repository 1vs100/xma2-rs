//! Prefix-code (VLC) decoding compatible with FFmpeg's
//! `ff_vlc_init_from_lengths`: codes are assigned canonically in table order,
//! left-aligned and strictly increasing.

use crate::bits::BitReader;

pub struct Vlc {
    /// Left-aligned 32-bit code interval start per entry, sorted ascending.
    starts: Vec<u32>,
    lens: Vec<u8>,
    syms: Vec<i32>,
    /// Direct lookup for the first `LUT_BITS` bits: (symbol, len) when the
    /// code fits in LUT_BITS, otherwise len == 0 -> fall back to search.
    lut: Vec<(i32, u8)>,
}

const LUT_BITS: u32 = 10;

impl Vlc {
    /// Build from (symbol, length) pairs in FFmpeg table order.
    /// `offset` is added to each symbol (FFmpeg's `offset` parameter).
    pub fn from_lengths(entries: impl Iterator<Item = (i32, u8)>, offset: i32) -> Self {
        let mut starts = Vec::new();
        let mut lens = Vec::new();
        let mut syms = Vec::new();
        let mut code: u64 = 0;
        for (sym, len) in entries {
            assert!((1..=32).contains(&len));
            starts.push(code as u32);
            lens.push(len);
            syms.push(sym + offset);
            code += 1u64 << (32 - len);
        }
        assert_eq!(code, 1u64 << 32, "incomplete or oversubscribed code");

        let mut lut = vec![(0i32, 0u8); 1 << LUT_BITS];
        for i in 0..starts.len() {
            if (lens[i] as u32) <= LUT_BITS {
                let base = starts[i] >> (32 - LUT_BITS);
                let count = 1u32 << (LUT_BITS - lens[i] as u32);
                for j in 0..count {
                    lut[(base + j) as usize] = (syms[i], lens[i]);
                }
            }
        }
        Vlc {
            starts,
            lens,
            syms,
            lut,
        }
    }

    /// Build from explicit (code, length) pairs in FFmpeg table order, as
    /// `vlc_init` does with literal huffcodes; the symbol for entry `i` is
    /// `i as i32`. Codes are the MSB-first bit patterns of `length` bits.
    pub fn from_codes(codes: &[u32], lens: &[u8]) -> Self {
        assert_eq!(codes.len(), lens.len());
        let mut entries: Vec<(u32, u8, i32)> = codes
            .iter()
            .zip(lens.iter())
            .enumerate()
            .map(|(i, (&c, &l))| {
                assert!((1..=32).contains(&l));
                (c << (32 - l), l, i as i32)
            })
            .collect();
        entries.sort_by_key(|&(start, _, _)| start);

        let starts: Vec<u32> = entries.iter().map(|e| e.0).collect();
        let lens: Vec<u8> = entries.iter().map(|e| e.1).collect();
        let syms: Vec<i32> = entries.iter().map(|e| e.2).collect();

        let mut lut = vec![(0i32, 0u8); 1 << LUT_BITS];
        for i in 0..starts.len() {
            if (lens[i] as u32) <= LUT_BITS {
                let base = starts[i] >> (32 - LUT_BITS);
                let count = 1u32 << (LUT_BITS - lens[i] as u32);
                for j in 0..count {
                    lut[(base + j) as usize] = (syms[i], lens[i]);
                }
            }
        }
        Vlc { starts, lens, syms, lut }
    }

    /// Decode one symbol, consuming its bits.
    #[inline]
    pub fn get(&self, br: &mut BitReader) -> i32 {
        let peek = br.show_bits(32);
        let (sym, len) = self.lut[(peek >> (32 - LUT_BITS)) as usize];
        if len != 0 {
            br.skip(len as usize);
            return sym;
        }
        // Binary search: last interval start <= peek.
        let idx = match self.starts.binary_search(&peek) {
            Ok(i) => i,
            Err(i) => i - 1,
        };
        br.skip(self.lens[idx] as usize);
        self.syms[idx]
    }
}
