//! Half-length inverse MDCT matching FFmpeg's AV_TX_FLOAT_MDCT inverse
//! transform (see ff_tx_mdct_naive_inv in libavutil/tx_template.c):
//!
//! for i in 0..N/2:
//!   dst[i]       =  scale * sum_j src[j] * cos((2j+1)(2N-2i-1) * pi/(4N))
//!   dst[i + N/2] = -scale * sum_j src[j] * cos((2j+1)(3N+2i+1) * pi/(4N))
//!
//! where N coefficients produce N output samples (the non-redundant half of
//! the 2N-sample window).

use std::collections::HashMap;
use std::f64::consts::PI;

pub struct Imdct {
    /// cos(k*pi/(4N)) for k in 0..8N, keyed by N.
    cos_tabs: HashMap<usize, Vec<f64>>,
}

impl Imdct {
    pub fn new() -> Self {
        Imdct { cos_tabs: HashMap::new() }
    }

    fn tab(&mut self, n: usize) -> &Vec<f64> {
        self.cos_tabs.entry(n).or_insert_with(|| {
            (0..8 * n)
                .map(|k| (k as f64 * PI / (4.0 * n as f64)).cos())
                .collect()
        })
    }

    /// `src.len() == dst.len() == n`, n a power of two.
    pub fn imdct_half(&mut self, src: &[f32], dst: &mut [f32], scale: f64) {
        let n = src.len();
        debug_assert_eq!(dst.len(), n);
        let tab = self.tab(n);
        let period = 8 * n;
        let half = n / 2;
        for i in 0..half {
            let i_d = 2 * n - 2 * i - 1;
            let i_u = 3 * n + 2 * i + 1;
            let mut sum_d = 0.0f64;
            let mut sum_u = 0.0f64;
            let mut k_d = i_d % period;
            let mut k_u = i_u % period;
            let step_d = (2 * i_d) % period;
            let step_u = (2 * i_u) % period;
            for &s in &src[..n] {
                let val = s as f64;
                sum_d += tab[k_d] * val;
                sum_u += tab[k_u] * val;
                k_d += step_d;
                if k_d >= period {
                    k_d -= period;
                }
                k_u += step_u;
                if k_u >= period {
                    k_u -= period;
                }
            }
            dst[i] = (sum_d * scale) as f32;
            dst[i + half] = (-sum_u * scale) as f32;
        }
    }
}
