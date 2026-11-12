//! Encoded XMA2 streams decode to the input.
#![cfg(feature = "encode")]

use xma2_rs::Xma2Decoder;
use xma2_rs::encode::{ByteRate, Channels, SampleRate, encode};

fn signal(rate: u32, channels: usize, samples: usize) -> Vec<i16> {
    let tau = std::f64::consts::TAU;
    (0..samples)
        .flat_map(|i| {
            let t = i as f64 / f64::from(rate);
            (0..channels).map(move |c| {
                (9000.0 * (tau * (440.0 + 220.0 * c as f64) * t).sin()
                    + 3000.0 * (tau * 2500.0 * t).sin()) as i16
            })
        })
        .collect()
}

/// Signal to noise ratio of `decoded` against `source` from `lag` samples on, in decibels.
fn snr(source: &[f32], decoded: &[f32], lag: i64) -> f64 {
    let (mut signal, mut noise) = (0.0, 0.0);
    for (i, &s) in source.iter().enumerate() {
        let Some(&d) = usize::try_from(i as i64 + lag)
            .ok()
            .and_then(|j| decoded.get(j))
        else {
            continue;
        };
        signal += f64::from(s).powi(2);
        noise += f64::from(s - d).powi(2);
    }
    10.0 * (signal / noise).log10()
}

#[test]
fn encoded_streams_decode_to_the_input() {
    for (rate, channels, seconds) in [(48000, 1, 3), (44100, 2, 3), (48000, 1, 10)] {
        let samples = seconds * rate as usize;
        let pcm = signal(rate, channels, samples);
        let layout = if channels == 1 {
            Channels::Mono
        } else {
            Channels::Stereo
        };
        let packed = encode(&pcm, SampleRate(rate), layout, ByteRate(16000)).unwrap();
        assert_eq!(
            packed.seek_table.len(),
            packed.data.len().div_ceil(32 * 2048)
        );
        assert_eq!(
            packed.samples.0 as usize,
            (samples + 64).div_ceil(512) * 512 + 512
        );
        let decoded = Xma2Decoder::new(rate, 1, channels)
            .unwrap()
            .decode(&packed.data)
            .unwrap();
        for (c, plane) in decoded.iter().enumerate() {
            let source: Vec<f32> = pcm
                .iter()
                .skip(c)
                .step_by(channels)
                .map(|&s| f32::from(s) / 32768.0)
                .collect();
            let (lag, best) = (-128..=128)
                .map(|lag| (lag, snr(&source, plane, lag)))
                .max_by(|a, b| a.1.total_cmp(&b.1))
                .unwrap();
            println!(
                "{rate} Hz channel {c}: {} decoded of {samples}, lag {lag}, {best:.1} dB",
                plane.len()
            );
            assert_eq!(lag, 0);
            assert!(best > 40.0, "{best} dB");
            assert!(plane.len() >= samples - 512, "{}", plane.len());
        }
    }
}

/// Frames that end exactly at a packet end occur at some rates and must not
/// lose the frames after them.
#[test]
fn streams_decode_whole_at_every_rate() {
    let rate = 48000;
    let mut seed = 1u32;
    let pcm: Vec<i16> = signal(rate, 1, 4 * rate as usize)
        .into_iter()
        .map(|s| {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            s / 2 + (seed >> 20) as i16 - 2048
        })
        .collect();
    let source: Vec<f32> = pcm.iter().map(|&s| f32::from(s) / 32768.0).collect();
    for byte_rate in (4000..=40000).step_by(250) {
        let packed = encode(&pcm, SampleRate(rate), Channels::Mono, ByteRate(byte_rate)).unwrap();
        let decoded = Xma2Decoder::new(rate, 1, 1)
            .unwrap()
            .decode(&packed.data)
            .unwrap();
        assert!(
            decoded[0].len() >= pcm.len(),
            "{byte_rate} B/s: {} of {}",
            decoded[0].len(),
            pcm.len()
        );
        let db = snr(&source, &decoded[0], 0);
        assert!(db > 5.0, "{byte_rate} B/s: {db:.1} dB");
    }
}
