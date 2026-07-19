# xma2-rs

A pure-Rust XMA2 and xWMA audio decoder. No FFmpeg, no C toolchain, no build
script — it's a plain `cargo add` away.

XMA2 is the Xbox 360's audio codec: a WMA Pro core wrapped in a packet/stream
layer. Decoding it normally means linking FFmpeg, which is a heavy dependency to
carry for one codec. This crate is a port of FFmpeg's `wmaprodec.c` plus the XMA
packet layer, trimmed to what XMA2 actually uses.

xWMA is XAudio2's other compressed-audio container (RIFF form `XWMA`): WMAv2
(or rarely WMAv1) superframes plus a `dpds` seek table. The `wmav2` module is a
port of FFmpeg's `wmadec.c`/`wma.c` covering both codec versions.

This exists because I wanted XMA2 support in another project without taking an
FFmpeg dependency. It's probably only useful to me, but if you've stumbled across
it and it helps, go crazy :)

## What works

- RIFF/WAVE files with `wFormatTag 0x0166` (`XMA2WAVEFORMATEX`), as produced by
  the Xbox 360 XMA2 encoder — see [`decode_xma2_file`].
- Raw XMA2 packet streams, when you already have the format headers from
  elsewhere — see [`Xma2Decoder`].
- Multi-stream files (XMA2 carries >2 channels as several 1–2 channel streams,
  decoded and recombined).
- xWMA files (`RIFF….XWMA`) carrying WMAv2 (`wFormatTag 0x0161`) or WMAv1
  (`0x0160`), 1–2 channels — see [`decode_xwma_file`]. Known-fake
  `nAvgBytesPerSec` values written by the xWMA encoder are normalized the same
  way FFmpeg's xwma demuxer does (plus the 96 kbps mono 22050 Hz case FFmpeg
  currently misses), since the decoder derives bitstream layout from the
  bitrate.

The WMA Pro core implements only the features reachable from XMA2 bitstreams:
1–2 channels per stream, 512 samples per frame, `decode_flags 0x10d6`. Anything
outside that returns a `DecodeError` rather than producing garbage — so a general
`.wma` file will not decode here. That's deliberate: the goal is XMA2, not a
complete WMA Pro implementation.

## Usage

The one-liner when you don't know (or care) which codec an asset uses —
dispatches on the RIFF form and `wFormatTag`:

```rust
let buf = std::fs::read("sound.wav")?;
let audio = xma2_rs::decode_audio_file(&buf)?; // XMA2, xWMA, or WMA-in-WAVE

println!("{} Hz, {} ch, {} frames", audio.sample_rate, audio.channels, audio.frames());
let pcm: Vec<i16> = audio.to_interleaved_i16();
```

Or per format, for a whole file in memory:

```rust
let buf = std::fs::read("sound.xma")?;
let audio = xma2_rs::decode_xma2_file(&buf)?;
```

If you need the header values (stream count, loop points, sample counts) before
committing to a decode, parse and decode in two steps:

```rust
let file = xma2_rs::parse_xma2_riff(&buf)?;
println!("{} streams @ {} Hz", file.num_streams(), file.sample_rate());

let audio = xma2_rs::decode_xma2(&file.wav_format, &file.xma2_format, file.data)?;
```

For data that isn't in a RIFF container — XMA2 lifted out of a game archive, say —
drive the decoder directly. You supply the parameters the container would have
carried:

```rust
use xma2_rs::Xma2Decoder;

let mut dec = Xma2Decoder::new(48_000, num_streams, total_channels)?;
let planes: Vec<Vec<f32>> = dec.decode(&packet_data)?;
```

xWMA works the same way — one call for a whole file, or parse first when you
want the header/dpds info:

```rust
let buf = std::fs::read("sound.xwma")?;
let audio = xma2_rs::decode_xwma_file(&buf)?;

// or:
let file = xma2_rs::parse_xwma(&buf)?;
let audio = xma2_rs::decode_xwma(&file)?;
```

## Output

Decoding produces planar `f32` — one `Vec<f32>` per channel, all the same length:

```rust
pub struct DecodedAudio {
    pub sample_rate: u32,
    pub channels: usize,
    pub planes: Vec<Vec<f32>>,
}
```

Planar is the codec's native shape, so it's what you get; `to_interleaved_i16`
and `to_interleaved_f32` are there when you need to hand samples to something
that wants them interleaved. `decode_xma2_to_pcm` goes straight to interleaved
i16 in one call.

## Layout

| Module       | What's in it                                                   |
|--------------|----------------------------------------------------------------|
| `xma`        | XMA2 packet/stream layer: packet framing, per-stream FIFOs, recombination |
| `wmapro`     | WMA Pro decoder core, ported from `wmaprodec.c`                 |
| `xwma`       | xWMA container parsing, bitrate normalization, decode driver    |
| `wmav2`      | WMA v1/v2 decoder core, ported from `wmadec.c`/`wma.c`          |
| `imdct`      | Inverse MDCT                                                    |
| `vlc`        | Variable-length code reader                                     |
| `bits`       | Bit-level reader/writer                                         |
| `tables`     | Static codec tables for WMA Pro (VLCs, quantisation, windows)   |
| `wma_tables` | Static codec tables for WMA v1/v2                               |
| `formats`    | `WAVEFORMATEX` / `XMA2WAVEFORMATEX` parsing                     |

## License

xma2-rs is LGPL-2.1-or-later licensed, as a port of FFmpeg's LGPL-licensed WMA
Pro and WMA v1/v2 decoders (`wmaprodec.c`, `wmaprodata.h`, `wmadec.c`, `wma.c`,
`wmadata.h`). Please refer to the LICENSE file for detailed information.
