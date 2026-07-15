//! XMA2 multi-stream packet scheduling and sample interleaving, ported from
//! FFmpeg's xma_decode_packet (wmaprodec.c).

use crate::wmapro::{DecodeError, WmaProDecoder};

pub const XMA_MAX_STREAMS: usize = 8;
const PACKET_SIZE: usize = 2048;

struct Fifo {
    data: Vec<f32>,
    read: usize,
}

impl Fifo {
    fn new() -> Self {
        Fifo { data: Vec::new(), read: 0 }
    }
    fn len(&self) -> usize {
        self.data.len() - self.read
    }
    fn write(&mut self, samples: &[f32]) {
        self.data.extend_from_slice(samples);
    }
    fn read(&mut self, n: usize) -> &[f32] {
        let s = &self.data[self.read..self.read + n];
        self.read += n;
        s
    }
}

pub struct Xma2Decoder {
    streams: Vec<WmaProDecoder>,
    start_channel: Vec<usize>,
    num_channels: usize,
    /// per stream, [left, right] sample fifos
    fifo: Vec<[Fifo; 2]>,
    started: Vec<bool>,
    current_stream: usize,
    trim_start: usize,
    trim_end: usize,
    flushed: bool,
}

impl Xma2Decoder {
    pub fn new(sample_rate: u32, num_streams: usize, total_channels: usize) -> Result<Self, DecodeError> {
        if num_streams == 0 || num_streams > XMA_MAX_STREAMS {
            return Err(DecodeError(format!("bad stream count {num_streams}")));
        }
        let mut streams = Vec::new();
        let mut start_channel = Vec::new();
        let mut start = 0usize;
        for i in 0..num_streams {
            let s = WmaProDecoder::new_xma2(sample_rate, i, total_channels)?;
            start_channel.push(start);
            start += s.nb_channels;
            streams.push(s);
        }
        if start != total_channels {
            return Err(DecodeError(format!(
                "stream channels {start} != total channels {total_channels}"
            )));
        }
        Ok(Xma2Decoder {
            streams,
            start_channel,
            num_channels: total_channels,
            fifo: (0..num_streams).map(|_| [Fifo::new(), Fifo::new()]).collect(),
            started: vec![false; num_streams],
            current_stream: 0,
            trim_start: 0,
            trim_end: 0,
            flushed: false,
        })
    }

    /// Decode a whole XMA2 data chunk to planar f32 channels.
    pub fn decode(&mut self, data: &[u8]) -> Result<Vec<Vec<f32>>, DecodeError> {
        let mut out: Vec<Vec<f32>> = vec![Vec::new(); self.num_channels];

        for packet in data.chunks(PACKET_SIZE) {
            if packet.len() < 4 {
                break;
            }
            let mut off = 0usize;
            while off < packet.len() {
                let consumed = self.step(&packet[off..], false, &mut out);
                if consumed == 0 {
                    break;
                }
                off += consumed;
            }
        }

        // EOF flush: mirror repeated empty packets until flushed
        let mut guard = 0;
        while !self.flushed && guard < 4 * XMA_MAX_STREAMS {
            self.step(&[], true, &mut out);
            guard += 1;
        }

        // XMA encoder delay: skip_samples = 64
        for ch in out.iter_mut() {
            if ch.len() >= 64 {
                ch.drain(..64);
            } else {
                ch.clear();
            }
        }
        Ok(out)
    }

    /// One xma_decode_packet call. Returns consumed bytes.
    fn step(&mut self, data: &[u8], eof_input: bool, out: &mut [Vec<f32>]) -> usize {
        let cs = self.current_stream;
        self.started[cs] = true;

        let mut got_frame: Option<Vec<Vec<f32>>> = None;
        let mut consumed = 0usize;

        if !self.streams[cs].eof_done {
            let (c, frame) = self.streams[cs].decode_packet(data);
            consumed = c;
            got_frame = frame;
        }

        let mut eof = false;
        if eof_input {
            eof = true;
            for i in 0..self.streams.len() {
                if !self.streams[i].eof_done && self.started[i] {
                    let (_, frame) = self.streams[i].decode_packet(&[]);
                    if i == cs && frame.is_some() {
                        got_frame = frame;
                    } else if let Some(f) = frame {
                        // FFmpeg drops non-current streams' flush output; we
                        // keep behaviour identical for byte-exact parity.
                        let _ = f;
                    }
                }
                eof &= self.streams[i].eof_done;
            }
        }

        if self.streams[0].trim_start != 0 {
            self.trim_start = self.streams[0].trim_start;
        }
        if self.streams[0].trim_end != 0 {
            self.trim_end = self.streams[0].trim_end;
        }

        // copy stream samples to fifo
        if let Some(frame) = got_frame {
            self.fifo[cs][0].write(&frame[0]);
            if self.streams[cs].nb_channels > 1 {
                self.fifo[cs][1].write(&frame[1]);
            }
        }

        // find next packet's owner stream and possibly emit output
        if self.streams[cs].packet_done || self.streams[cs].packet_loss {
            if self.streams[cs].skip_packets != 0 {
                let mut min_idx = 0usize;
                let mut min_val = self.streams[0].skip_packets;
                for i in 1..self.streams.len() {
                    if self.streams[i].skip_packets < min_val {
                        min_val = self.streams[i].skip_packets;
                        min_idx = i;
                    }
                }
                self.current_stream = min_idx;
            }

            let mut nb_samples = usize::MAX;
            for i in 0..self.streams.len() {
                self.streams[i].skip_packets = self.streams[i].skip_packets.saturating_sub(1);
                nb_samples = nb_samples.min(self.fifo[i][0].len());
            }

            if !eof && !data.is_empty() {
                nb_samples -= nb_samples.min(4096);
            }

            if (nb_samples > 0 || eof || data.is_empty()) && !self.flushed {
                if eof {
                    let trim = (self.trim_end + self.trim_start).saturating_sub(128 + 64);
                    nb_samples -= trim.min(nb_samples);
                    self.flushed = true;
                }
                for i in 0..self.streams.len() {
                    let start_ch = self.start_channel[i];
                    let nb = nb_samples.min(self.fifo[i][0].len());
                    out[start_ch].extend_from_slice(self.fifo[i][0].read(nb));
                    if self.streams[i].nb_channels > 1 {
                        let nb = nb_samples.min(self.fifo[i][1].len());
                        out[start_ch + 1].extend_from_slice(self.fifo[i][1].read(nb));
                    }
                }
            }
        }

        consumed
    }
}
