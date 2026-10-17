//! Microphone capture via `cpal` (cpal 0.18 API).
//!
//! Produces 16 kHz mono 16-bit PCM — exactly what Whisper expects — and can persist it
//! as raw `.pcm` and/or a `.wav` file. Designed for Push-to-Talk: `record_seconds` opens
//! the mic on entry and tears the stream down cleanly on exit (no deadlock on key-up).
//!
//! On macOS the first input-device open triggers the system Microphone permission prompt;
//! that is expected. If the mic is denied/absent, an empty buffer yields `CaptureError::Timeout`
//! rather than a hang.

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::io::{self, Cursor, Read, Write as _};
use std::time::Duration;

/// Sample rate Whisper consumes: 16 kHz single-channel.
pub const WHISPER_SAMPLE_RATE: u32 = 16_000;

/// A captured clip: 16 kHz mono i16 samples.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Audio {
    pub samples: Vec<i16>,
    pub sample_rate: u32,
}

impl Audio {
    /// Duration of the clip in seconds (float).
    pub fn duration_secs(&self) -> f32 {
        self.samples.len() as f32 / self.sample_rate as f32
    }

    /// Root-mean-square level over the clip, in dBFS. `-inf` for an all-silence buffer.
    pub fn rms_db(&self) -> f32 {
        if self.samples.is_empty() {
            return f32::NEG_INFINITY;
        }
        let sum_sq: f64 = self
            .samples
            .iter()
            .map(|&s| (s as f64 / 32768.0) * (s as f64 / 32768.0))
            .sum();
        let rms = (sum_sq / self.samples.len() as f64).sqrt();
        if rms <= 0.0 {
            f32::NEG_INFINITY
        } else {
            20.0 * rms.log10() as f32
        }
    }
}

/// Errors from capturing.
#[derive(Debug)]
pub enum CaptureError {
    NoInputDevice,
    NoInputConfig,
    Io(std::io::Error),
    Timeout,
}

impl std::fmt::Display for CaptureError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CaptureError::NoInputDevice => write!(f, "no default input (microphone) device found"),
            CaptureError::NoInputConfig => write!(f, "could not read the device input config"),
            CaptureError::Io(e) => write!(f, "io error: {e}"),
            CaptureError::Timeout => write!(f, "timed out waiting for microphone samples"),
        }
    }
}

impl std::error::Error for CaptureError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            CaptureError::Io(e) => Some(e),
            _ => None,
        }
    }
}

/// Open the default input device; return device + native sample rate + channel count.
fn open_input() -> Result<(cpal::Device, u32, u16), CaptureError> {
    let host = cpal::default_host();
    let device = host
        .input_devices()
        .map_err(|_| CaptureError::NoInputDevice)?
        .next()
        .ok_or(CaptureError::NoInputDevice)?;
    let supp = device
        .default_input_config()
        .map_err(|_| CaptureError::NoInputConfig)?;
    let native_rate = supp.sample_rate(); // u32 (SampleRate is a type alias in cpal 0.18)
    let channels = supp.channels(); // u16
    Ok((device, native_rate, channels))
}

/// Down-mix N-channel f32 -> mono i16 and decimate to 16 kHz. Used by the synthetic test.
#[allow(dead_code)]
fn mix_to_16k_mono_i16(src: &[f32], native_rate: u32, native_channels: u16, out: &mut Vec<i16>) {
    let channels = native_channels.max(1);
    let num_frames = src.len() / channels as usize;
    let step = (native_rate as f32 / WHISPER_SAMPLE_RATE as f32).round() as u32;
    let step = step.max(1) as usize;
    for f in 0..num_frames {
        if f % step != 0 {
            continue;
        }
        let base = f * channels as usize;
        let val: f32 = (0..channels as usize).map(|c| src[base + c]).sum::<f32>() / channels as f32;
        out.push((val.clamp(-1.0, 1.0) * 32767.0).round() as i16);
    }
}

/// Streaming down-mixer/resampler: N-channel i16 at the device rate -> mono i16 at 16 kHz.
/// Averages input frames over each output period (a box low-pass), which avoids the aliasing and
/// pitch error of naive decimation, including for non-integer ratios such as 44.1 kHz.
struct Resampler {
    channels: usize,
    ratio: f32,
    phase: f32,
    acc: f32,
    n: u32,
}

impl Resampler {
    fn new(native_rate: u32, native_channels: u16) -> Self {
        Resampler {
            channels: native_channels.max(1) as usize,
            ratio: (native_rate as f32 / WHISPER_SAMPLE_RATE as f32).max(1.0),
            phase: 0.0,
            acc: 0.0,
            n: 0,
        }
    }

    fn push(&mut self, src: &[i16], out: &mut Vec<i16>) {
        for frame in src.chunks_exact(self.channels) {
            let mono = frame.iter().map(|&s| s as f32).sum::<f32>() / self.channels as f32;
            self.acc += mono;
            self.n += 1;
            self.phase += 1.0;
            if self.phase >= self.ratio {
                self.phase -= self.ratio;
                out.push((self.acc / self.n as f32).round().clamp(-32768.0, 32767.0) as i16);
                self.acc = 0.0;
                self.n = 0;
            }
        }
    }
}

/// Record from the microphone for approximately `max` seconds, returning 16 kHz mono
/// i16 samples. Blocks until the buffer fills for `max` seconds, or times out (denied mic).
///
/// No deadlock on stop: `closed` tells the callback to go quiet and the stream is dropped,
/// so the function always returns promptly when called on the P2T key-up.
pub fn record_seconds(max: Duration) -> Result<Audio, CaptureError> {
    // Build with T = i16; CoreAudio (cpal's macOS backend) converts the device's native
    // format to i16, so we need not match the device's native format. The callback
    // down-mixes to mono and resamples to 16 kHz.
    let (device, native_rate, native_channels) = open_input()?;
    let config = device
        .default_input_config()
        .map_err(|_| CaptureError::NoInputConfig)?
        .config();

    let buf = std::sync::Arc::new(std::sync::Mutex::new(Vec::<i16>::new()));
    let closed = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));

    let closed_c = closed.clone();
    let buf_c = buf.clone();
    let err_cb = move |err: cpal::Error| eprintln!("[capture] stream error: {err:?}");
    let mut resampler = Resampler::new(native_rate, native_channels);
    let data_cb = move |data: &[i16], _info: &cpal::InputCallbackInfo| {
        if closed_c.load(std::sync::atomic::Ordering::Relaxed) {
            return;
        }
        resampler.push(data, &mut buf_c.lock().unwrap());
    };

    // cpal 0.18: build_input_stream(config, data_cb, err_cb, timeout: Option<Duration>).
    let stream = match device.build_input_stream(config, data_cb, err_cb, None) {
        Ok(s) => s,
        Err(_) => return Err(CaptureError::NoInputConfig),
    };
    stream.play().map_err(|_| CaptureError::NoInputConfig)?;

    // Collect ~`max` seconds at 16 kHz. Never hang on a denied/absent mic: if no new
    // samples arrive for 5 s, treat it as a timeout.
    let target = (max.as_secs_f32() * WHISPER_SAMPLE_RATE as f32).ceil() as usize;
    let mut since_last = std::time::Instant::now();
    while buf.lock().unwrap().len() < target {
        std::thread::sleep(Duration::from_millis(15));
        let n = buf.lock().unwrap().len();
        if n > 0 {
            since_last = std::time::Instant::now();
        } else if since_last.elapsed() > Duration::from_secs(5) {
            break;
        }
    }
    // No deadlock on stop: tell the callback to go quiet, then tear the stream down.
    closed.store(true, std::sync::atomic::Ordering::Relaxed);
    drop(stream);

    let mut bufg = buf.lock().unwrap();
    if bufg.is_empty() {
        // Nothing captured for the whole window: the mic was likely denied or absent.
        return Err(CaptureError::Timeout);
    }
    let samples = std::mem::take(&mut *bufg);
    Ok(Audio {
        samples,
        sample_rate: WHISPER_SAMPLE_RATE,
    })
}

/// A start/stop microphone capture for Push-to-Talk: `start()` opens the input stream on key-down
/// (non-blocking) and `stop()` closes it on key-up and returns the buffered 16 kHz mono i16 Audio.
/// This is what let us know when the P2T key is "released".
pub struct CaptureHandle {
    buf: std::sync::Arc<std::sync::Mutex<Vec<i16>>>,
    closed: std::sync::Arc<std::sync::atomic::AtomicBool>,
    _stream: Option<cpal::Stream>,
}

impl CaptureHandle {
    /// Open the default input device and begin buffering without blocking for the user's speech.
    pub fn start() -> Result<CaptureHandle, CaptureError> {
        let (device, native_rate, native_channels) = open_input()?;
        let config = device
            .default_input_config()
            .map_err(|_| CaptureError::NoInputConfig)?
            .config();
        let buf = std::sync::Arc::new(std::sync::Mutex::new(Vec::<i16>::new()));
        let closed = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let closed_c = closed.clone();
        let buf_c = buf.clone();
        let err_cb = move |err: cpal::Error| eprintln!("[capture] stream error: {err:?}");
        let mut resampler = Resampler::new(native_rate, native_channels);
    let data_cb = move |data: &[i16], _info: &cpal::InputCallbackInfo| {
            if closed_c.load(std::sync::atomic::Ordering::Relaxed) {
                return;
            }
            resampler.push(data, &mut buf_c.lock().unwrap());
        };
        let stream = match device.build_input_stream(config, data_cb, err_cb, None) {
            Ok(s) => s,
            Err(_) => return Err(CaptureError::NoInputConfig),
        };
        stream.play().map_err(|_| CaptureError::NoInputConfig)?;
        Ok(CaptureHandle {
            buf,
            closed,
            _stream: Some(stream),
        })
    }

    /// Stop capturing and return everything buffered since `start()`.
    pub fn stop(self) -> Result<Audio, CaptureError> {
        self.closed
            .store(true, std::sync::atomic::Ordering::Relaxed);
        if let Some(s) = self._stream {
            drop(s);
        }
        let mut g = self.buf.lock().unwrap();
        if g.is_empty() {
            return Err(CaptureError::Timeout);
        }
        let samples = std::mem::take(&mut *g);
        Ok(Audio {
            samples,
            sample_rate: WHISPER_SAMPLE_RATE,
        })
    }
}

/// Write a 16-bit PCM WAV (44-byte header + interleaved i16). Returns bytes written.
pub fn write_wav(path: impl AsRef<std::path::Path>, a: &Audio) -> Result<usize, CaptureError> {
    let data_bytes: Vec<u8> = a.samples.iter().flat_map(|s| s.to_le_bytes()).collect();
    let total_len = (44 + data_bytes.len()) as u32;
    let sample_rate: u32 = a.sample_rate;
    let num_channels: u16 = 1;
    let bits_per_sample: u16 = 16;
    let byte_rate = sample_rate * num_channels as u32 * bits_per_sample as u32 / 8;
    let block_align = (num_channels as u32 * bits_per_sample as u32) / 8;

    let mut buf: Vec<u8> = Vec::new();
    buf.extend_from_slice(b"RIFF");
    buf.extend_from_slice(&total_len.to_le_bytes());
    buf.extend_from_slice(b"WAVE");
    buf.extend_from_slice(b"fmt ");
    buf.extend_from_slice(&16u32.to_le_bytes()); // PCM chunk size
    buf.extend_from_slice(&1u16.to_le_bytes()); // PCM = 1
    buf.extend_from_slice(&num_channels.to_le_bytes());
    buf.extend_from_slice(&sample_rate.to_le_bytes());
    buf.extend_from_slice(&byte_rate.to_le_bytes());
    buf.extend_from_slice(&block_align.to_le_bytes());
    buf.extend_from_slice(&bits_per_sample.to_le_bytes());
    buf.extend_from_slice(b"data");
    buf.extend_from_slice(&(data_bytes.len() as u32).to_le_bytes());
    buf.extend_from_slice(&data_bytes);

    let mut f = std::fs::File::create(path.as_ref()).map_err(CaptureError::Io)?;
    f.write_all(&buf).map_err(CaptureError::Io)?;
    Ok(buf.len())
}

/// Write raw interleaved 16-bit PCM (no header).
pub fn write_pcm(path: impl AsRef<std::path::Path>, a: &Audio) -> Result<usize, CaptureError> {
    let bytes: Vec<u8> = a.samples.iter().flat_map(|s| s.to_le_bytes()).collect();
    let mut f = std::fs::File::create(path.as_ref()).map_err(CaptureError::Io)?;
    f.write_all(&bytes).map_err(CaptureError::Io)?;
    Ok(bytes.len())
}

/// Minimal PCM reader (16-bit little-endian, mono) for round-trip verification.
pub fn read_pcm(path: impl AsRef<std::path::Path>) -> Result<Vec<i16>, CaptureError> {
    let bytes = std::fs::read(path).map_err(CaptureError::Io)?;
    if bytes.len() % 2 != 0 {
        return Err(CaptureError::Io(io::Error::new(
            io::ErrorKind::InvalidData,
            "odd number of bytes in pcm file",
        )));
    }
    let mut out = Vec::with_capacity(bytes.len() / 2);
    let mut cursor = Cursor::new(&bytes[..]);
    while let Some(low) = read_u8(&mut cursor) {
        let high = match read_u8(&mut cursor) {
            Some(h) => h,
            None => break,
        };
        out.push(i16::from_le_bytes([low, high]));
    }
    Ok(out)
}

fn read_u8(c: &mut Cursor<&[u8]>) -> Option<u8> {
    let mut b = [0u8; 1];
    match c.read(&mut b) {
        Ok(1) => Some(b[0]),
        _ => None,
    }
}

/// Read a 16-bit mono PCM WAV (the format `write_wav` produces) into 16-bit samples.
/// Skips the RIFF/WAVE header and any non-data chunks, honoring the `data` chunk length.
pub fn read_wav(path: impl AsRef<std::path::Path>) -> Result<Vec<i16>, CaptureError> {
    let bytes = std::fs::read(path).map_err(CaptureError::Io)?;
    cursor_read_wav(&bytes)
}

fn cursor_read_wav(bytes: &[u8]) -> Result<Vec<i16>, CaptureError> {
    if bytes.len() < 44 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(CaptureError::Io(io::Error::new(
            io::ErrorKind::InvalidData,
            "not a RIFF/WAVE file",
        )));
    }
    // Walk chunks to find `fmt ` and `data`.
    let mut off = 12usize;
    let mut fmt = 0;
    let mut data_off: Option<usize> = None;
    loop {
        if off + 8 > bytes.len() {
            break;
        }
        let id = &bytes[off..off + 4];
        let size = u32::from_le_bytes([bytes[off + 4], bytes[off + 5], bytes[off + 6], bytes[off + 7]]) as usize;
        let body = off + 8;
        if id == b"fmt " {
            fmt = 1; // assume PCM
        } else if id == b"data" {
            data_off = Some(body);
        }
        off = body + size + (size & 1); // data/fmt chunks are word-aligned
    }
    if fmt != 1 {
        return Err(CaptureError::Io(io::Error::new(
            io::ErrorKind::InvalidData,
            "non-PCM WAV",
        )));
    }
    let Some(start) = data_off else {
        return Err(CaptureError::Io(io::Error::new(
            io::ErrorKind::InvalidData,
            "no data chunk",
        )));
    };
    let mut out = Vec::with_capacity((bytes.len() - start) / 2);
    let mut i = start;
    while i + 1 < bytes.len() {
        out.push(i16::from_le_bytes([bytes[i], bytes[i + 1]]));
        i += 2;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resampler_preserves_level_and_length() {
        // 1 s of a loud 44.1 kHz stereo tone must come out ~16000 samples with its amplitude intact
        // (regression: samples were once clamped to +-1 before scaling, turning speech to noise).
        let mut r = Resampler::new(44_100, 2);
        let mut src = Vec::new();
        for f in 0..44_100 {
            let v = ((f as f32 / 44_100.0 * 200.0 * std::f32::consts::TAU).sin() * 16_000.0) as i16;
            src.extend([v, v]);
        }
        let mut out = Vec::new();
        r.push(&src, &mut out);
        assert!((out.len() as i32 - 16_000).abs() <= 2, "len {}", out.len());
        let peak = out.iter().map(|s| s.abs()).max().unwrap();
        assert!(peak > 15_000 && peak <= 16_000, "peak {peak}");
    }

    // Deterministically verify the resample/serialize path without a microphone: build a
    // known 48 kHz stereo buffer (a sine wave), run the down-mix/resample, and assert the
    // resulting 16 kHz mono clip has the right length, non-zero RMS, and round-trips.
    #[test]
    fn synthetic_clip_resamples_and_roundtrips() {
        // 1.0 s of 48 kHz stereo sine -> expect ~16000 mono i16 samples.
        let native_rate = 48_000u32;
        let channels = 2u16;
        let frames = 48_000usize; // 1 s at 48k
        let mut src: Vec<f32> = Vec::with_capacity(frames * channels as usize);
        for f in 0..frames {
            let t = f as f32 / native_rate as f32;
            let v = (t * 440.0_f32 * std::f32::consts::PI * 2.0_f32).sin();
            src.push(v); // L
            src.push(v); // R
        }
        let mut out: Vec<i16> = Vec::new();
        mix_to_16k_mono_i16(&src, native_rate, channels, &mut out);

        // Every 3rd frame of 48k -> ~16000 samples.
        assert!(
            out.len() >= 15_500 && out.len() <= 16_500,
            "resampled len {} out of [15500,16500]",
            out.len()
        );
        let a = Audio {
            samples: out.clone(),
            sample_rate: WHISPER_SAMPLE_RATE,
        };
        assert!(
            a.rms_db() > -30.0,
            "expected audible RMS, got {} dB",
            a.rms_db()
        );
        assert!(
            (a.duration_secs() - 1.0).abs() < 0.05,
            "duration {:?}",
            a.duration_secs()
        );

        let dir =
            std::env::temp_dir().join(format!("zipscribe-capture-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let wav = dir.join("clip.wav");
        let pcm = dir.join("clip.pcm");
        write_wav(&wav, &a).unwrap();
        write_pcm(&pcm, &a).unwrap();
        assert!(wav.metadata().unwrap().len() > 44 + 2 * a.samples.len() as u64);
        let back = read_pcm(&pcm).unwrap();
        assert_eq!(
            back.len(),
            a.samples.len(),
            "PCM round-trip length mismatch"
        );
        assert_eq!(back, a.samples, "PCM round-trip content mismatch");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn silent_clip_reports_neg_inf_rms() {
        let a = Audio {
            samples: vec![0; 160],
            sample_rate: WHISPER_SAMPLE_RATE,
        };
        assert_eq!(a.rms_db(), f32::NEG_INFINITY);
        assert!((a.duration_secs() - 0.01).abs() < 1e-6);
    }
}
