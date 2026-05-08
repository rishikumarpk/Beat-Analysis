"""
Beat Analyzer Backend — Flask + NumPy FFT
==========================================
Implements Fourier analysis from scratch using numpy.fft
to detect beats, estimate BPM, and return graph data for
the frontend to visualize.
"""

import io, struct, wave, math, json
from flask import Flask, request, jsonify, send_from_directory

import numpy as np
from scipy.signal import butter, lfilter, find_peaks

app = Flask(__name__, static_folder="../frontend", static_url_path="")

# ─────────────────────────────────────────────────────────────
# CORS — manual header injection (no flask-cors needed)
# ─────────────────────────────────────────────────────────────
@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response

@app.route("/", methods=["OPTIONS"])
@app.route("/analyze", methods=["OPTIONS"])
def options():
    return "", 204


# ─────────────────────────────────────────────────────────────
# AUDIO DECODING
# ─────────────────────────────────────────────────────────────
def decode_wav(data: bytes) -> tuple[np.ndarray, int]:
    """Read WAV bytes → (mono float32 samples, sample_rate)."""
    with wave.open(io.BytesIO(data)) as wf:
        sr = wf.getframerate()
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    dtype_map = {1: np.int8, 2: np.int16, 4: np.int32}
    dtype = dtype_map.get(sample_width, np.int16)
    samples = np.frombuffer(raw, dtype=dtype).astype(np.float32)

    # Normalise to [-1, 1]
    samples /= float(np.iinfo(dtype).max)

    # Mix to mono
    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1)

    return samples, sr


# ─────────────────────────────────────────────────────────────
# STEP 1 — FOURIER TRANSFORM (explained)
# ─────────────────────────────────────────────────────────────
def compute_fft(samples: np.ndarray, sr: int, max_points: int = 2048):
    """
    The Discrete Fourier Transform (DFT) decomposes a time-domain
    signal x[n] into its frequency components:

        X[k] = Σ_{n=0}^{N-1} x[n] · e^{-i 2π k n / N}

    numpy.fft.rfft computes this efficiently using the FFT algorithm
    (O(N log N) instead of O(N²)).

    Returns freq bins (Hz) and magnitude spectrum.
    """
    N = len(samples)
    # Apply Hann window to reduce spectral leakage
    window = np.hanning(N)
    windowed = samples * window

    # FFT — only real-valued input so rfft gives N/2+1 complex bins
    fft_complex = np.fft.rfft(windowed)

    # Magnitude: |X[k]| = sqrt(Re² + Im²)
    magnitudes = np.abs(fft_complex)

    # Convert bin indices → frequency in Hz:  f = k * sr / N
    freqs = np.fft.rfftfreq(N, d=1.0 / sr)

    # Downsample for API response
    step = max(1, len(freqs) // max_points)
    return freqs[::step].tolist(), magnitudes[::step].tolist()


# ─────────────────────────────────────────────────────────────
# STEP 2 — SHORT-TIME FOURIER TRANSFORM (STFT)
# ─────────────────────────────────────────────────────────────
def compute_stft(samples: np.ndarray, sr: int,
                 frame_size: int = 2048, hop: int = 512):
    """
    STFT slides a window over time and computes FFT for each frame.
    This gives us a spectrogram: frequency × time → magnitude.

    Returns:
        times (s), freqs (Hz), magnitude matrix [freq × time]
    """
    n_frames = 1 + (len(samples) - frame_size) // hop
    window = np.hanning(frame_size)

    # Limit to 200 frames and 256 freq bins for API size
    max_frames = min(n_frames, 200)
    max_freq_bins = 256

    spec = np.zeros((max_freq_bins, max_frames), dtype=np.float32)
    times = []

    for i in range(max_frames):
        start = i * hop
        frame = samples[start: start + frame_size]
        if len(frame) < frame_size:
            frame = np.pad(frame, (0, frame_size - len(frame)))
        windowed_frame = frame * window
        fft_frame = np.fft.rfft(windowed_frame)
        mag = np.abs(fft_frame[:max_freq_bins])
        spec[:, i] = mag
        times.append(start / sr)

    freqs = np.fft.rfftfreq(frame_size, d=1.0 / sr)[:max_freq_bins]

    # Log-scale magnitude for display
    spec_db = 20 * np.log10(spec + 1e-6)
    spec_db -= spec_db.min()
    if spec_db.max() > 0:
        spec_db /= spec_db.max()

    return times, freqs.tolist(), spec_db.tolist()


# ─────────────────────────────────────────────────────────────
# STEP 3 — ONSET ENVELOPE + BEAT DETECTION
# ─────────────────────────────────────────────────────────────
def butter_bandpass(lowcut, highcut, sr, order=4):
    nyq = 0.5 * sr
    low = lowcut / nyq
    high = min(highcut / nyq, 0.99)
    b, a = butter(order, [low, high], btype="band")
    return b, a


def compute_onset_envelope(samples: np.ndarray, sr: int,
                            frame_size: int = 2048, hop: int = 512):
    """
    Onset strength = how much energy increases between frames.
    We use the bass band (60–250 Hz) because that's where
    kick drums and rhythmic bass live in Tamil music.

    Method:
    1. Band-pass filter to bass frequencies
    2. Compute RMS energy per frame
    3. Half-wave rectify the difference (keep only increases)
    """
    # Bass band-pass filter
    b, a = butter_bandpass(60, 250, sr)
    bass = lfilter(b, a, samples)

    n_frames = 1 + (len(bass) - frame_size) // hop
    energies = np.zeros(n_frames)

    for i in range(n_frames):
        start = i * hop
        frame = bass[start: start + frame_size]
        energies[i] = np.sqrt(np.mean(frame ** 2))  # RMS

    # Difference + half-wave rectification
    onset_env = np.diff(energies, prepend=energies[0])
    onset_env = np.maximum(onset_env, 0)

    # Normalise
    if onset_env.max() > 0:
        onset_env /= onset_env.max()

    frame_times = np.arange(n_frames) * hop / sr
    return frame_times.tolist(), onset_env.tolist()


def detect_beats(onset_times, onset_env, sr, hop=512):
    """
    Find peaks in the onset envelope.
    Uses adaptive threshold = mean + 0.5 * std
    Returns beat times (s) and estimated BPM.
    """
    env = np.array(onset_env)
    threshold = env.mean() + 0.5 * env.std()
    min_dist = int(0.3 * sr / hop)  # minimum 300ms between beats

    peaks, props = find_peaks(env, height=threshold, distance=min_dist)

    beat_times = [onset_times[p] for p in peaks if p < len(onset_times)]

    # BPM from median inter-beat interval
    bpm = 0.0
    if len(beat_times) >= 2:
        intervals = np.diff(beat_times)
        median_interval = np.median(intervals)
        bpm = round(60.0 / median_interval, 1) if median_interval > 0 else 0.0

    return beat_times, float(bpm), float(threshold)


# ─────────────────────────────────────────────────────────────
# STEP 4 — TEMPO AUTOCORRELATION
# ─────────────────────────────────────────────────────────────
def compute_autocorrelation(onset_env, sr, hop=512, max_points=512):
    """
    Autocorrelation of the onset envelope reveals periodicity.
    R[lag] = Σ onset[n] · onset[n + lag]

    Peaks in R correspond to the beat period.
    Convert lag → BPM: bpm = 60 * sr / (lag * hop)
    """
    env = np.array(onset_env)
    N = len(env)
    # Efficient autocorrelation via FFT convolution
    fft_env = np.fft.rfft(env, n=2 * N)
    acorr = np.fft.irfft(fft_env * np.conj(fft_env))[:N]
    acorr /= acorr[0] + 1e-9  # normalise

    # Convert lags to BPM
    lags = np.arange(1, N)
    bpm_axis = 60.0 * sr / (lags * hop)
    mask = (bpm_axis >= 50) & (bpm_axis <= 250)
    bpm_vals = bpm_axis[mask]
    acorr_vals = acorr[1:][mask]

    step = max(1, len(bpm_vals) // max_points)
    return bpm_vals[::step].tolist(), acorr_vals[::step].tolist()


# ─────────────────────────────────────────────────────────────
# STEP 5 — WAVEFORM DOWNSAMPLE
# ─────────────────────────────────────────────────────────────
def downsample_waveform(samples, target=2000):
    step = max(1, len(samples) // target)
    times = np.arange(0, len(samples), step) / 44100  # approx
    amps = samples[::step]
    return times.tolist(), amps.tolist()


# ─────────────────────────────────────────────────────────────
# STEP 6 — PHASE SPECTRUM
# ─────────────────────────────────────────────────────────────
def compute_phase(samples, sr, max_points=512):
    """
    Phase spectrum: angle of the complex FFT output.
    X[k] = |X[k]| · e^{iφ[k]}    →    φ[k] = arctan(Im/Re)

    Phase tells us the offset of each frequency component.
    """
    N = min(len(samples), 8192)
    window = np.hanning(N)
    fft_c = np.fft.rfft(samples[:N] * window)
    freqs = np.fft.rfftfreq(N, d=1.0 / sr)
    phase = np.angle(fft_c)  # radians
    step = max(1, len(freqs) // max_points)
    return freqs[::step].tolist(), phase[::step].tolist()


# ─────────────────────────────────────────────────────────────
# API ROUTES
# ─────────────────────────────────────────────────────────────
@app.route("/analyze", methods=["POST"])
def analyze():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400

    audio_file = request.files["audio"]
    raw = audio_file.read()

    try:
        samples, sr = decode_wav(raw)
    except Exception as e:
        return jsonify({"error": f"Could not decode WAV: {e}"}), 400

    duration = len(samples) / sr

    # Run all analysis steps
    fft_freqs, fft_mags = compute_fft(samples, sr)
    stft_times, stft_freqs, stft_spec = compute_stft(samples, sr)
    onset_times, onset_env = compute_onset_envelope(samples, sr)
    beat_times, bpm, threshold = detect_beats(onset_times, onset_env, sr)
    acorr_bpm, acorr_vals = compute_autocorrelation(onset_env, sr)
    wave_times, wave_amps = downsample_waveform(samples)
    phase_freqs, phase_vals = compute_phase(samples, sr)

    return jsonify({
        "meta": {
            "sample_rate": sr,
            "duration": round(duration, 2),
            "n_samples": len(samples),
            "bpm": bpm,
            "n_beats": len(beat_times),
        },
        "waveform": {
            "times": wave_times,
            "amplitudes": wave_amps,
        },
        "fft": {
            "freqs": fft_freqs,
            "magnitudes": fft_mags,
            "description": "Full-signal FFT magnitude spectrum (Hann-windowed)",
        },
        "phase": {
            "freqs": phase_freqs,
            "values": phase_vals,
            "description": "Phase angle φ[k] = arctan(Im / Re) in radians",
        },
        "spectrogram": {
            "times": stft_times,
            "freqs": stft_freqs,
            "magnitudes": stft_spec,
            "description": "STFT — FFT computed on sliding 2048-sample Hann windows",
        },
        "onset": {
            "times": onset_times,
            "envelope": onset_env,
            "beat_times": beat_times,
            "threshold": threshold,
            "description": "Bass-band RMS energy difference — half-wave rectified",
        },
        "autocorrelation": {
            "bpm_axis": acorr_bpm,
            "values": acorr_vals,
            "description": "Autocorrelation of onset envelope reveals rhythmic period",
        },
    })


@app.route("/")
def index():
    return send_from_directory("../frontend", "index.html")


if __name__ == "__main__":
    print("Starting Beat Analyzer backend on http://localhost:5050")
    app.run(debug=True, port=5050)
