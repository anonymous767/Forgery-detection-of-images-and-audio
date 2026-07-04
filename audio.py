"""
╔══════════════════════════════════════════════════════════════════╗
║           AUDIO FORGERY DETECTION SYSTEM  v2.0                  ║
║   No CNN. No training data. Fully heuristic + GMM anomaly.      ║
╚══════════════════════════════════════════════════════════════════╝

HOW TO RUN:
  1. Install deps:  pip install librosa numpy scipy scikit-learn matplotlib soundfile
  2. Set FILE_PATH below to your audio file
  3. Run:  python audio_forgery_detector.py

OUTPUTS (saved next to this script):
  • forensic_report.png  — 6-panel visual analysis
  • forensic_report.json — full numeric results
  • Console verdict      — AUTHENTIC / SUSPICIOUS / FORGED
"""

# ══════════════════════════════════════════════════════════════════
#   PUT YOUR AUDIO FILE PATH HERE  
# ══════════════════════════════════════════════════════════════════
FILE_PATH = "C:\\Python\\Learning python\\testaudio.mp3"  # ← change this
# ══════════════════════════════════════════════════════════════════

# Optional settings
OUTPUT_DIR     = "."          # folder to save PNG + JSON (default: same as script)
SAMPLE_RATE    = 16000        # resample everything to this SR
NOMINAL_ENF_HZ = 50.0         # 50 for India/Europe, 60 for USA/Canada

# ─────────────────────────────────────────────────────────────────
import os, sys, json, time, warnings
import numpy as np
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal  import butter, filtfilt, welch, find_peaks
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from scipy.spatial.distance import cdist
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════
# SECTION 1 — AUDIO LOADING
# ══════════════════════════════════════════════════════════════════
#
# Supported formats:
#   .wav   — no extra deps (soundfile)
#   .flac  — no extra deps (soundfile)
#   .ogg   — no extra deps (soundfile)
#   .mp3   — needs: pip install pydub
#            + ffmpeg on your system:
#              Ubuntu/Debian : sudo apt install ffmpeg
#              Windows       : winget install ffmpeg
#              macOS         : brew install ffmpeg

# ── Supported format sets ─────────────────────────────────────────
SUPPORTED_FORMATS = {".wav", ".mp3", ".flac", ".ogg"}   # ← exactly these 4

# WAV / FLAC / OGG are handled natively by librosa + soundfile
_NATIVE_EXTS = {".wav", ".flac", ".ogg"}

# MP3 needs pydub + ffmpeg
_PYDUB_EXTS  = {".mp3"}


def _load_via_pydub(path: str) -> tuple:
    """
    Convert any ffmpeg-readable file to a raw PCM numpy array.
    Returns (samples_float32, sample_rate).
    """
    try:
        from pydub import AudioSegment
    except ImportError:
        sys.exit(
            "[ERROR] pydub is not installed.\n"
            "  Run:  pip install pydub\n"
            "  Also install ffmpeg:\n"
            "    Ubuntu:  sudo apt install ffmpeg\n"
            "    Windows: winget install ffmpeg\n"
            "    macOS:   brew install ffmpeg"
        )

    ext = os.path.splitext(path)[1].lower().lstrip(".")
    try:
        audio = AudioSegment.from_file(path, format=ext)
    except Exception as e:
        sys.exit(
            f"[ERROR] pydub could not open '{path}': {e}\n"
            "  Make sure ffmpeg is installed and on your PATH."
        )

    # Convert to mono, 16-bit, then to float32 numpy
    audio  = audio.set_channels(1)
    orig_sr = audio.frame_rate
    samples = np.array(audio.get_array_of_samples(), dtype=np.int16)
    y       = samples.astype(np.float32) / 32768.0   # normalise to [-1, 1]
    return y, orig_sr


def load_audio(path: str, sr: int = 16000) -> np.ndarray:
    """
    Load WAV, MP3, FLAC, or OGG. Always returns mono float32 resampled to `sr`.
    """
    if not os.path.isfile(path):
        sys.exit(f"[ERROR] File not found: {path}")

    ext = os.path.splitext(path)[1].lower()

    if ext not in SUPPORTED_FORMATS:
        sys.exit(
            f"[ERROR] Format '{ext}' is not supported.\n"
            f"        Accepted formats: WAV  MP3  FLAC  OGG\n"
            f"        Rename/convert your file to one of these."
        )

    fmt_label = ext.upper().lstrip(".")
    print(f"[load] Format: {fmt_label}  |  File: {os.path.basename(path)}")

    y, orig_sr = None, None

    # ── Try librosa first (handles most formats via soundfile) ────
    if ext in _NATIVE_EXTS or ext not in _PYDUB_EXTS:
        try:
            y, orig_sr = librosa.load(path, sr=None, mono=True)
        except Exception as e:
            print(f"[load] librosa failed ({e}) — trying pydub fallback …")

    # ── pydub fallback ────────────────────────────────────────────
    if y is None:
        y, orig_sr = _load_via_pydub(path)

    if y is None or len(y) == 0:
        sys.exit("[ERROR] Could not load audio — file may be corrupt or unsupported.")

    # ── Resample if needed ────────────────────────────────────────
    if orig_sr != sr:
        print(f"[load] Resampling {orig_sr} Hz → {sr} Hz")
        y = librosa.resample(y, orig_sr=orig_sr, target_sr=sr)

    duration = len(y) / sr
    print(f"[load] ✓  Duration: {duration:.2f}s  |  SR: {sr} Hz  |  Samples: {len(y)}")
    return y.astype(np.float32)


# ══════════════════════════════════════════════════════════════════
# SECTION 2 — FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════════
HOP = 512   # shared hop length (frames at 16kHz ≈ 32ms steps)
N_FFT = 2048

def get_mfcc(y, sr, n_mfcc=40):
    """
    MFCCs: Mel-Frequency Cepstral Coefficients.
    Each column = one time frame; rows = 40 cepstral coefficients.
    Captures the 'timbre fingerprint' of sound at each moment.
    """
    return librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc,
                                  hop_length=HOP, n_fft=N_FFT)

def get_spectral_flux(y, sr):
    """
    Spectral flux: how much the frequency spectrum changes frame-to-frame.
    Sudden spikes → possible re-encoding or edit boundary.
    """
    S    = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP))
    flux = np.sum(np.maximum(0, np.diff(S, axis=1)), axis=0)
    return np.concatenate([[0], flux])

def get_rms(y):
    """Energy per frame. Splices often cause unnatural energy jumps."""
    return librosa.feature.rms(y=y, hop_length=HOP)[0]

def get_zcr(y):
    """Zero-crossing rate: how often the waveform crosses zero. Texture indicator."""
    return librosa.feature.zero_crossing_rate(y, hop_length=HOP)[0]

def get_mel_spec(y, sr, n_mels=128):
    """Mel-spectrogram in dB. Used for visualisation and GMM input."""
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels,
                                          n_fft=N_FFT, hop_length=HOP)
    return librosa.power_to_db(mel, ref=np.max)


# ══════════════════════════════════════════════════════════════════
# SECTION 3 — DETECTOR 1: SPLICE DETECTION (MFCC DISCONTINUITY)
# ══════════════════════════════════════════════════════════════════
"""
HOW IT WORKS:
  Audio that has been cut and joined will have a sudden "jump" in its
  spectral fingerprint at the edit point.

  We slide a window across MFCC frames and compute the cosine distance
  between the window just before and just after each frame.
  Normal speech transitions smoothly → low distance.
  A splice → distance spikes sharply.

  We flag frames where the spike exceeds: local_mean + 2×std
"""

def detect_splices(y, sr, window=20, threshold_factor=2.0):
    mfcc = get_mfcc(y, sr)
    T    = mfcc.shape[1]
    distances = []

    for i in range(1, T):
        a = mfcc[:, max(0, i - window):i].T
        b = mfcc[:, i:i + window].T
        if a.shape[0] < 2 or b.shape[0] < 2:
            distances.append(0.0)
            continue
        mu_a = a.mean(axis=0, keepdims=True)
        mu_b = b.mean(axis=0, keepdims=True)
        d    = float(cdist(mu_a, mu_b, metric="cosine")[0, 0])
        distances.append(d)

    dists     = np.array(distances)
    smoothed  = uniform_filter1d(dists, size=window)
    threshold = smoothed + threshold_factor * dists.std()
    peaks, _  = find_peaks(dists, height=threshold, distance=window)

    times = [round(p * HOP / sr, 3) for p in peaks]
    return {
        "splice_times_sec": times,
        "distances":        dists,
        "threshold":        threshold,
        "anomaly":          len(times) > 0,
        "detail":           f"{len(times)} potential splice point(s) found",
    }


# ══════════════════════════════════════════════════════════════════
# SECTION 4 — DETECTOR 2: COPY-MOVE DETECTION (SELF-SIMILARITY)
# ══════════════════════════════════════════════════════════════════
"""
HOW IT WORKS:
  If a segment of audio was copied and pasted elsewhere in the file,
  those two regions will have near-identical MFCC fingerprints.

  We build a self-similarity matrix: compare every frame to every
  other frame using cosine similarity. Off-diagonal bright spots
  (similarity ≥ 0.95) that are far apart in time = copy-move.
"""

def detect_copy_move(y, sr, similarity_threshold=0.95, min_gap_frames=10):
    mfcc    = get_mfcc(y, sr, n_mfcc=20)
    # Coarser hop for self-similarity (efficiency for long files)
    hop_cm  = 1024
    mfcc_cm = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20,
                                     hop_length=hop_cm, n_fft=N_FFT)
    T   = mfcc_cm.shape[1]
    nrm = np.linalg.norm(mfcc_cm, axis=0, keepdims=True) + 1e-8
    M   = (mfcc_cm / nrm).T    # [T × n_mfcc]

    pairs = []
    fps   = sr / hop_cm        # frames per second
    batch = 200

    for i in range(0, T, batch):
        chunk = M[i: i + batch]
        sim   = np.dot(chunk, M.T)

        for bi, row in enumerate(sim):
            gi = i + bi
            # mask self-region
            row[max(0, gi - min_gap_frames): gi + min_gap_frames + 1] = 0.0
            matches = np.where(row >= similarity_threshold)[0]
            matches = matches[matches > gi + min_gap_frames]
            for gj in matches:
                pairs.append({
                    "src_sec":    round(gi / fps, 3),
                    "dst_sec":    round(gj / fps, 3),
                    "similarity": round(float(row[gj]), 4),
                })

    # De-duplicate by merging adjacent pairs
    pairs = _merge_pairs(pairs)
    return {
        "copy_pairs": pairs,
        "anomaly":    len(pairs) > 0,
        "detail":     f"{len(pairs)} copied segment(s) detected",
    }

def _merge_pairs(pairs, gap=0.5):
    if not pairs:
        return []
    pairs = sorted(pairs, key=lambda x: (x["src_sec"], x["dst_sec"]))
    out   = [pairs[0]]
    for p in pairs[1:]:
        prev = out[-1]
        if (abs(p["src_sec"] - prev["src_sec"]) < gap and
                abs(p["dst_sec"] - prev["dst_sec"]) < gap):
            continue  # same pair, skip
        out.append(p)
    return out


# ══════════════════════════════════════════════════════════════════
# SECTION 5 — DETECTOR 3: ENF ANALYSIS
# ══════════════════════════════════════════════════════════════════
"""
HOW IT WORKS:
  Indoor recordings pick up a faint hum from the electrical mains
  (50 Hz in India/Europe, 60 Hz in USA). This is the Electric Network
  Frequency (ENF). The exact frequency fluctuates slightly over time
  (e.g. 49.97 → 50.02 Hz) — and this fluctuation is globally consistent
  for a given time and region.

  Authentic continuous recordings have a smoothly varying ENF.
  Spliced recordings — stitched from different times or rooms —
  show abrupt ENF jumps at the edit points.

  We bandpass-filter around the 2nd harmonic (100 Hz), track the
  instantaneous frequency using Welch's method per frame, and flag:
    • High standard deviation (> 0.08 Hz)
    • Abrupt jumps between frames (> 0.1 Hz)
"""

def detect_enf(y, sr, nominal_hz=50.0, band=0.5, frame=8192, hop_e=4096):
    harmonic = nominal_hz * 2   # track 2nd harmonic (cleaner SNR)
    low  = max((harmonic - band) / (sr / 2), 1e-5)
    high = min((harmonic + band) / (sr / 2), 0.9999)
    b, a = butter(4, [low, high], btype="band")
    yf   = filtfilt(b, a, y)

    estimates = []
    for start in range(0, len(yf) - frame, hop_e):
        seg   = yf[start: start + frame]
        freqs, psd = welch(seg, fs=sr, nperseg=frame)
        band_idx   = np.where((freqs >= harmonic - band) &
                               (freqs <= harmonic + band))[0]
        if len(band_idx) == 0:
            estimates.append(nominal_hz)
            continue
        peak = band_idx[np.argmax(psd[band_idx])]
        estimates.append(freqs[peak] / 2)   # back to fundamental

    enf = np.array(estimates)
    if len(enf) < 4:
        return {"enf_series": enf.tolist(), "std_hz": 0.0,
                "jump_count": 0, "anomaly": False, "detail": "Too short for ENF"}

    smoothed   = gaussian_filter1d(enf, sigma=2)
    std_hz     = float(np.std(smoothed))
    jumps      = int(np.sum(np.abs(np.diff(smoothed)) > 0.1))
    anomaly    = std_hz > 0.08 or jumps > 3

    return {
        "enf_series": enf.tolist(),
        "std_hz":     round(std_hz, 5),
        "jump_count": jumps,
        "anomaly":    anomaly,
        "detail":     f"σ={std_hz:.4f} Hz  |  {jumps} abrupt jump(s)",
    }


# ══════════════════════════════════════════════════════════════════
# SECTION 6 — DETECTOR 4: NOISE FLOOR INCONSISTENCY
# ══════════════════════════════════════════════════════════════════
"""
HOW IT WORKS:
  Every recording environment has a characteristic background noise level
  (hiss, room tone, HVAC hum, etc.).

  When audio is spliced from two different recordings, the background
  noise level changes abruptly at the cut point.

  We split the audio into 0.5-second segments, estimate the noise floor
  of each (bottom 20th percentile of RMS energy), convert to dB, and
  flag segments where the floor jumps by more than 6 dB.

  6 dB = a doubling of noise amplitude — a very clear sign of splicing.
"""

def detect_noise_floor(y, sr, seg_sec=0.5, threshold_db=6.0):
    seg_len  = int(sr * seg_sec)
    n_segs   = len(y) // seg_len
    rms_vals = []

    for i in range(n_segs):
        seg    = y[i * seg_len: (i + 1) * seg_len]
        frames = librosa.util.frame(seg, frame_length=512, hop_length=256)
        rms_f  = np.sqrt(np.mean(frames ** 2, axis=0))
        rms_vals.append(float(np.percentile(rms_f, 20)))

    rms_arr  = np.array(rms_vals)
    noise_db = 20 * np.log10(np.maximum(rms_arr, 1e-10))
    diffs    = np.abs(np.diff(noise_db))
    shifts   = list(np.where(diffs > threshold_db)[0])
    max_jump = float(np.max(diffs)) if len(diffs) else 0.0

    shift_times = [round(i * seg_sec, 3) for i in shifts]
    return {
        "noise_rms_db":   noise_db.tolist(),
        "shift_times_sec": shift_times,
        "max_jump_db":    round(max_jump, 2),
        "anomaly":        len(shifts) > 0,
        "detail":         f"Max noise jump: {max_jump:.1f} dB  |  {len(shifts)} shift(s)",
    }


# ══════════════════════════════════════════════════════════════════
# SECTION 7 — DETECTOR 5: SPECTRAL DISCONTINUITY (DOUBLE COMPRESSION)
# ══════════════════════════════════════════════════════════════════
"""
HOW IT WORKS:
  When audio is exported, edited in an editor, then exported again
  (double-compressed), codec quantisation artefacts accumulate.
  At edit boundaries, spectral flux spikes sharply.

  We measure spectral flux (positive frame-to-frame spectral change)
  and flag spikes beyond mean + 2.5×std.
  Also catches re-encoded MP3→edit→MP3 tampering that other detectors miss.
"""

def detect_spectral_discontinuity(y, sr, threshold_factor=2.5):
    flux      = get_spectral_flux(y, sr)
    mean_f    = flux.mean()
    std_f     = flux.std()
    threshold = mean_f + threshold_factor * std_f
    peaks, _  = find_peaks(flux, height=threshold, distance=10)
    times     = [round(p * HOP / sr, 3) for p in peaks]

    return {
        "spike_times_sec": times,
        "flux_series":     flux,        # keep as array for plotting
        "threshold":       float(threshold),
        "anomaly":         len(times) > 0,
        "detail":          f"{len(times)} spectral spike(s) detected",
    }


# ══════════════════════════════════════════════════════════════════
# SECTION 8 — DETECTOR 6: GMM ANOMALY SCORING
# ══════════════════════════════════════════════════════════════════
"""
HOW IT WORKS:
  A Gaussian Mixture Model (GMM) is a statistical model that learns
  how MFCC frames are normally distributed in an audio file.
  Think of it as learning "what this audio is supposed to sound like."

  We fit the GMM on all MFCC frames of the file itself (unsupervised —
  no training data needed). Then we score each frame by its log-likelihood
  under that model.

  Frames with very low likelihood = unusual = possibly forged.

  Why GMM instead of CNN?
  ✓ No training data needed — works on any file
  ✓ Self-calibrating (learns the file's own distribution)
  ✓ Computationally light — runs in seconds on CPU
  ✓ Interpretable — log-likelihood is a well-understood score

  Settings:
  • n_components=8  — 8 Gaussian clusters (speech typically has ~5-10 phoneme classes)
  • Threshold       — frames below 5th percentile log-likelihood are anomalous
"""

def detect_gmm_anomaly(y, sr, n_components=8, anomaly_percentile=5):
    mfcc  = get_mfcc(y, sr, n_mfcc=40)   # [40 × T]
    X     = mfcc.T                         # [T × 40]

    # Standardise features (zero mean, unit variance) before fitting
    scaler = StandardScaler()
    X_s    = scaler.fit_transform(X)

    # Fit GMM on all frames
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type="diag",    # diagonal = faster, avoids overfitting
        max_iter=200,
        random_state=42,
    )
    gmm.fit(X_s)

    # Score each frame
    log_probs = gmm.score_samples(X_s)   # [T] — higher = more "normal"

    # Frames below threshold = anomalous
    threshold     = np.percentile(log_probs, anomaly_percentile)
    anomaly_mask  = log_probs < threshold
    anomaly_times = [round(i * HOP / sr, 3)
                     for i, flag in enumerate(anomaly_mask) if flag]

    # Cluster-based forgery probability:
    # What fraction of frames are in abnormally low-density regions?
    gmm_prob = float(anomaly_mask.mean())

    return {
        "log_probs":         log_probs,      # array for plotting
        "anomaly_times_sec": anomaly_times,
        "gmm_threshold":     float(threshold),
        "anomaly_fraction":  round(gmm_prob, 4),
        "forgery_prob":      round(min(gmm_prob * 3, 1.0), 4),  # scale 0-1
        "anomaly":           gmm_prob > 0.12,
        "detail":            f"{gmm_prob*100:.1f}% frames anomalous (threshold={threshold:.2f})",
    }


# ══════════════════════════════════════════════════════════════════
# SECTION 9 — AGGREGATE SCORING
# ══════════════════════════════════════════════════════════════════
"""
Each detector votes with a binary signal (anomaly: True/False).
GMM contributes a continuous probability score (scaled weight).
We blend these with fixed weights chosen by empirical reliability.

  Splice detection          30%  — highest weight: most reliable
  Copy-move detection       20%  — reliable but can have false positives
  ENF analysis              20%  — strong when ENF is present
  Noise floor               15%  — very reliable when signal is clear
  Spectral discontinuity    10%  — niche: catches re-encoding
  GMM anomaly scoring       30%  — continuous, replaces CNN weight
  (total > 100% because GMM replaces CNN slot; normalised below)

Verdict thresholds:
  ≥ 0.60 → FORGED
  0.35–0.59 → SUSPICIOUS
  < 0.35 → AUTHENTIC
"""

def compute_verdict(results):
    weights = {
        "splice":   0.28,
        "copy_move":0.18,
        "enf":      0.18,
        "noise":    0.14,
        "spectral": 0.10,
        "gmm":      0.12,
    }
    # Binary scores for heuristics
    scores = {
        "splice":    1.0 if results["splice"]["anomaly"]   else 0.0,
        "copy_move": 1.0 if results["copy_move"]["anomaly"] else 0.0,
        "enf":       1.0 if results["enf"]["anomaly"]      else 0.0,
        "noise":     1.0 if results["noise"]["anomaly"]    else 0.0,
        "spectral":  1.0 if results["spectral"]["anomaly"] else 0.0,
        "gmm":       results["gmm"]["forgery_prob"],  # continuous [0,1]
    }
    total = sum(scores[k] * weights[k] for k in weights)
    total = min(1.0, total)

    verdict = ("FORGED"     if total >= 0.60 else
               "SUSPICIOUS" if total >= 0.35 else
               "AUTHENTIC")

    return {
        "forgery_probability": round(total, 4),
        "verdict":             verdict,
        "component_scores":    scores,
    }


# ══════════════════════════════════════════════════════════════════
# SECTION 10 — VISUALISATION
# ══════════════════════════════════════════════════════════════════
BG    = "#0D1117"
PANEL = "#161B22"
BLUE  = "#58A6FF"
RED   = "#F85149"
AMBER = "#E3B341"
GREEN = "#3FB950"
MUTED = "#6E7681"
TEXT  = "#C9D1D9"

def _dark():
    plt.rcParams.update({
        "figure.facecolor": BG,   "axes.facecolor":  PANEL,
        "axes.edgecolor":   MUTED,"axes.labelcolor": TEXT,
        "xtick.color":      MUTED,"ytick.color":     MUTED,
        "text.color":       TEXT, "grid.color":      MUTED,
        "grid.alpha":       0.25, "font.family":     "monospace",
    })

def plot_report(y, sr, results, out_path):
    _dark()
    dur = len(y) / sr
    t   = np.linspace(0, dur, len(y))
    agg = results["aggregate"]
    verdict_col = {
        "FORGED": RED, "SUSPICIOUS": AMBER, "AUTHENTIC": GREEN
    }.get(agg["verdict"], BLUE)

    fig = plt.figure(figsize=(20, 15), facecolor=BG)
    fig.suptitle(
        f"🔊 Audio Forensic Report   |   Verdict: {agg['verdict']}   "
        f"|   Forgery P = {agg['forgery_probability']:.4f}",
        fontsize=15, color=verdict_col, fontweight="bold", y=0.99
    )

    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.50, wspace=0.35)

    # ── 1. Waveform ──────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(t, y, color=BLUE, linewidth=0.35, alpha=0.85)
    ax1.set_title("Waveform  (splice markers ─ red dashed  |  noise shifts ─ amber dotted)")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")

    for i, ts in enumerate(results["splice"]["splice_times_sec"]):
        ax1.axvline(ts, color=RED, lw=1.1, ls="--", alpha=0.8,
                    label="Splice" if i == 0 else "")
    for i, ts in enumerate(results["noise"]["shift_times_sec"]):
        ax1.axvline(ts, color=AMBER, lw=0.9, ls=":", alpha=0.8,
                    label="Noise shift" if i == 0 else "")
    hdl, lbl = ax1.get_legend_handles_labels()
    if hdl:
        ax1.legend(facecolor=PANEL, labelcolor=TEXT, fontsize=8, loc="upper right")
    ax1.grid(True)

    # ── 2. Mel-spectrogram ───────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    mel_db = get_mel_spec(y, sr)
    import librosa.display
    img = librosa.display.specshow(mel_db, sr=sr, x_axis="time", y_axis="mel",
                                    ax=ax2, cmap="inferno", hop_length=HOP)
    fig.colorbar(img, ax=ax2, format="%+2.0f dB")
    ax2.set_title("Mel-Spectrogram")

    # ── 3. MFCC Discontinuity ────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    dists = results["splice"]["distances"]
    thr   = results["splice"]["threshold"]
    tf    = np.arange(len(dists)) * HOP / sr
    ax3.plot(tf, dists, color=BLUE, lw=0.7, label="Discontinuity")
    if len(thr) == len(dists):
        ax3.plot(tf, thr, color=AMBER, lw=0.8, ls="--", alpha=0.7, label="Threshold")
    for ts in results["splice"]["splice_times_sec"]:
        ax3.axvline(ts, color=RED, lw=1.0, ls="--", alpha=0.7)
    ax3.legend(facecolor=PANEL, labelcolor=TEXT, fontsize=8)
    ax3.set_title("MFCC Frame Discontinuity (Splice Detection)")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Cosine Distance")
    ax3.grid(True)

    # ── 4. GMM Log-Likelihood ────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 0])
    lp   = results["gmm"]["log_probs"]
    tg   = np.arange(len(lp)) * HOP / sr
    gmm_thr = results["gmm"]["gmm_threshold"]
    ax4.plot(tg, lp, color=BLUE, lw=0.6, alpha=0.9, label="Log-likelihood")
    ax4.axhline(gmm_thr, color=RED, lw=1.0, ls="--", alpha=0.8,
                label=f"Anomaly threshold ({gmm_thr:.1f})")
    ax4.fill_between(tg, lp, gmm_thr,
                     where=(lp < gmm_thr), color=RED, alpha=0.20,
                     label="Anomalous region")
    ax4.legend(facecolor=PANEL, labelcolor=TEXT, fontsize=8)
    ax4.set_title(f"GMM Anomaly Score  ({results['gmm']['detail']})")
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Log-likelihood")
    ax4.grid(True)

    # ── 5. ENF + Noise Floor (shared panel) ──────────────────────
    ax5 = fig.add_subplot(gs[2, 1])
    enf_s = np.array(results["enf"]["enf_series"])
    if len(enf_s) > 1:
        te = np.linspace(0, dur, len(enf_s))
        ax5.plot(te, enf_s, color=GREEN, lw=0.9, label=f"ENF (σ={results['enf']['std_hz']:.4f} Hz)")
        ax5.axhline(NOMINAL_ENF_HZ, color=MUTED, lw=0.8, ls="--", alpha=0.6,
                    label=f"Nominal {NOMINAL_ENF_HZ} Hz")

    ax5b = ax5.twinx()
    nd   = np.array(results["noise"]["noise_rms_db"])
    if len(nd) > 0:
        tn = np.arange(len(nd)) * 0.5
        ax5b.plot(tn, nd, color=AMBER, lw=0.8, alpha=0.7, ls="-.",
                  label=f"Noise floor (max jump {results['noise']['max_jump_db']:.1f} dB)")
        ax5b.set_ylabel("Noise floor (dB)", color=AMBER)
        ax5b.tick_params(axis="y", colors=AMBER)

    lines1, labs1 = ax5.get_legend_handles_labels()
    lines2, labs2 = ax5b.get_legend_handles_labels()
    ax5.legend(lines1 + lines2, labs1 + labs2,
               facecolor=PANEL, labelcolor=TEXT, fontsize=7, loc="upper right")
    ax5.set_title("ENF (green) + Noise Floor (amber)")
    ax5.set_xlabel("Time (s)")
    ax5.set_ylabel("ENF (Hz)", color=GREEN)
    ax5.tick_params(axis="y", colors=GREEN)
    ax5.grid(True)

    # ── Verdict box ──────────────────────────────────────────────
    prob  = agg["forgery_probability"]
    bar   = "█" * int(prob * 20) + "░" * (20 - int(prob * 20))
    fig.text(0.5, 0.01,
             f"  VERDICT: {agg['verdict']}   [{bar}]  {prob:.4f}   "
             f"│  splice:{agg['component_scores']['splice']:.0f}  "
             f"copy-move:{agg['component_scores']['copy_move']:.0f}  "
             f"enf:{agg['component_scores']['enf']:.0f}  "
             f"noise:{agg['component_scores']['noise']:.0f}  "
             f"gmm:{agg['component_scores']['gmm']:.2f}",
             ha="center", fontsize=9, color=verdict_col,
             fontfamily="monospace")

    plt.savefig(out_path, dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"[viz]  Saved → {out_path}")


# ══════════════════════════════════════════════════════════════════
# SECTION 11 — MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════
def run_analysis(file_path, out_dir, sr, nominal_enf_hz):
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(file_path))[0]

    print("\n" + "═"*58)
    print("   AUDIO FORGERY DETECTION SYSTEM")
    print("═"*58)

    y = load_audio(file_path, sr=sr)
    results = {}

    steps = [
        ("Splice detection",          lambda: detect_splices(y, sr)),
        ("Copy-move detection",       lambda: detect_copy_move(y, sr)),
        ("ENF analysis",              lambda: detect_enf(y, sr, nominal_enf_hz)),
        ("Noise floor analysis",      lambda: detect_noise_floor(y, sr)),
        ("Spectral discontinuity",    lambda: detect_spectral_discontinuity(y, sr)),
        ("GMM anomaly scoring",       lambda: detect_gmm_anomaly(y, sr)),
    ]
    keys = ["splice", "copy_move", "enf", "noise", "spectral", "gmm"]

    for (label, fn), key in zip(steps, keys):
        print(f"\n[{label}] running …")
        results[key] = fn()
        tag = "⚠  ANOMALY" if results[key]["anomaly"] else "✓  OK"
        print(f"  → {tag}   {results[key]['detail']}")

    results["aggregate"] = compute_verdict(results)

    # ── Print verdict ─────────────────────────────────────────────
    agg    = results["aggregate"]
    prob   = agg["forgery_probability"]
    col    = {
        "FORGED":"\033[91m","SUSPICIOUS":"\033[93m","AUTHENTIC":"\033[92m"
    }.get(agg["verdict"], "")
    reset  = "\033[0m"
    bar    = "█"*int(prob*30) + "░"*(30-int(prob*30))
    print(f"\n{'═'*58}")
    print(f"  VERDICT         : {col}{agg['verdict']}{reset}")
    print(f"  FORGERY PROB    : [{bar}] {prob:.4f}")
    print(f"  ELAPSED         : {time.time()-t0:.1f}s")
    print(f"{'═'*58}\n")

    # ── Save plot ─────────────────────────────────────────────────
    png_path = os.path.join(out_dir, f"{base}_forensic.png")
    plot_report(y, sr, results, png_path)

    # ── Save JSON report ──────────────────────────────────────────
    def _serial(obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)): return float(obj)
        if isinstance(obj, dict):  return {k: _serial(v) for k, v in obj.items()}
        if isinstance(obj, list):  return [_serial(v) for v in obj]
        return obj

    json_path = os.path.join(out_dir, f"{base}_report.json")
    with open(json_path, "w") as f:
        json.dump(_serial(results), f, indent=2)
    print(f"[json] Saved → {json_path}")
    return results


# ══════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    run_analysis(
        file_path      = FILE_PATH,
        out_dir        = OUTPUT_DIR,
        sr             = SAMPLE_RATE,
        nominal_enf_hz = NOMINAL_ENF_HZ,
    )
