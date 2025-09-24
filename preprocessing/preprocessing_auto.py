# Libraries 
# === Core ===
import os
import numpy as np
import pandas as pd
import scipy.io

# === Scipy utilities ===
from scipy.io import loadmat, savemat
from scipy.signal import welch, butter, filtfilt, hilbert
from scipy.stats import zscore
from scipy.spatial.distance import cosine
from scipy.fft import fft

# === Visualization ===
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import SymLogNorm

# === MNE: core + IO ===
import mne
from mne import create_info, read_labels_from_annot
from mne.io import RawArray

# === MNE: preprocessing ===
from mne.preprocessing import ICA, create_ecg_epochs

# === MNE: datasets ===
from mne.datasets import fetch_fsaverage

# === MNE: source localization ===
from mne.minimum_norm import make_inverse_operator, apply_inverse_raw

# === Connectivity ===
from mne_connectivity import envelope_correlation, symmetric_orth

# === Filtering ===
from mne.filter import filter_data


# Load .txt data
def load_metadata_txt(ruta_archivo, encoding="utf-8"):
    metadata = {}
    try:
        with open(ruta_archivo, "r", encoding=encoding) as f:
            for line in f:
                if ":" in line:
                    key, value = line.strip().split(":", 1)
                    key = key.strip()
                    value = value.strip()

                    # Try to convert values to correct types
                    if value.lower() in ["true", "false"]:
                        value = value.lower() == "true"
                    elif value.lower() == "nan":
                        value = None
                    else:
                        try:
                            if "." in value:
                                value = float(value)
                            else:
                                value = int(value)
                        except ValueError:
                            pass  # keep as string

                    metadata[key] = value
    except Exception as e:
        print(f"Error al cargar el archivo TXT: {e}")
    
    return metadata



# Load cvs as df
def load_or_create_patient_csv(ruta_archivo, verbose=True):
    columnas = ["patient", "hospital", "age", "sex", "CPC", "BCI","outcome"]

    if os.path.isfile(ruta_archivo):
        # Load existing CSV
        df = pd.read_csv(ruta_archivo)
        if verbose:
            print(f"Archivo encontrado y cargado: {ruta_archivo}")

        # Ensure required columns exist
        for col in columnas:
            if col not in df.columns:
                df[col] = None
                if verbose:
                    print(f"Columna faltante añadida: {col}")

    else:
        # Create new DataFrame with empty columns
        df = pd.DataFrame(columns=columnas)
        df.to_csv(ruta_archivo, index=False)
        if verbose:
            print(f"Archivo no encontrado, creado nuevo: {ruta_archivo}")

    return df


# Load .mat and .hea data
def load_signal_pair(mat_file, hea_file):
    # Cargar matriz desde archivo .mat
    mat = loadmat(mat_file)
    signal = mat['val']  # [n_channels, n_samples]

    # Leer archivo .hea
    with open(hea_file, 'r') as f:
        lines = f.readlines()

    header_main = lines[0].strip().split()
    num_channels = int(header_main[1])

    # Frecuencia de muestreo (fs) está en la primera línea, tercer campo
    fs = float(lines[0].split()[2])

    channel_lines = [line for line in lines[1:] if len(line.split()) >= 9]
    # Extraer nombre del canal (9º campo en cada línea válida)
    channels = [line.split()[8] for line in channel_lines]

    assert signal.shape[0] == len(channels), f"Canales en {mat_file} no coinciden con .hea"

    return {
        "signal": signal,
        "channels": channels,
        "fs": fs
    }




# Create raw arrays
def create_raw_array(signal, channels, sfreq, group):
    eeg_names = ['Fp1', 'Fp2', 'F7', 'F8', 'F3', 'F4', 'T3', 'T4', 'C3', 'C4',
                 'T5', 'T6', 'P3', 'P4', 'O1', 'O2', 'Fz', 'Cz', 'Pz', 'Fpz', 'Oz', 'F9']
    ecg_names = ['ECG', 'ECG1', 'ECG2', 'ECGL', 'ECGR']

    ch_types = []
    for ch in channels:
        if ch in eeg_names:
            ch_types.append('eeg')
        elif ch in ecg_names:
            ch_types.append('ecg')
        else:
            ch_types.append('misc')

    info = mne.create_info(ch_names=channels, sfreq=sfreq, ch_types=ch_types)

    # Solo escalar si el grupo es EEG (o si el canal es eeg)
    if all(t == 'eeg' for t in ch_types):
        raw = mne.io.RawArray(signal * 1e-6, info, verbose=False)
    else:
        raw = mne.io.RawArray(signal, info, verbose=False)

    raw.info['description'] = f'{group} data'
    return raw



# Apply Band-pass filter
def preprocess_raw_signal(raw, signal_type="eeg", l_freq=1., h_freq=40.):
    raw_copy = raw.copy()

    # Apply EEG montage if it's EEG
    if signal_type.lower() == "eeg":
        montage = mne.channels.make_standard_montage("standard_1020")
        raw_copy.set_montage(montage)

    # Filter depending on signal type
    raw_filtered = raw_copy.filter(l_freq=l_freq, h_freq=h_freq, picks=signal_type.lower())

    return raw_filtered


# Identify and interpolate bad channels
def reject_and_interpolate(raw, picks="eeg", flat_thresh=1e-6, noisiness_z=4.0, verbose=True):
    raw_copy = raw.copy().load_data()
    data = raw_copy.get_data(picks=picks)
    ch_names = np.array(raw_copy.ch_names)[mne.pick_types(raw_copy.info, **{picks: True, "exclude": []})]

    bads = []

    # Detectar canales planos
    ptps = np.ptp(data, axis=1)
    bads.extend(ch_names[np.where(ptps < flat_thresh)[0]])

    # Detectar canales ruidosos (varianza anómala)
    variances = data.var(axis=1)
    vz = (variances - variances.mean()) / variances.std()
    bads.extend(ch_names[np.where(np.abs(vz) > noisiness_z)[0]])

    # Eliminar duplicados y añadir a info['bads']
    bads = list(set(bads))
    raw_copy.info['bads'].extend([ch for ch in bads if ch not in raw_copy.info['bads']])

    if verbose:
        print("Canales malos detectados:", bads if bads else "Ninguno")

    # Interpolar
    if bads:
        raw_interp = raw_copy.interpolate_bads(reset_bads=True)
    else:
        raw_interp = raw_copy

    return raw_interp


# Remove artefacts with ICA
def apply_ica_ecg_removal(filtered_eeg, filtered_ecg, ecg_ch_name="ECG",
                          n_components=0.95, random_state=97, max_iter="auto",
                          verbose=True):
    # --- 1. Combine EEG and ECG channels ---
    raw_combined = filtered_eeg.copy().add_channels([filtered_ecg], force_update_info=True)

    # --- 2. Fit ICA model ---
    ica = ICA(n_components=n_components, random_state=random_state, max_iter=max_iter)
    ica.fit(raw_combined)

    # --- 3. Detect ECG-related ICA components ---
    ecg_inds, scores = ica.find_bads_ecg(raw_combined, ch_name=ecg_ch_name)
    if verbose:
        print("Automatically detected ECG-related components:", ecg_inds)

    # --- 3.1 Fallback if no components detected ---
    if len(ecg_inds) == 0:
        manual_ecg = int(np.argmax(scores))
        if verbose:
            print(f"Forcing manual exclusion of component {manual_ecg}")
        ica.exclude = [manual_ecg]
    else:
        ica.exclude = ecg_inds

    if verbose:
        print("Excluded components:", ica.exclude)

    # --- 4. Apply ICA cleaning ---
    raw_clean = raw_combined.copy()
    ica.apply(raw_clean)

    # --- 5. Keep only EEG channels ---
    raw_eeg_clean = raw_clean.copy().pick(picks="eeg")

    # --- 6. Sanity check ---
    data_orig, _ = filtered_eeg[:, :1000]
    data_clean, _ = raw_eeg_clean[:, :1000]

    if np.allclose(data_orig, data_clean):
        if verbose:
            print("ICA had no effect (signals are identical).")
    else:
        if verbose:
            print("ICA applied successfully (signals differ).")

    return raw_eeg_clean



# ------------------------------------------------------------------------

# Other dependencies
def detect_rpp(data, fs, win_sec=4, step_sec=2, amp_thresh=2, entropy_thresh=0.7, freq_band=(0.5, 7)):
    def spectral_entropy(sig, fs):
        f, Pxx = welch(sig, fs)
        Pxx /= np.sum(Pxx)
        return -np.sum(Pxx * np.log2(Pxx + np.finfo(float).eps))

    win = int(win_sec * fs)
    step = int(step_sec * fs)
    rpp_epochs = []

    rms_bg = np.sqrt(np.mean(data**2))
    for start in range(0, len(data) - win, step):
        segment = data[start:start + win]
        amp = np.sqrt(np.mean(segment**2))
        se = spectral_entropy(segment, fs)
        f, Pxx = welch(segment, fs)
        dom_freq = f[np.argmax(Pxx)]
        if amp > amp_thresh * rms_bg and se < entropy_thresh and freq_band[0] <= dom_freq <= freq_band[1]:
            rpp_epochs.append((start, start + win))

    return merge_epochs(rpp_epochs, step)


def merge_epochs(epochs, step):
    if not epochs:
        return []
    merged = [epochs[0]]
    for current in epochs[1:]:
        if current[0] <= merged[-1][1] + step:
            merged[-1] = (merged[-1][0], max(merged[-1][1], current[1]))
        else:
            merged.append(current)
    return merged


def compute_bci_star(data, fs, win_sec=1, step_sec=0.5, thresh_factor=0.2):
    win = int(win_sec * fs)
    step = int(step_sec * fs)
    thresh = thresh_factor * np.std(data)
    continuity = []
    for start in range(0, len(data) - win, step):
        segment = data[start:start + win]
        continuity.append(np.std(segment) > thresh)
    return np.mean(continuity)


def compute_relative_discharge_power(data, fs, rpp_epochs, band=(1, 30)):
    total_power = bandpower(data, fs, band)
    rpp_power = sum(bandpower(data[start:end], fs, band) for start, end in rpp_epochs)
    return rpp_power / total_power if total_power > 0 else 0


def bandpower(data, fs, band):
    f, Pxx = welch(data, fs)
    idx_band = np.logical_and(f >= band[0], f <= band[1])
    return np.sum(Pxx[idx_band])


def compute_discharge_frequency(data, fs, rpp_epochs):
    total_duration = sum((end - start) for start, end in rpp_epochs) / fs
    total_zero_crossings = 0
    for start, end in rpp_epochs:
        segment = data[start:end]
        zero_crossings = np.sum(np.diff(np.sign(segment)) != 0)
        total_zero_crossings += zero_crossings
    return (total_zero_crossings / 2) / total_duration if total_duration > 0 else 0


def compute_shape_similarity(data, fs, rpp_epochs):
    segments = [zscore(data[start:end]) for start, end in rpp_epochs if end - start > 1]
    similarities = []
    for i in range(len(segments)):
        for j in range(i + 1, len(segments)):
            min_len = min(len(segments[i]), len(segments[j]))
            sim = 1 - cosine(segments[i][:min_len], segments[j][:min_len])
            similarities.append(sim)
    return np.mean(similarities) if similarities else 0


def artifact_detection(eeg, fs, amplitude_thresh=500):
    """
    Simple artifact detection: excludes channels with large signal excursions.
    eeg: samples x channels
    returns: binary array of excluded channels (1 = exclude)
    """
    return np.any(np.abs(eeg) > amplitude_thresh, axis=0).astype(int)


def calculate_qeeg_features(eeg, fs):
    """
    Compute Background Continuity Index (BCI) and
    Burst Suppression Amplitude Ratio (BSAR) per Ruijter et al. 2018.
    """
    # --- Parameters
    cutoff_value = 10  # microvolt threshold for suppression
    min_supp_duration = 0.5  # seconds
    min_burst_duration = 0.2  # seconds
    bsar_lim = [0.01, 0.99]
    min_channels = 12

    n_samples, n_channels = eeg.shape
    excl_channel = artifact_detection(eeg, fs)

    if np.sum(~excl_channel) < min_channels:
        return np.nan, np.nan

    eeg = eeg[:, ~excl_channel]

    # Bandpass filter
    b, a = butter(3, [0.5, 30], btype='bandpass', fs=fs)
    eeg = filtfilt(b, a, eeg, axis=0)

    eeg_supp = np.zeros_like(eeg, dtype=int)

    for ch in range(eeg.shape[1]):
        ch_data = eeg[:, ch]
        low_amp = np.abs(ch_data) < (cutoff_value / 2)
        suppress = np.zeros_like(ch_data, dtype=int)

        # Detect suppressions
        padded = np.concatenate([[0], low_amp.astype(int), [0]])
        starts = np.where(np.diff(padded) == 1)[0]
        ends = np.where(np.diff(padded) == -1)[0] - 1

        for s, e in zip(starts, ends):
            if e - s + 1 >= int(min_supp_duration * fs):
                suppress[s:e + 1] = 1

        # Remove short bursts
        padded2 = np.concatenate([[1], suppress, [1]])
        burst_starts = np.where(np.diff(padded2) == -1)[0]
        burst_ends = np.where(np.diff(padded2) == 1)[0] - 1

        for s, e in zip(burst_starts, burst_ends):
            if e - s + 1 < int(min_burst_duration * fs):
                suppress[s:e + 1] = 1

        eeg_supp[:, ch] = suppress

    # Remove filter edges
    eeg = eeg[fs:-fs, :]
    eeg_supp = eeg_supp[fs:-fs, :]

    bci_all = []
    bsar_all = []

    for ch in range(eeg.shape[1]):
        signal = eeg[:, ch]
        suppress = eeg_supp[:, ch]

        bci = 1 - np.mean(suppress)
        bci_all.append(bci)

        if bsar_lim[0] <= bci <= bsar_lim[1]:
            burst_amp = np.std(signal[suppress == 0])
            supp_amp = np.std(signal[suppress == 1]) if np.any(suppress) else 1
            bsar = burst_amp / supp_amp if supp_amp != 0 else 1
        else:
            bsar = 1

        bsar_all.append(bsar)

    return np.mean(bci_all), np.mean(bsar_all)

# ---------------------------------------------------------------------------

# Apply conditional filter based on BCI* score and RPP detection
def conditional_filter(raw_eeg_limpio, bci_thresh=0.5, verbose=True):
    # --- Extract data and sampling frequency ---
    data = raw_eeg_limpio.get_data(picks="eeg")
    data_flat = data[-2, :]  # Example: Cz or last-but-one channel
    fs = raw_eeg_limpio.info["sfreq"]

    # --- Compute quality metrics ---
    bci_star = compute_bci_star(data_flat, fs)
    rpp_epochs = detect_rpp(data_flat, fs)

    if verbose:
        print(f"BCI*: {bci_star:.2f}, RPPs detected: {len(rpp_epochs)}")

    # --- Apply filter conditions ---
    if bci_star > bci_thresh and len(rpp_epochs) == 0:
        if verbose:
            print("✓ Conditions met: signal accepted")
        signal_out = raw_eeg_limpio
    else:
        if verbose:
            print("✘ Conditions failed: signal rejected")
        signal_out = None

    return signal_out



# Get and check fsaverage
def get_and_check_fsaverage(verbose=False):
    # Download fsaverage if not present
    fs_dir = fetch_fsaverage(verbose=verbose)
    subjects_dir = os.path.dirname(fs_dir)
    subject = "fsaverage"

    # Paths to cortical surfaces
    surf_path = os.path.join(subjects_dir, subject, "surf")
    lh_white = os.path.join(surf_path, "lh.white")
    rh_white = os.path.join(surf_path, "rh.white")

    # Validate existence of required files
    if not (os.path.isfile(lh_white) and os.path.isfile(rh_white)):
        raise FileNotFoundError(
            "Missing cortical surface files (lh.white/rh.white). "
            "The fsaverage download may have failed."
        )
    else:
        if verbose:
            print("fsaverage is available and correctly configured.")

    return subjects_dir, subject


# Setup source space and BEM model
def setup_src_and_bem(subject, subjects_dir, spacing="oct6", conductivity=(0.3, 0.006, 0.3), verbose=True):
    # --- Source space setup ---
    src = mne.setup_source_space(
        subject=subject,
        spacing=spacing,
        subjects_dir=subjects_dir,
        add_dist=False
    )

    # --- BEM model setup ---
    bem_model = mne.make_bem_model(
        subject=subject,
        ico=4,
        conductivity=conductivity,
        subjects_dir=subjects_dir
    )
    bem = mne.make_bem_solution(bem_model)

    if verbose:
        print(f"Source space ({spacing}) and BEM model created for subject: {subject}")

    return src, bem


# Create the forward model for EEG source localization and run inverse solution with eLORETA
def run_source_reconstruction(
    raw_eeg_limpio,
    src,
    bem,
    trans="fsaverage",
    montage="standard_1020",
    mindist=5.0,
    method="eLORETA",
    duration=30, # 30 seconds
    snr=3.0,
    verbose=True,
):
    # --- 1. Apply montage ---
    montage_obj = mne.channels.make_standard_montage(montage)
    raw_con_montaje = raw_eeg_limpio.copy().set_montage(montage_obj)

    # --- 2. Apply average reference (mandatory for EEG) ---
    raw_con_montaje.set_eeg_reference("average", projection=True)

    # --- 3. Forward solution ---
    fwd = mne.make_forward_solution(
        raw_con_montaje.info,
        trans=trans,
        src=src,
        bem=bem,
        eeg=True,
        meg=False,
        mindist=mindist,
    )

    # --- 4. Noise covariance ---
    noise_cov = mne.compute_raw_covariance(raw_con_montaje, tmin=0, tmax=None)

    # --- 5. Inverse operator ---
    inverse_operator = make_inverse_operator(
        raw_con_montaje.info, fwd, noise_cov, loose=0.2, depth=0.8
    )

    # --- 6. Time window ---
    sfreq = raw_con_montaje.info["sfreq"]
    start = 0
    stop = int(sfreq * duration)

    if verbose:
        print(f"Sampling frequency: {sfreq} Hz")
        print(f"Analyzing first {duration} seconds ({start}:{stop} samples).")

    # --- 7. Regularization ---
    lambda2 = 1.0 / snr**2

    # --- 8. Apply inverse solution ---
    stc_fragment = apply_inverse_raw(
        raw_con_montaje,
        inverse_operator,
        lambda2=lambda2,
        method=method,
        start=start,
        stop=stop,
        buffer_size=1000,
        pick_ori=None,
    )

    if verbose:
        print(f"Inverse solution ({method}) computed successfully.")

    return stc_fragment




# Extract virtual electrodes by averaging source activity in predefined regions
def extract_virtual_electrodes(stc_fragment, subjects_dir, subject="fsaverage",
                               parc="aparc.a2009s",  verbose=True):
     # --- Load atlas labels ---
    labels = read_labels_from_annot(subject=subject, parc=parc, subjects_dir=subjects_dir)

    # --- Select regions ---
    # Indices Gong 2009 (0-based)
    gong_indices = np.array([
        27,21,5,25,9,15,3,7,11,13,23,19,69,1,17,57,59,61,63,65,67,49,51,53,
        43,45,47,55,79,81,85,89,83,87,39,31,33,35,29,28,22,6,26,10,16,4,8,
        12,14,24,20,70,2,18,58,60,62,64,66,68,50,52,54,44,46,48,56,80,82,86,
        90,84,88,40,32,34,36,30
    ]) - 1

    labels_selected = [labels[i] for i in gong_indices if i < len(labels)]

    # --- Allocate virtual electrodes matrix ---
    n_regions = len(labels_selected)
    n_times = len(stc_fragment.times)
    virtual_electrodes = np.zeros((n_regions, n_times))

    # --- Extract average per region ---
    for i, label in enumerate(labels_selected):
        stc_label = stc_fragment.in_label(label)  # extract sources in the region
        virtual_electrodes[i, :] = stc_label.data.mean(axis=0)  # average spatially

    if verbose:
        print(f"Extracted {n_regions} virtual electrodes from atlas {parc}.")

    return virtual_electrodes



# Save a NumPy array as a .mat file
def save_array_as_mat(array, ruta_archivo, nombre_variable):
    if not isinstance(array, np.ndarray):
        raise TypeError("El argumento 'array' debe ser un numpy.ndarray")
    
    # Ensure the directory exists
    directory = os.path.dirname(ruta_archivo)
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directorio creado: {directory}")
    
    try:
        scipy.io.savemat(ruta_archivo, {nombre_variable: array})
        print(f"Archivo guardado correctamente en: {ruta_archivo}")
    except Exception as e:
        print(f"Error al guardar el archivo .mat: {e}")



# Save a pandas DataFrame to a CSV file
def save_dataframe(df, filepath,mode="w", index=False):
    filepath
    # Ensure folder exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    # Save DataFrame
    try:
        df.to_csv(filepath, mode=mode, index=index)
        print(f"DataFrame saved at: {filepath}")
    except Exception as e:
        print(f"Error saving DataFrame: {e}")

# -------------------------------------------------------------------
def bandpass_filter(data, low, high, fs, order=4):
    b, a = butter(order, [low / (fs / 2), high / (fs / 2)], btype='band')
    return filtfilt(b, a, data, axis=1)


def process_virtual_electrodes(virtual_electrodes, fs=500, low_freq=8.0, high_freq=12.0, n_ROIs=78):
    # --- 0. Ensure regions are rows ---
    if virtual_electrodes.shape[0] != n_ROIs and virtual_electrodes.shape[1] == n_ROIs:
        virtual_electrodes = virtual_electrodes.T

    if virtual_electrodes.shape[0] != n_ROIs:
        raise ValueError(
            f"Input must have {n_ROIs} regions, but got shape {virtual_electrodes.shape}"
        )

    signals = virtual_electrodes
    n_ROIs, n_samples = signals.shape

    # --- 1. Bandpass filter ---
    signals_filtered = bandpass_filter(signals, low_freq, high_freq, fs)

    # --- 2. Rank check ---
    rank = np.linalg.matrix_rank(signals_filtered)
    print(f"Virtual electrodes - Rank before correction: {rank}/{n_ROIs}")

    # --- 3. Correct rank if necessary ---
    if rank < n_ROIs:
        U, S, Vt = np.linalg.svd(signals_filtered, full_matrices=False)
        k = min(len(S), n_ROIs)
        signals_filtered = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]

        if k < n_ROIs:
            padding = np.zeros((n_ROIs - k, signals_filtered.shape[1]))
            signals_filtered = np.vstack((signals_filtered, padding))

    # --- 4. Symmetric orthogonalization ---
    label_ts_orth = symmetric_orth(signals_filtered)

    # --- 5. Hilbert transform → envelope ---
    analytic_signal = hilbert(label_ts_orth, axis=1)
    envelope = np.abs(analytic_signal)

    # --- 6. Pearson correlation ---
    corr_envelope = np.corrcoef(envelope)

    return corr_envelope
# -------------------------------------------------------------------

# Clean signal
def clean_signal(dict_patient):
    # Load raw data
    eeg_data = load_signal_pair(dict_patient["eeg_mat"], dict_patient["eeg_hea"])
    ecg_data = load_signal_pair(dict_patient["ecg_mat"], dict_patient["ecg_hea"])
    # Create raw array
    raw_eeg = create_raw_array(eeg_data['signal'], eeg_data['channels'], eeg_data['fs'], 'EEG')
    raw_ecg = create_raw_array(ecg_data['signal'], ecg_data['channels'], ecg_data['fs'], 'ECG')
    # Apply band-pass filter
    filtered_eeg = preprocess_raw_signal(raw_eeg, signal_type="eeg")
    filtered_ecg = preprocess_raw_signal(raw_ecg, signal_type="ecg")
    # Interpolate bad channels
    raw_eeg_int = reject_and_interpolate(filtered_eeg)
    raw_ecg_int = reject_and_interpolate(filtered_ecg, picks="ecg")
    # Remove actefacts
    raw_eeg_clean = apply_ica_ecg_removal(raw_eeg_int, raw_ecg_int)

    # Load txt data
    patient_metadata = load_metadata_txt(dict_patient["txt"])
    return raw_eeg_clean, patient_metadata


# Función para reconstuir la señal
def reconstruct_signal(raw_eeg_clean):
    # Check fsaverage
    subjects_dir, subject = get_and_check_fsaverage(verbose=True)
    # BEM model
    src, bem = setup_src_and_bem(subject, subjects_dir, spacing="oct6")
    # Forward model and ineverse solution
    stc_fragment= run_source_reconstruction(raw_eeg_clean, src, bem, method="eLORETA", duration=30)
    # Virtual electrodes
    virtual_electrodes = extract_virtual_electrodes(stc_fragment, subjects_dir, subject="fsaverage", parc="aparc.a2009s")
    return virtual_electrodes


# Update csv
def update_df(patient_metadata, df_patient, patient_id, bci_score):
    # Build a row with only the required fields
    row = {
        "patient": patient_id,
        "hospital": patient_metadata.get("Hospital"),
        "age": patient_metadata.get("Age"),
        "sex": patient_metadata.get("Sex"),
        "CPC": patient_metadata.get("CPC"),
        "BCI": bci_score,
        "outcome": patient_metadata.get("Outcome"),
    }

    # Convert to DataFrame row
    new_row = pd.DataFrame([row])

    # Append row to existing DataFrame
    df_patient = pd.concat([df_patient, new_row], ignore_index=True)

    return df_patient



# Test BCI score for a patient, update patient registry, and reconstruct signal if passed
def test_bci_score(raw_eeg_clean, patient_metadata, base_dir, csv_path, patient_id, bci_thresh=0.5):
    # Load patient registry
    df_patients = load_or_create_patient_csv(csv_path)

    # Apply conditional filter
    signal_out = conditional_filter(raw_eeg_clean, bci_thresh=bci_thresh)

    if signal_out is not None:
        # Reconstruct virtual electrodes
        virtual_electrodes = reconstruct_signal(raw_eeg_clean)
        
        save_array_as_mat(virtual_electrodes, f"{base_dir}{patient_id}/filtered_raw/{patient_id}_filtered_raw.mat", "data")
        # Update DataFrame with PASS
        df_patients = update_df(patient_metadata, df_patients, patient_id, "Pass")
        # Upload csv
        save_dataframe(df_patients, csv_path, mode="w", index=False)

    else:
        virtual_electrodes = None
        # Update DataFrame with FAIL
        df_patients = update_df(patient_metadata, df_patients, patient_id,  "Fail")
        # Upload csv
        save_dataframe(df_patients, csv_path, mode="w", index=False)

    return virtual_electrodes

# Apply symmetric orthogonalization
def get_corr_matrix(virtual_electrodes, id_patient, base_dir):
    corr_matrix = process_virtual_electrodes(virtual_electrodes, fs=500, low_freq=8.0, high_freq=12.0, n_ROIs=78)
    # Save matix
    save_array_as_mat(corr_matrix, f"{base_dir}{id_patient}/filtered_raw/{id_patient}_corr_matrix.mat", "data")



# Collect EEG, ECG, and TXT file paths for each patient folder
def collect_patient_files(base_dir):
    patients_data = []

    for patient_id in os.listdir(base_dir):
        patient_dir = os.path.join(base_dir, patient_id)
        if not os.path.isdir(patient_dir):
            continue  # skip if it's not a folder

        patient_files = {
            "eeg_mat": None,
            "eeg_hea": None,
            "ecg_mat": None,
            "ecg_hea": None,
            "txt": None,
        }

        for fname in os.listdir(patient_dir):
            fpath = os.path.join(patient_dir, fname)

            if fname.endswith("_EEG.mat"):
                patient_files["eeg_mat"] = fpath
            elif fname.endswith("_EEG.hea"):
                patient_files["eeg_hea"] = fpath
            elif fname.endswith("_ECG.mat"):
                patient_files["ecg_mat"] = fpath
            elif fname.endswith("_ECG.hea"):
                patient_files["ecg_hea"] = fpath
            elif fname.endswith(".txt") and fname.startswith(patient_id):
                patient_files["txt"] = fpath

        # Save as {id_paciente: {...}}
        patients_data.append({patient_id: patient_files})

    return patients_data





def iterate_patients(base_dir):
    csv_path = "patients.csv"
    patients_data = collect_patient_files(base_dir)

    for elemento in patients_data:
        # Each element is structured as {patient_id: patient_files}
        patient_id, patient_files = next(iter(elemento.items()))
        print(f"\nProcesando paciente {patient_id}: {patient_files}")  
        try:
            print(f"\nProcesando paciente {patient_id}: {patient_files}")  
            # Clean signal and get metadata
            #raw_eeg_clean, patient_metadata = clean_signal(patient_files)

            # Test BCI score → returns virtual electrodes
            # virtual_electrodes = test_bci_score(
            #     raw_eeg_clean,
            #     patient_metadata,
            #     base_dir,
            #     csv_path,
            #     patient_id,
            #     bci_thresh=0.5
            # )

            # Skip if virtual electrodes not available
            # if virtual_electrodes is None:
            #     print(f"Skipping {patient_id}: no virtual electrodes (Fail case).")
            #     continue  

            # If electrodes exist → process correlation
            #get_corr_matrix(virtual_electrodes, patient_id, base_dir)

        except Exception as e:
            print(f"Error processing {patient_id}: {e}")


base_dir = "/home/mriesn/Universidad/2025-2/Tesis/Code/data/"
iterate_patients(base_dir)