'''
[ Archivo .dat Crudo (Cualquier Fs) ]
                  │
                  ▼
   [ WFDB FRONT-END & SELECCIÓN ] ───► Extrae únicamente la fila de la "Derivación II"
                  │
                  ▼
     [ DSP: REMUESTREO ADAPTATIVO ] ──► Si Fs = 500 Hz: Pasa directo.
                  │                     Si Fs ≠ 500 Hz: Resampleo.
                  ▼
       [ FILTRADO DE FRECUENCIA ] ───► Butterworth Pasa-Banda (0.5 - 40 Hz) + Notch (50/60 Hz)
                  │
                  ▼
       [ DETECTOR PAN-TOMPKINS ] ───► Localiza picos R + Filtro de guarda de amplitud
                  │
                  ▼
     [ SEGMENTACIÓN DE VENTANAS ] ───► Ventanas de 1500 muestras centradas en el pico R
                  │
                  ▼
       [ NORMALIZACIÓN LOCAL ] ───► Z-Score por ventana (Inmunidad a cambios de ganancia)
                  │
                  ▼
   [ EXTRACCIÓN FEATURES RITMO ] ───► Diferencia de Transición RR (Historial de 8 latidos)
                  │
                  ▼
 [ RED MULTIMODAL CONGELADA ] ───► CNN + BiLSTM + SE + PolyFocal (F1-Macro: 0.6930)
                  │
                  ▼
   [ POST-PROCESO DE UMBRALES ] ───► Multiplicador 1.5 a Clase Normal (Suprime Falsos Positivos)
                  │
                  ▼
         [ DIAGNÓSTICO FINAL ] ───► Clasificación AAMI: N, S, V, Q
'''

import os
import wfdb
import numpy as np
import pandas as pd
from typing import List, Dict
from scipy.signal import butter, filtfilt, iirnotch, find_peaks

#-------------------------------------------------- Patient to test ----------------------------------------------------------
TEST_PATIENT = "01004_hr"
#==============================================================================================================================

#-------------------------------------------- Download de data (PTB-XL in this case) -----------------------------------------
ptb_xl_dir = os.path.join("data/test_data", "ptb_xl_data")
'''
os.makedirs(ptb_xl_dir, exist_ok=True)

if not os.listdir(ptb_xl_dir):
    print("Downloading the PTB-XL database...")
    wfdb.dl_database('ptb-xl', dl_dir=ptb_xl_dir, overwrite=False)
    print(f"Database correctly download into {ptb_xl_dir}")
else:
    print("Database already downloaded!")
#==============================================================================================================================
'''
#----------------------------------------------- Example of patient number 803 ------------------------------------------------
patient_values = wfdb.rdrecord(ptb_xl_dir+"/"+TEST_PATIENT)
psignal_patient = patient_values.p_signal
channel_name = patient_values.sig_name

print(f"Values contained: \n{psignal_patient}")
print(f"The channel names are: {channel_name}")
print(f"The frequency of this dataset is {patient_values.fs}Hz")
#==============================================================================================================================

#--------------------------------------------------- Functions declaration -----------------------------------------------------
def resampling(signal, fs_signal: int, fs_target: int=500):
    """In case the ecg comes in a different frequency tha 500Hz, the function resample it

    Args:
        signal (list): _description_
        fs_signal (int): _description_
        fs_target (int, optional): _description_. Defaults to 500.

    Returns:
        list: _description_
    """
    if fs_signal == fs_target:
        return signal

    num_samples_target = int(len(signal) * (fs_target / fs_signal))

    original_indexes = np.arange(len(signal)) #muestras equidistantes del largo de la señal
    new_indexes = np.linspace(0, len(signal) - 1, num_samples_target) #division del número de muestras de forma equidistante

    signal_resampled = np.interp(new_indexes, original_indexes, signal) #rellena los huecos de los nuevos índices basándose en los viejos

    return signal_resampled

def reshape(signal: list[list]) -> list[list]:
    """turns the signal of 12 leads into the 2 that the training was done, the MLII and V5

    Args:
        signal (list): signal of a patient

    Returns:
        list: a list of lists, each of one correspond to each lead
    """
    p_signal = signal.p_signal
    channel_name = signal.sig_name

    ml2_id = None
    v5_id = None

    for idx, lead in enumerate(channel_name):
        if lead in ["II", "MLII"]:
            ml2_id = idx
        elif lead == "V5":
            v5_id = idx
        else:
            continue

    channel_ii = p_signal[:, ml2_id]
    channel_v5 = p_signal[:, v5_id]

    processed_signal = np.array([channel_ii, channel_v5])
    return processed_signal

def apply_filters(signal_reshaped: list[list], fs: int=500) -> list[list]:
    """_summary_aply the butterworth and north filters to the signals so low and high frequencies are cut off

    Args:
        signal (list[list]): the signal that has been reshaped
        fs (int, optional): frequency of the signal, must be 500Hz. Defaults to 500.

    Returns:
        list[list]: the same reshaped signal but with the filters applied
    """
    nyquist = 0.5 * fs

    low_cut = 0.5 / nyquist
    high_cut = 40 / nyquist

    b_butter, a_butter = butter(N=4, Wn=[low_cut, high_cut], btype='bandpass')
    b_notch, a_notch = iirnotch(w0=50.0, Q=30.0, fs=fs)

    filtered_channels = []

    for channel in signal_reshaped:
        f_butter = filtfilt(b_butter, a_butter, channel)
        f_total = filtfilt(b_notch, a_notch, f_butter)

        filtered_channels.append(f_total)

    return np.array(filtered_channels)

def find_QRS_peak(signal_filtered, fs: int=500) -> np.ndarray:
    """detects the QRS peaks using Pan-Tompkins, derivating the signal and then doing its square to make them positive

    Args:
        signal_filtered (_type_): signals that has been cleaned
        fs (int, optional): frequency of the ecg. Defaults to 500.

    Returns:
        np.ndarray: find_peaks is a tuple of ndarray, dict(preperties), we only need the array
    """
    channel_ml2 = signal_filtered[0]

    prepared_signal = np.diff(channel_ml2)**2
    threshold = prepared_signal.max()*0.15

    peaks, _ = find_peaks(
        prepared_signal,
        height=threshold,
        distance=round(fs*0.3)
    )

    return peaks

def segmentation_window(signal_filtered, peaks: np.ndarray, window_size: int=1500) -> list[dict]:
    """divide the whole signal into windows of 1500 samples

    Args:
        signal_filtered (_type_): signal that has been filtered
        peaks (np.ndarray): the peaks of the signal
        window_size (int, optional): the size of the windows that the signal is going to be divided to. Defaults to 1500.

    Returns:
        list[dict]: a list containing dictionaries for each window
    """
    half_window = window_size // 2
    num_samples = signal_filtered.shape[1]

    window_beats = []
    for idx, peak in enumerate(peaks):
        if peak - half_window < 0 or peak + half_window > num_samples:
            continue # límites de la ventana del ecg

        ml2_window = signal_filtered[0, peak - half_window : peak + half_window]
        v5_window = signal_filtered[1, peak - half_window : peak + half_window]

        window_beats.append({
            "peaks": peak,
            "MLII": ml2_window,
            "V5": v5_window
        })

    return window_beats

def normalize(signal_windowed: list[dict]) -> list[dict]:
    """normalize the values of the signal

    Args:
        signal_windowed (list[dict]): the signal that has been divided into window size

    Returns:
        list[dict]: the same signal list of dictionaries but values ranging from
    """
    epsilon = 1e-8
    normalized_beats = []
    for beat in signal_windowed:
        ml2_normalized = (beat["MLII"] - np.mean(beat["MLII"]) / np.std(beat["MLII"]) + epsilon)
        v5_normalized = (beat["V5"] - np.mean(beat["V5"]) / np.std(beat["V5"]) + epsilon)

        normalized_beats.append({
            "peaks": beat["peaks"],
            "MLII": ml2_normalized,
            "V5": v5_normalized
        })

    return normalized_beats
#==============================================================================================================================

#--------------------------------------------------- PIPELINE FUNCTION --------------------------------------------------------
def run_ecg_pipeline(patient_filepath: str, fs_target: int=500):
    #-------------- 1. Load the patient
    print(f"[1/10] Loading patient from {patient_filepath}...")
    record = wfdb.rdrecord(patient_filepath)
    p_signal = record.p_signal
    channel = record.sig_name
    print("Patient loaded correctly.")

    #-------------- 2. Resample thesignal if necessary
    if record.fs == 500:
        resampled_signal = record
        print(f"[2/10] Signal is at correct frequency of {record.fs}Hz")
    else:
        resampled_signal = resampling(signal=record, fs_signal=record.fs)
        print(f"[2/10] Signas has been resampled from {record.fs}Hz to {fs_target}Hz.")

    #-------------- 3. Reshape the signal
    reshaped_signal = reshape(resampled_signal)
    print(f"[3/10] The signal has been reshaped from {p_signal.shape} to {reshaped_signal.shape}")

    #-------------- 4. Apply the filters
    filtered_signal = apply_filters(signal_reshaped=reshaped_signal)
    print(f"[4/10] Filters Butterworth (0.5Hz - 40Hz) and Notch (50Hz) applied.")

    #-------------- 5. Detect the QRS peaks
    qrs_peaks = find_QRS_peak(signal_filtered=filtered_signal)
    print(f"[5/10] The QRS peaks has been detected.")

    #-------------- 6. Segment the signal with a window size
    windowed_signal = segmentation_window(signal_filtered=filtered_signal, peaks=qrs_peaks)
    print(f"[6/10] Window segmentation applied.")

    #-------------- 7. Normalization for all the values in the patients ecg
    normalized_signal = normalize(signal_windowed=windowed_signal)
    print(f"[7/10] Normalization of the ECG completed.")

test = run_ecg_pipeline(ptb_xl_dir+"/"+TEST_PATIENT)





















#==============================================================================================================================
'''
#------------------------------------------------------ TESTING ---------------------------------------------------------------
final_signal = reshape(patient_values)

print(f"The reshaped channel MLII: {final_signal[0]}")
print(f"The reshaped channel V5: {final_signal[1]}")

print(f"The reshaped channel MLII max: {final_signal[0].max()}")
print(f"The reshaped channel V5 max: {final_signal[1].max()}")

print(f"The reshaped channel MLII min: {final_signal[0].min()}")
print(f"The reshaped channel V5 min: {final_signal[1].min()}")

print(psignal_patient.shape)
print(final_signal.shape)

filtered = apply_filters(final_signal)
print(f"filtered signal: {filtered}")
print(f"filtered signal MLII max: {filtered[0].max()}")
print(f"filtered signal V5 max: {filtered[1].max()}")
print(f"filtered signal MLII min: {filtered[0].min()}")
print(f"filtered signal V5 min: {filtered[1].min()}")

peaks = find_QRS_peak(filtered)
windowed = segmentation_window(filtered, peaks)
print(f"The values of the windows: {windowed}")

normalized = normalize(windowed)
print(f"Values normalized max (MLII): {normalized[0]['MLII'].max():.4f}")
print(f"Values normalized min (MLII): {normalized[0]['MLII'].min():.4f}")


psignal_patient_reshaped = np.reshape(psignal_patient, )
print(p)

#==============================================================================================================================
'''