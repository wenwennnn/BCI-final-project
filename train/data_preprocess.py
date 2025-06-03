import scipy.io
import numpy as np
import mne
from autoreject import AutoReject

# === 1. 載入資料與通道資訊 ===
mat = scipy.io.loadmat("Dreamer/DREAMER.mat")
dreamer = mat['DREAMER'][0, 0]
sfreq = dreamer['EEG_SamplingRate'][0, 0]
electrode_names = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1',
                   'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
info = mne.create_info(ch_names=electrode_names, sfreq=sfreq, ch_types="eeg")

# === 2. 頻帶定義 ===
bands = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 40)
}

X, Y = [], []

# === 3. 擷取每位受試者的每段 trial 特徵 ===
for subj in range(dreamer['Data'].shape[1]):
    print(f"Extract Subj: {subj} Feature...")
    subject_tuple = dreamer['Data'][0, subj][0, 0]
    eeg_struct = subject_tuple[2]
    arousal = subject_tuple[5].flatten()
    valence = subject_tuple[6].flatten()
    stimuli = eeg_struct['stimuli'][0, 0]

    for trial in range(18):
        print(f"Current Trial : {trial}")
        eeg = stimuli[trial, 0].T  # shape: (14, 8064)
        try:
            raw = mne.io.RawArray(eeg, info, verbose=False)
            raw.set_montage("standard_1020")
            epochs = mne.make_fixed_length_epochs(raw, duration=4.0, preload=True, verbose=False)
            ar = AutoReject()
            cleaned = ar.fit_transform(epochs)
            psd = cleaned.compute_psd(fmin=1, fmax=40)
            psd_avg = psd.get_data().mean(axis=0)  # shape: (14, freq_bins)

            # 計算每個頻帶的 power
            band_power = []
            for ch_data in psd_avg:
                ch_band = []
                for fmin, fmax in bands.values():
                    idx = np.where((psd.freqs >= fmin) & (psd.freqs < fmax))[0]
                    ch_band.append(np.mean(ch_data[idx]))
                band_power.append(ch_band)  # shape: (14, 5)
            X.append(np.array(band_power))
            Y.append([int(arousal[trial]) - 1, int(valence[trial]) - 1])  # 類別 0~4

        except Exception as e:
            print(f"Skip subj {subj} trial {trial}: {e}")
            continue

# === 4. 儲存為 npz ===
X = np.array(X)  # shape: (N, 14, 5)
Y = np.array(Y)  # shape: (N, 2)
np.savez_compressed("dreamer_psd_features.npz", X=X, Y=Y)
print("✅ PSD features saved to 'dreamer_psd_features.npz'")
