from flask import Flask, request, render_template
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import butter, filtfilt, find_peaks

import pandas as pd
# Define the allowed file extensions
ALLOWED_EXTENSIONS = {'txt', 'csv'}

def allowed_file(filename):
    """
    Check if a file has an allowed extension
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


app = Flask("Smart_ECG")

app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['TEMPLATE_FOLDER'] = 'templates'

def detect_diseases_from_ecg_file(filename):
    with open(filename, 'r') as f:
        ecg_data =  pd.read_csv(f, encoding='utf-8-sig')

    fs = 250  # Sampling frequency
    lowcut = 5  # Lower cutoff frequency
    highcut = 15  # Upper cutoff frequency
    order = 4  # Filter order
    padlen = 27  # Padding length

    if len(ecg_data) < 27:
        ecg_data_padded = np.pad(ecg_data, (0, 27 - len(ecg_data)), 'constant')
    else:
        ecg_data_padded = ecg_data

    # Pad the input signal
    ecg_data_padded = np.pad(ecg_data, (padlen // 2, padlen // 2), 'constant')

    # Design the filter
    nyquist_freq = 0.5 * fs
    low = lowcut / nyquist_freq
    high = highcut / nyquist_freq
    b, a = butter(order, [low, high], btype="band")

    # Apply the filter
    filtered_data = filtfilt(b, a, ecg_data_padded)
    filtered_data_flat = filtered_data.flatten()

    threshold = np.mean(filtered_data_flat)
    rpeaks, _ = find_peaks(filtered_data_flat, height=threshold)
    rr_intervals = np.diff(rpeaks) / fs

    mean_rr = np.mean(rr_intervals)
    sdnn = np.std(rr_intervals)
    rmssd = np.sqrt(np.mean(np.diff(rr_intervals) ** 2))
    pnn50 = np.sum(np.abs(np.diff(rr_intervals)) > 50) / len(rr_intervals)

    diseases = []

    if (mean_rr < 600) or (mean_rr > 1200) or (sdnn < 50) or (sdnn > 100):
        diseases.append("Arrhythmia")
    if rmssd < 0.1:
        diseases.append("Ventricular fibrillation")
    afib_threshold = 0.03
    if rmssd < afib_threshold:
        diseases.append("Atrial fibrillation")
    r_peaks, _ = signal.find_peaks(filtered_data_flat, distance=100)
    rr_intervals = np.diff(r_peaks) / fs
    heart_rate = 60 / np.mean(rr_intervals)
    if heart_rate < 60:
        diseases.append("Bradycardia arrhythmia")
    if heart_rate > 100:
        diseases.append("Tachycardia arrhythmia")
    if 60 <= heart_rate <= 100:
        diseases.append("Normal sinus rhythm")
    # Calculate the mean RR interval
    mean_rr_interval = np.mean(rr_intervals)
    # Define a threshold for hypocalcaemia
    threshold = 1.5 * mean_rr_interval
    if any(rr_intervals > threshold):
        diseases.append("Hypocalcaemia")
    tall_t_waves = (np.max(filtered_data ) > 0.5) and (np.min(filtered_data ) < -0.5)
    wide_qrs_complex = (len(np.where(filtered_data  < -0.1)[0]) > 3)
    narrow_p_wave = (len(np.where(filtered_data  > 0.1)[0]) < 2)
    if tall_t_waves and wide_qrs_complex and narrow_p_wave:
         diseases.append("Hyperkalemia")
    r_peaks, _ = find_peaks(filtered_data_flat, height=0.5*np.max(filtered_data), distance=100)
    # Find the PR intervals between consecutive R-peaks
    pr_intervals = np.diff(r_peaks) / fs * 1000
    # Determine if second degree heart block is present
    if np.any(pr_intervals > 200):
        diseases.append("Second degree heart block")
    return ", ".join(diseases)

@app.route('/', methods=['GET', 'POST'])
def index():
 
    if request.method == 'POST':
        file = request.files['file']
        filename = file.filename
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        result = detect_diseases_from_ecg_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return render_template('result.html', result=result)
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.config['UPLOAD_FOLDER'] = 'uploads'
    app.run(debug=False)
