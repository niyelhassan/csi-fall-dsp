# data_processing.py

import os
import numpy as np
import glob
from sklearn.model_selection import train_test_split
import concurrent.futures
from scipy.stats import skew, kurtosis, iqr, median_abs_deviation as mad

# Configuration Parameters
labels = ('bed', 'fall', 'pickup', 'run', 'sitdown', 'standup', 'walk')
win_len = 1000
thrshd = 0.6
step = 500


# Define the directories to process
dirs_files = {
    'butterworth_cutoff_10_order_5': 'pd_bw_time_lstm.npz',
    "dwt_denoised_wavelet_db8_level_3": "pd_dwt_time_lstm.npz",
     "hampel_filter": "pd_ham_time_lstm.npz",
     "../data_amp": "pd_nf_time_lstm.npz"
}

def read_merged_data(file_path):
    # Read the merged CSV file and return its content
    return np.loadtxt(file_path, delimiter=",")

def extract_features(csi_window):
    # Extract time domain features from a window of data
    mean = np.mean(csi_window, axis=0)
    std_dev = np.std(csi_window, axis=0)
    max_val = np.max(csi_window, axis=0)
    min_val = np.min(csi_window, axis=0)
    variance = np.var(csi_window, axis=0)
    median = np.median(csi_window, axis=0)
    mad_val = mad(csi_window, axis=0)
    iqr_val = iqr(csi_window, axis=0)
    velocity_change = np.mean(np.diff(csi_window, axis=0), axis=0)
    peak_to_peak = max_val - min_val
    skewness = skew(csi_window, axis=0)
    kurt = kurtosis(csi_window, axis=0)
    return np.concatenate([mean, std_dev, max_val, min_val, variance, median, mad_val, iqr_val, velocity_change, peak_to_peak, skewness, kurt])

def apply_sliding_window(csi_array, label_array):
    merged_samples = []
    for index in range(0, csi_array.shape[0] - win_len + 1, step):
        if np.sum(label_array[index:index + win_len]) < thrshd * win_len:
            continue
        cur_sample = csi_array[index:index + win_len, :]
        cur_sample = extract_features(cur_sample)
        merged_samples.append(cur_sample[np.newaxis, ...])

    sample_batch = np.concatenate(merged_samples, axis=0)
    return sample_batch, len(merged_samples)

def extract_samples_for_label(target_label):
    merged_data_pattern = os.path.join(merged_folder, f'merged_*{target_label}*.csv')
    merged_files = sorted(glob.glob(merged_data_pattern))

    all_samples = []
    total_files = len(merged_files)
    files_processed = 0
    for merged_file in merged_files:
        merged_data = read_merged_data(merged_file)
        csi_array, label_array = merged_data[:, :-1], merged_data[:, -1]

        sample_batch, num_samples = apply_sliding_window(csi_array, label_array)
        all_samples.append(sample_batch)
        files_processed += 1
        print(f"Extracted {num_samples} samples from file for label '{target_label}'.")
        percent_label_done = (files_processed / total_files) * 100
        print(f"Progress for label '{target_label}': {percent_label_done:.2f}% complete.")

    sample_array = np.concatenate(all_samples, axis=0)
    label_array = np.zeros((sample_array.shape[0], len(labels)))
    label_array[:, labels.index(target_label)] = 1
    return sample_array, label_array

def extract_all_samples():
    print("Starting the sample extraction process...")
    total_samples_extracted = 0
    all_samples = []
    total_labels = len(labels)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        future_to_label = {executor.submit(extract_samples_for_label, label): label for label in labels}
        for future in concurrent.futures.as_completed(future_to_label):
            label = future_to_label[future]
            try:
                samples, label_data = future.result()
            except Exception as exc:
                print(f'{label} generated an exception: {exc}')
            else:
                num_samples = samples.shape[0]
                total_samples_extracted += num_samples
                all_samples.append((samples, label_data))
                print(f"Done. Extracted {num_samples} samples for '{label}'.")
    
    print(f"\nTotal samples extracted: {total_samples_extracted}")
    return all_samples

def train_valid_split(all_samples):
    print("\nSplitting the data into training and validation sets...")
    x_all = np.concatenate([samples for samples, _ in all_samples], axis=0)
    y_all = np.concatenate([label_data for _, label_data in all_samples], axis=0)
    x_train, x_valid, y_train, y_valid = train_test_split(x_all, y_all, train_size=0.9, random_state=379, stratify=y_all)
    print(f"Split complete. Training set size: {x_train.shape[0]}, Validation set size: {x_valid.shape[0]}")
    return x_train, x_valid, y_train, y_valid


def train_valid_test_split(all_samples):
    print("\nSplitting the data into training, validation, and test sets...")
    x_all = np.concatenate([samples for samples, _ in all_samples], axis=0)
    y_all = np.concatenate([label_data for _, label_data in all_samples], axis=0)
    
    # First split the data into training (85%) and temporary (15%) sets
    x_train, x_temp, y_train, y_temp = train_test_split(x_all, y_all, test_size=0.15, random_state=379, stratify=y_all)
    
    # Then split the temporary set into validation (5%) and test (10%) sets
    x_valid, x_test, y_valid, y_test = train_test_split(x_temp, y_temp, test_size=2/3, random_state=379, stratify=y_temp)
    
    print(f"Split complete. Training set size: {x_train.shape[0]}, Validation set size: {x_valid.shape[0]}, Test set size: {x_test.shape[0]}")
    return x_train, x_valid, x_test, y_train, y_valid, y_test

# Main execution
print("Starting the CSI classification data processing...")

for dir, output_file_name in dirs_files.items():
    merged_folder = dir  # Folder with merged data

    # Extract samples for all labels
    all_samples = extract_all_samples()

    # Split data into training and validation sets
    #x_train, x_valid, y_train, y_valid = train_valid_split(all_samples)

    # Split data into training, validation, and test sets
    x_train, x_valid, x_test, y_train, y_valid, y_test = train_valid_test_split(all_samples)

    # check to see if the output file already exists and if so add new in capital letters to the end of the file name
    # if os.path.exists(output_file_name):
    #     output_file_name = output_file_name[:-4] + "_NEW.npz"

    # Save processed data to files
    #np.savez(output_file_name,  x_train=x_train, x_valid=x_valid, y_train=y_train, y_valid=y_valid)
    np.savez(output_file_name,  x_train=x_train, x_valid=x_valid, x_test=x_test, y_train=y_train, y_valid=y_valid, y_test=y_test)
    print(f"\nProcessed data saved to '{output_file_name}'.")