# data_processing.py

import os
import numpy as np
import glob
from sklearn.model_selection import train_test_split
import concurrent.futures

# --- Configuration Parameters ---
# Define the activity labels to be classified
labels = ('bed', 'fall', 'pickup', 'run', 'sitdown', 'standup', 'walk')

# Define sliding window parameters
win_len = 1000  # length of the window
thrshd = 0.6    # threshold for label majority
step = 500      # step size for the sliding window

# Define the directories to process
dirs_files = {
    "hampel_filter": "pd_ham_raw_lstm.npz",
    "../data_amp": "pd_nf_raw_lstm.npz",
    "butterworth_cutoff_10_order_5": "pd_bw_raw_lstm.npz",
    "dwt_denoised_wavelet_db8_level_3": "pd_dwt_raw_lstm.npz"
}

# --- Function Definitions ---

def read_merged_data(file_path):
    """Read a CSV file and return its contents as a numpy array."""
    return np.loadtxt(file_path, delimiter=",")

def apply_sliding_window(csi_array, label_array):
    """Apply sliding window approach to the given CSI and label array."""
    merged_samples = []
    # Slide window across the CSI array and collect samples
    for index in range(0, csi_array.shape[0] - win_len + 1, step):
        # Only consider windows with sufficient label majority
        if np.sum(label_array[index:index + win_len]) < thrshd * win_len:
            continue
        cur_sample = csi_array[index:index + win_len, :]
        merged_samples.append(cur_sample[np.newaxis, ...])

    # Stack samples together in one array
    sample_batch = np.concatenate(merged_samples, axis=0)
    return sample_batch, len(merged_samples)

def extract_samples_for_label(target_label):
    """Extract and return all samples for a given label."""
    # Locate all files matching the target label pattern
    merged_data_pattern = os.path.join(merged_folder, f'merged_*{target_label}*.csv')
    merged_files = sorted(glob.glob(merged_data_pattern))

    all_samples = []
    total_files = len(merged_files)
    files_processed = 0
    # Process each file
    for merged_file in merged_files:
        merged_data = read_merged_data(merged_file)
        csi_array, label_array = merged_data[:, :-1], merged_data[:, -1]

        sample_batch, num_samples = apply_sliding_window(csi_array, label_array)
        all_samples.append(sample_batch)
        files_processed += 1
        # Log progress
        print(f"Extracted {num_samples} samples from file for label '{target_label}'.")
        percent_label_done = (files_processed / total_files) * 100
        print(f"Progress for label '{target_label}': {percent_label_done:.2f}% complete.")

    # Combine all samples and create label arrays
    sample_array = np.concatenate(all_samples, axis=0)
    label_array = np.zeros((sample_array.shape[0], len(labels)))
    label_array[:, labels.index(target_label)] = 1
    return sample_array, label_array

def extract_all_samples():
    """Extract samples for all labels using multiprocessing."""
    print("Starting the sample extraction process...")
    total_samples_extracted = 0
    all_samples = []

    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Map each label to future objects
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
    """Split all samples into training and validation sets."""
    print("\nSplitting the data into training and validation sets...")
    # Combine all samples and labels
    x_all = np.concatenate([samples for samples, _ in all_samples], axis=0)
    y_all = np.concatenate([label_data for _, label_data in all_samples], axis=0)
    # Perform train-validation split
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

# --- Main Execution ---

if __name__ == "__main__":
    print("Starting the CSI classification data processing...")

    for dir, output_file_name in dirs_files.items():
        merged_folder = dir  # Folder with merged data

        # Extract samples for all labels
        all_samples = extract_all_samples()

        # Split data into training and validation sets
        #x_train, x_valid, y_train, y_valid = train_valid_split(all_samples)

        # Split data into training, validation, and test sets
        x_train, x_valid, x_test, y_train, y_valid, y_test = train_valid_test_split(all_samples)

        # # check to see if the output file already exists and if so add new in capital letters to the end of the file name
        # if os.path.exists(output_file_name):
        #     output_file_name = output_file_name[:-4] + "_NEW.npz"

        # Save processed data to files
        #np.savez(output_file_name,  x_train=x_train, x_valid=x_valid, y_train=y_train, y_valid=y_valid)
        np.savez(output_file_name,  x_train=x_train, x_valid=x_valid, x_test=x_test, y_train=y_train, y_valid=y_valid, y_test=y_test)

        print(f"\nProcessed data saved to '{output_file_name}'.")
