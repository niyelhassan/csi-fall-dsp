import os
import numpy as np
import time
from scipy.signal import butter, lfilter
from multiprocessing import Pool

# Butterworth low pass filter parameters
cutoffs = [10]  # Desired cutoff frequency of the filter, Hz
nyq = 0.5 * 1000  # Nyquist Frequency
orders = [5]  # Order of the filter

# Source directory
dir = "../data_amp"

def apply_butterworth_filter(data, b, a):
    # Apply the filter to all columns of the array
    for i in range(data.shape[1]):
        data[:, i] = lfilter(b, a, data[:, i])
    return data

def process_file(filename):
    start_time = time.time()
    # Apply the Butterworth low pass filter with different parameters
    for cutoff in cutoffs:
        for order in orders:
            # Create the destination directory if it doesn't exist
            filtered_dir = os.path.join(f'butterworth_cutoff_{cutoff}_order_{order}')
            os.makedirs(filtered_dir, exist_ok=True)

            # Check if the filtered file already exists
            filtered_file_path = os.path.join(filtered_dir, filename)
            if os.path.exists(filtered_file_path):
                print(f'Skipping file: {filename} with cutoff: {cutoff} and order: {order} as filtered file already exists.')
                continue

            print(f'Processing file: {filename} with cutoff: {cutoff} and order: {order}')
            
            # Load the data
            data = np.genfromtxt(os.path.join(dir, filename), delimiter=',', skip_header=1)

            # Apply the Butterworth low pass filter to the amplitude data
            normal_cutoff = cutoff / nyq
            b, a = butter(order, normal_cutoff, btype='low', analog=False)
            data[:, 0:89] = apply_butterworth_filter(data[:, 1:90], b, a)

            # Save the original data to a new CSV file
            np.savetxt(filtered_file_path, data, delimiter=',', fmt='%g')

    end_time = time.time()
    print(f'Processing file: {filename} took {end_time - start_time} seconds.')
    return end_time - start_time

# Get a list of all CSV files in the source directory
csv_files = [f for f in os.listdir(dir) if f.endswith('.csv')]

# Create a multiprocessing Pool
with Pool() as p:
    # Use the Pool's map function to apply the filter function to all CSV files
    start_time = time.time()
    processing_times = p.map(process_file, csv_files)
    end_time = time.time()

print(f'Processing all files took {end_time - start_time} seconds.')
print(f'Average time per file: {np.mean(processing_times)} seconds.')