import os
import numpy as np
import time
from multiprocessing import Pool

# Hampel filter parameters
n = 3  # Number of deviations from the median

# Source directory
dir = "../data_amp"

def apply_hampel_filter(data, n):
    # Apply the filter to all columns of the array
    for i in range(data.shape[1]):
        median = np.median(data[:, i])
        mad = np.median(np.abs(data[:, i] - median))
        data[:, i] = np.where(np.abs(data[:, i] - median) > n * mad, median, data[:, i])
    return data

def process_file(filename):
    start_time = time.time()
    # Create the destination directory if it doesn't exist
    filtered_dir = os.path.join('hampel_filter')
    os.makedirs(filtered_dir, exist_ok=True)

    # Check if the filtered file already exists
    filtered_file_path = os.path.join(filtered_dir, filename)
    if os.path.exists(filtered_file_path):
        print(f'Skipping file: {filename} as filtered file already exists.')
        return

    print(f'Processing file: {filename}')
    
    # Load the data
    data = np.genfromtxt(os.path.join(dir, filename), delimiter=',', skip_header=1)

    # Apply the Hampel filter to the amplitude data
    data[:, 0:89] = apply_hampel_filter(data[:, 0:89], n)

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