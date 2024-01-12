import os
import numpy as np
import time
import pywt
from multiprocessing import Pool

# Source directory
dir = "../data_amp"

def apply_dwt_denoising(data, wavelet, level):
    # Apply the DWT denoising to all columns of the array
    for i in range(data.shape[1]):
        # Decompose to get the wavelet coefficients
        coeff = pywt.wavedec(data[:, i], wavelet, level=level)
        
        # Set the detail coefficients to zero
        for i in range(1, len(coeff)):
            coeff[i] = pywt.threshold(coeff[i], np.std(coeff[i])/2)
        
        # Reconstruct the signal
        reconstructed = pywt.waverec(coeff, wavelet)
        data[:, i] = reconstructed[:data.shape[0]]
    return data

def process_file(filename):
    start_time = time.time()
    
    # Apply the DWT denoising
    wavelet = 'db8'
    level = 3

    # Create the destination directory if it doesn't exist
    denoised_dir = os.path.join(f'dwt_denoised_wavelet_{wavelet}_level_{level}')
    os.makedirs(denoised_dir, exist_ok=True)

    # Check if the denoised file already exists
    denoised_file_path = os.path.join(denoised_dir, filename)
    if os.path.exists(denoised_file_path):
        print(f'Skipping file: {filename} as denoised file already exists.')
        return

    print(f'Processing file: {filename}')

    # Load the data
    data = np.genfromtxt(os.path.join(dir, filename), delimiter=',', skip_header=1)

    # Apply the DWT denoising to the amplitude data
    data[:, 0:89] = apply_dwt_denoising(data[:, 0:89], wavelet, level)

    # Save the original data to a new CSV file
    np.savetxt(denoised_file_path, data, delimiter=',', fmt='%g')

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