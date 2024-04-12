import pandas as pd
import numpy as np
import time
from pyts.image import MarkovTransitionField

def generate_mtf_images(data_path, output_file, image_size=240):
    """
    This function reads time series data from a CSV file, processes it into MTF images, and saves the output to a .npy file.
    
    Parameters:
        data_path (str): Path to the input CSV file containing time series data.
        output_file (str): Path where the MTF images numpy array will be saved.
        image_size (int): Size of the square MTF images (default is 240).
    """
    # Load the time series data
    df = pd.read_csv(data_path)
    df.drop('SEQN', inplace=True, axis=1) # Remove unique identifiers for image processing
    
    # Parameters
    n_series = len(df)
    
    # Initialize MTF transformer
    mtf = MarkovTransitionField(image_size=image_size)
    
    # Preallocate a 3D NumPy array: [n_series, image_size, image_size]
    mtf_images = np.zeros((n_series, image_size, image_size))
    
    # Start timing
    start_time = time.time()
    
    # Convert each time series and store in the array
    for i, (index, row) in enumerate(df.iterrows()):
        time_series = row.values.reshape(1, -1)
        mtf_image = mtf.fit_transform(time_series)
        mtf_images[i, :, :] = mtf_image
        if i % 200 == 0:
            print(f"Processing at index: {i}")
    
    # End timing
    end_time = time.time()
    
    # Calculate duration
    duration = end_time - start_time
    
    # Print duration
    print(f"Conversion completed in {duration} seconds.")
    
    # Save the array to a numpy file
    np.save(output_file, mtf_images)
    print(f"MTF images saved to '{output_file}'.")

# Example usage:
generate_mtf_images('data/AC_train.csv', 'data/mtf_images_train.npy')
generate_mtf_images('data/AC_valid.csv', 'data/mtf_images_valid.npy')
generate_mtf_images('data/AC_test.csv', 'data/mtf_images_test.npy')
