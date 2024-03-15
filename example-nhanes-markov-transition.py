import pandas as pd
import numpy as np
from pyts.image import MarkovTransitionField
import time

# Load the time series data
df = pd.read_csv('data/binned_sl_AC_sample.csv')

# Parameters
n_series = len(df)
image_size = 240

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
    
# End timing
end_time = time.time()

# Calculate duration
duration = end_time - start_time

# Print duration
print(f"Conversion completed in {duration} seconds.")

# Now, `mtf_images` is your 3D array containing all MTF images

# Save the array to an HDF5 file for R compatibility
np.save('data/mtf_images.npy', mtf_images)

print("MTF images saved to 'mtf_images.npy'.")
