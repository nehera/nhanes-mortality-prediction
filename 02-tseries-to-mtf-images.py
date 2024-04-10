import pandas as pd
import numpy as np
from pyts.image import MarkovTransitionField
import time

# Load the time series data
df = pd.read_csv('data/act_bin_sl_D.csv')
df.drop('SEQN', inplace=True, axis=1) # Remove unique identifiers for image processing

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
    if i % 200 == 0:
        print(f"Processing at index: {i}")

# End timing
end_time = time.time()

# Calculate duration
duration = end_time - start_time

# Print duration
print(f"Conversion completed in {duration} seconds.")

# Now, `mtf_images` is your 3D array containing all MTF images

# Save the array to a numpy file, which can be imported into R as an array using the reticulate package
np.save('data/mtf_images_D.npy', mtf_images)

print("MTF images saved to 'mtf_images.npy'.")
