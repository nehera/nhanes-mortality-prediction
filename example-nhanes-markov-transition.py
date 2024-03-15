# Author: Johann Faouzi <johann.faouzi@gmail.com>
# License: BSD-3-Clause

import numpy as np
import matplotlib.pyplot as plt
from pyts.image import MarkovTransitionField
import pandas as pd


# Import a toy time series
# time_points = np.linspace(0, 4 * np.pi, 1000)
# x = np.sin(time_points)

# Load the CSV file
df = pd.read_csv('sub1.csv')

# Convert the 'minute' column to a numerical type, assuming float
time_points = df['minute'].astype(float)

# Convert the 'activity' column to a numerical type, assuming float
x = df['activity'].astype(float)

# Now, 't' and 'x' are pandas Series objects with float data types
# We convert 'x' to a np.array
X = np.array([x])

# Compute Gramian angular fields
mtf = MarkovTransitionField(n_bins=8)
X_mtf = mtf.fit_transform(X)

# # Plot the time series and its Markov transition field
# width_ratios = (2, 7, 0.4)
# height_ratios = (2, 7)
# width = 6
# height = width * sum(height_ratios) / sum(width_ratios)
# fig = plt.figure(figsize=(width, height))
# gs = fig.add_gridspec(2, 3,  width_ratios=width_ratios,
#                       height_ratios=height_ratios,
#                       left=0.1, right=0.9, bottom=0.1, top=0.9,
#                       wspace=0.05, hspace=0.05)

# # Define the ticks and their labels for both axes
# time_ticks = time_points

# # Plot the time series on the left with inverted axes
# ax_left = fig.add_subplot(gs[1, 0])
# ax_left.plot(x, time_points)
# ax_left.set_yticks(time_ticks)

# # Plot the Gramian angular fields on the bottom right
# ax_mtf = fig.add_subplot(gs[1, 1])
# im = ax_mtf.imshow(X_mtf[0], cmap='rainbow', origin='lower', vmin=0., vmax=1.,
#                    extent=[0, np.max(time_points), 0, np.max(time_points)])
# ax_mtf.set_xticks([])
# ax_mtf.set_yticks([])
# ax_mtf.set_title('Markov Transition Field', y=-0.09)

# # Add colorbar
# ax_cbar = fig.add_subplot(gs[1, 2])
# fig.colorbar(im, cax=ax_cbar)

# plt.show()