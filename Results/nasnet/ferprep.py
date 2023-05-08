import numpy as np
from PIL import Image
import pandas as pd
import os

# Load the CSV file
df = pd.read_csv('/root/DataPrep/new/fer2013.csv')

# Convert the pixel values to images and save to disk
image_dir = 'imagesFer'
if not os.path.exists(image_dir):
    os.mkdir(image_dir)
for i in range(len(df)):
    pixels = df['pixels'][i]
    pixels = np.array(pixels.split(), dtype='uint8')
    img = Image.fromarray(pixels.reshape((48, 48)))
    filename = os.path.join(image_dir, f"{i}.png")
    img.save(filename)
    df.at[i, 'pixels'] = filename

# Save the updated CSV file
df.to_csv('fer2013_updated.csv', index=False)
