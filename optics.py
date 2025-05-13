import numpy as np
from astropy.io import fits
import os
from scipy.special import erf
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Define the Gaussian error function (from lab document Tip 3)
def gauss_erf(x, a, b, c, d):
    return a * erf(b * (x - c)) + d

# Define Gaussian for fitting sharpness vs. position
def gaussian(x, a, mu, sigma, offset):
    return a * np.exp(-((x - mu)**2) / (2 * sigma**2)) + offset

# Directory containing FITS files (adjust as needed)
directory = './'

# Green filter FITS files
green_files = [
    'green_10.fits',
    'green-2.fits', 'green-3.fits', 'green-4.fits', 'green-5.fits', 'green-6.fits', 'green-7.fits', 'green-8.fits', 'green-9.fits',
    'green-2_25.fits', 'green-3_25.fits', 'green-4_25.fits', 'green-5_25.fits', 'green-6_25.fits', 'green-7_25.fits', 'green-8_25.fits', 'green-9_25.fits',
    'green-2_5.fits', 'green-3_5.fits', 'green-4_5.fits', 'green-5_5.fits', 'green-6_5.fits', 'green-7_5.fits', 'green-8_5.fits', 'green-9_5.fits',
    'green-2_75.fits', 'green-3_75.fits', 'green-4_75.fits', 'green-5_75.fits', 'green-6_75.fits', 'green-7_75.fits', 'green-8_75.fits', 'green-9_75.fits'
]

# Red filter FITS files
red_files = [
    'red-3_25.fits', 'red-3_75.fits', 'red-4_25.fits', 'red-4_75.fits', 'red-5_25.fits', 'red-5_75.fits', 'red-6_25.fits', 'red-6_75.fits',
    'red-3_5.fits', 'red-3.fits', 'red-4_5.fits', 'red-4.fits', 'red-5_5.fits', 'red-5.fits', 'red-6_5.fits', 'red-6.fits',
    'red-10_25.fits', 'red-10_75.fits', 'red-11.fits', 'red-7_25.fits', 'red-7_75.fits', 'red-8_5.fits', 'red-8.fits', 'red-9_5.fits', 'red-9.fits',
    'red-10_5.fits', 'red-10.fits', 'red-7_00-centre.fits', 'red-7_5.fits', 'red-8_25.fits', 'red-8_75.fits', 'red-9_25.fits', 'red-9_75.fits'
]

# Define sub-region coordinates for each filter
coordinates = {
    'green': {'x_start': 500, 'y_start': 650},
    'red': {'x_start': 700, 'y_start': 650}  # Placeholder: Update with correct values
}

# Function to parse stage positions from file names
def parse_stage_position(fname, filter_type):
    num = fname.replace(f'{filter_type}_', '').replace(f'{filter_type}-', '').replace('.fits', '')
    num = num.replace('_00-centre', '')  # Handle 'red-7_00-centre.fits'
    num = num.replace('_', '.')  # Replace '_' with '.' for decimal positions
    try:
        return float(num)
    except ValueError:
        print(f"Could not parse stage position from {fname}, skipping")
        return None

# Data structures
datasets = {
    'green': {'files': green_files, 'sharpness': [], 'xy_data': {}, 'color': 'g'},
    'red': {'files': red_files, 'sharpness': [], 'xy_data': {}, 'color': 'r'}
}

# Process each dataset (green and red)
for filter_type, data in datasets.items():
    for fname in data['files']:
        position = parse_stage_position(fname, filter_type)
        if position is None:
            continue
        file_path = os.path.join(directory, fname)
        try:
            with fits.open(file_path) as hdul:
                image = hdul[0].data  # Assume data is in primary HDU

                # Select a sub-region using filter-specific coordinates
                x_start = coordinates[filter_type]['x_start']
                y_start = coordinates[filter_type]['y_start']
                sub_region = image[y_start:y_start+150, x_start:x_start+10]  # Shape: (150, 10)

                # Compute median pixel values across rows for each column (10 values)
                median_counts = np.median(sub_region, axis=1)  # Shape: (10,)

                # Store X and Y
                X = np.arange(len(median_counts))  # Column indices
                Y = median_counts
                data['xy_data'][fname] = {'X': X, 'Y': Y}

                # Fit Gaussian error function
                p0 = [np.max(Y) - np.min(Y), 0.1, np.argmax(np.gradient(Y)), np.min(Y)]
                try:
                    popt, _ = curve_fit(gauss_erf, X, Y, p0=p0)
                    b_par = popt[1]  # Sharpness
                    data['sharpness'].append((position, b_par))
                    # print(f"Processed {fname}: Sharpness (b) = {b_par:.4f}")

                    # # Plot X, Y, and fitted curve
                    # plt.figure(figsize=(6, 4))
                    # plt.scatter(X, Y, color=data['color'], label='Data', s=20)
                    # X_fit = np.linspace(min(X), max(X), 100)
                    # Y_fit = gauss_erf(X_fit, *popt)
                    # plt.plot(X_fit, Y_fit, 'k-', label='Gaussian Error Function Fit')
                    # plt.xlabel('Column Index')
                    # plt.ylabel('Median Counts')
                    # plt.title(f'Edge Profile: {fname} (Sharpness b = {b_par:.4f})')
                    # plt.legend()
                    # plt.tight_layout()
                    # plt.savefig(f'plots/edge_profile_{fname}.png', dpi=300, bbox_inches='tight')
                    # plt.show()

                except RuntimeError:
                    print(f"Fit failed for {fname}")
                    data['sharpness'].append((position, None))
                    data['xy_data'][fname]['fit_failed'] = True

        except FileNotFoundError:
            print(f"File {fname} not found")
            data['sharpness'].append((position, None))
            data['xy_data'][fname] = {'error': 'File not found'}
        except Exception as e:
            print(f"Error processing {fname}: {e}")
            data['sharpness'].append((position, None))
            data['xy_data'][fname] = {'error': str(e)}

# Plot sharpness vs. stage position for both filters
plt.figure(figsize=(10, 6))
best_focus = {}

for filter_type, data in datasets.items():
    # Filter and sort sharpness values
    sharpness_values = [(pos, b) for pos, b in data['sharpness'] if b is not None]
    sharpness_values.sort()
    if not sharpness_values:
        print(f"No valid sharpness data for {filter_type} filter")
        continue
    positions, sharpness = zip(*sharpness_values)

    # Plot sharpness
    plt.scatter(positions, np.abs(sharpness), color=data['color'], label=f'{filter_type.capitalize()} Filter Sharpness')

    # Fit Gaussian
    try:
        p0 = [0.2, positions[np.argmax(sharpness)], 1, 3]
        popt, _ = curve_fit(gaussian, positions, sharpness, p0=p0)
        x_fit = np.linspace(min(positions), max(positions), 100)
        y_fit = gaussian(x_fit, *popt)
        plt.plot(x_fit, y_fit, f'{data["color"]}-', label=f'{filter_type.capitalize()} Gaussian Fit (Best Focus = {popt[1]:.2f} mm)')
        best_focus[filter_type] = popt[1]
        print(f"Best-focus position ({filter_type} filter): {popt[1]:.2f} mm")
    except RuntimeError:
        print(f"Gaussian fit failed for {filter_type} filter")

plt.xlabel('Relative Position (mm)')
plt.ylabel('Sharpness (b)')
#plt.title('Sharpness vs. Relati Position')
plt.legend()
plt.tight_layout()
plt.savefig('plots/sharpness_plot.png', dpi=300, bbox_inches='tight')
plt.show()

# Task d: Compare best-focus positions
if 'green' in best_focus and 'red' in best_focus:
    print("\nTask d: Comparison of Best-Focus Positions")
    print(f"Green filter best-focus position: {best_focus['green']:.2f} mm")
    print(f"Red filter best-focus position: {best_focus['red']:.2f} mm")
    difference = abs(best_focus['green'] - best_focus['red'])
    print(f"Difference in focus positions: {difference:.2f} mm")
    if best_focus['red'] > best_focus['green']:
        print("Red filter focuses farther from the lens than green filter.")
    else:
        print("Green filter focuses farther from the lens than red filter.")
    print("This makes sense because red light (longer wavelength) experiences less refraction than green light, so its focal point is farther from the lens due to chromatic aberration.")
else:
    print("\nTask d: Could not compare focus positions due to missing best-focus data.")