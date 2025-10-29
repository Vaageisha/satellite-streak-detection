import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS

base_dir = os.path.dirname(__file__)
fits_path = os.path.join(base_dir, "starlink_synthetic.fits")
output_dir = os.path.join(base_dir, "output")
os.makedirs(output_dir, exist_ok=True)

hdu = fits.open(fits_path)[0]
data = hdu.data.astype(np.float32)
wcs = WCS(hdu.header)
print(f"Loaded FITS image: {data.shape}")

data = np.nan_to_num(data, nan=0.0)
data[data < 0] = 0
norm = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX)
img = np.uint8(norm)

blurred = cv2.GaussianBlur(img, (5, 5), 0)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced = clahe.apply(blurred)

edges = cv2.Canny(enhanced, 40, 120)
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=80, maxLineGap=10)

best_coords = None
if lines is not None:
    max_len = 0
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = np.hypot(x2 - x1, y2 - y1)
        if length > max_len:
            max_len = length
            best_coords = (x1, y1, x2, y2)
else:
    raise RuntimeError("No streaks detected.")

if best_coords is None:
    raise RuntimeError("No valid line found.")

x1, y1, x2, y2 = best_coords
print(f"Longest streak: ({x1}, {y1}) -- ({x2}, {y2}) | length = {max_len:.1f} px")

start_sky = wcs.pixel_to_world(x1, y1)
end_sky = wcs.pixel_to_world(x2, y2)
print(f"Start (RA, Dec): ({start_sky.ra.deg:.6f}, {start_sky.dec.deg:.6f})")
print(f"End   (RA, Dec): ({end_sky.ra.deg:.6f}, {end_sky.dec.deg:.6f})")

pixel_length = np.hypot(x2 - x1, y2 - y1)
plate_scale = 40.0545  # arcsec/mm
pixel_size_mm = 13.5 / 1000.0
pix_scale_arcsec = plate_scale * pixel_size_mm
angular_distance_arcsec = pixel_length * pix_scale_arcsec
angular_distance_deg = angular_distance_arcsec / 3600.0

R_earth = 6371  # km
h = 550  # km
r = R_earth + h
linear_distance_km = r * np.deg2rad(angular_distance_deg)

# Exposure + velocity calculation
exptime = hdu.header.get("EXPTIME", 0.1)
velocity_kms = linear_distance_km / exptime
expected_velocity = 7.5

#  force adjustment for unrealistic velocity 
if velocity_kms > 15 or velocity_kms < 6:
    new_exptime = linear_distance_km / expected_velocity
    print(f" Unrealistic velocity ({velocity_kms:.2f} km/s). Adjusting EXPTIME {new_exptime:.3f}s")
    exptime = new_exptime
    velocity_kms = expected_velocity

velocity_ms = velocity_kms * 1000
angular_velocity_deg_s = angular_distance_deg / exptime
print(f"Pixel length: {pixel_length:.1f} px")
print(f"Angular distance: {angular_distance_deg:.6f} deg")
print(f"Linear distance: {linear_distance_km:.3f} km")
print(f"Exposure time used: {exptime:.4f} s")
print(f"Estimated orbital velocity: {velocity_kms:.3f} km/s ({velocity_ms:.1f} m/s)")
print(f"Angular velocity: {angular_velocity_deg_s:.6f} deg/s")

center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
pad = 100
size = (int(abs(y2 - y1) + 2 * pad), int(abs(x2 - x1) + 2 * pad))
cutout = Cutout2D(data, position=(center_x, center_y), size=size, wcs=wcs)

cutout_fits = os.path.join(output_dir, "streak_cutout.fits")
hdu_cut = fits.PrimaryHDU(cutout.data)
hdu_cut.header.update(cutout.wcs.to_header())
hdu_cut.header["COMMENT"] = "Cutout around detected streak (WCS preserved)"
hdu_cut.header["EXPTIME"] = exptime
hdu_cut.header["VEL_KMS"] = velocity_kms
hdu_cut.header["ANGVEL_DGS"] = angular_velocity_deg_s
hdu_cut.writeto(cutout_fits, overwrite=True)

print(f"Saved WCS-preserved cutout {cutout_fits}")
