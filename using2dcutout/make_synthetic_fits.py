import numpy as np
from astropy.io import fits
from astropy import units as u
from astropy.wcs import WCS
from astropy.coordinates import EarthLocation
from astropy.time import Time
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import os

out_dir = os.path.dirname(__file__)
fits_out = os.path.join(out_dir, "starlink_synthetic.fits")
png_out = os.path.join(out_dir, "starlink_synthetic_preview.png")

plate_scale = 40.0545  # arcsec/mm
pixel_size_um = 13.5   # micron
pixel_size_mm = pixel_size_um / 1000.0
pix_scale_arcsec = plate_scale * pixel_size_mm
pix_scale_deg = pix_scale_arcsec / 3600.0

location = EarthLocation(
    lon=79.685 * u.deg,
    lat=29.361 * u.deg,
    height=2450 * u.m
)
obs_time = Time("2025-03-11T02:58:24.294708", scale="utc")

ra_cen = 165.09
dec_cen = 53.056
size = 2048
rotation_deg = 0.7

theta = np.deg2rad(rotation_deg)
scale = pix_scale_deg

w = WCS(naxis=2)
w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
w.wcs.crval = [ra_cen, dec_cen]
w.wcs.crpix = [size / 2, size / 2]
w.wcs.cd = [
    [-scale * np.cos(theta), scale * np.sin(theta)],
    [scale * np.sin(theta),  scale * np.cos(theta)]
]
w.wcs.cunit = ["deg", "deg"]
w.wcs.equinox = 2000.0

data = np.zeros((size, size), dtype=np.float32)
np.random.seed(42)

n_stars = 900
xs = np.random.randint(0, size, n_stars)
ys = np.random.randint(0, size, n_stars)
fluxes = np.random.uniform(200, 800, n_stars)

for x, y, f in zip(xs, ys, fluxes):
    data[y, x] += f
 
data = gaussian_filter(data, sigma=1.0)

x1, y1 = 400, 850
x2, y2 = 1600, 1150
num_points = 2000
x_vals = np.linspace(x1, x2, num_points)
y_vals = np.linspace(y1, y2, num_points)

for x, y in zip(x_vals, y_vals):
    xi, yi = int(x), int(y)
    if 1 <= xi < size - 1 and 1 <= yi < size - 1:
        data[yi - 2:yi + 2, xi - 2:xi + 2] += 2.5e4

data = gaussian_filter(data, sigma=0.5)

hdu = fits.PrimaryHDU(data)
hdr = hdu.header
hdr.update(w.to_header())
hdr["DATE-OBS"] = obs_time.isot
hdr["OBSGEO-L"] = location.lon.deg
hdr["OBSGEO-B"] = location.lat.deg
hdr["OBSGEO-H"] = location.height.to(u.m).value
hdr["TELESCOP"] = "3.6m Devasthal Optical Telescope"
hdr["INSTRUME"] = "CCD Imager"
hdr["OBSNAME"] = "Devasthal Optical Telescope (ARIES)"
hdr["OBJECT"] = "Starlink-1008"
hdr["EXPTIME"] = 0.1
hdr["COMMENT"] = "Synthetic Starlink streak with realistic WCS and Devasthal metadata"

hdu.writeto(fits_out, overwrite=True)
print(f"Wrote FITS: {fits_out}")

plt.figure(figsize=(6, 6))
plt.imshow(np.log10(data + 1), origin="lower", cmap="plasma")
plt.title("Synthetic Starlink-1008 (Devasthal WCS)")
plt.axis("off")
plt.tight_layout()
plt.savefig(png_out, dpi=150)
plt.close()
print(f"Wrote preview PNG: {png_out}")

print("\nWCS and Telescope Info")
print(f"Center (RA, Dec): ({ra_cen}, {dec_cen})")
print(f"Rotation: {rotation_deg}°")
print(f"Pixel scale: {pix_scale_arcsec:.3f} arcsec/pix")
print(f"Observatory: Devasthal @ ({location.lat.deg:.3f}N, {location.lon.deg:.3f}E, {location.height.to(u.m).value} m)")
print(f"Streak from ({x1},{y1}) → ({x2},{y2}), length = {np.hypot(x2-x1, y2-y1):.1f} px")
