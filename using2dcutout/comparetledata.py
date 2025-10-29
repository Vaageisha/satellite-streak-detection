import os
import numpy as np
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import EarthLocation, ITRS, GCRS, AltAz, SkyCoord
from sgp4.api import Satrec, jday
import astropy.units as u

print("compare_tledata.py started")

# Load FITS header info 
fits_path = os.path.join(os.path.dirname(__file__), "starlink_synthetic.fits")
print(f"Looking for FITS file at: {fits_path}")

if not os.path.exists(fits_path):
    raise FileNotFoundError("FITS file not found. Please check the path.")

hdu = fits.open(fits_path)[0]
hdr = hdu.header

print("\n FITS Header Summary ")
for key in ["OBJECT", "DATE-OBS", "TELESCOP", "INSTRUME", "EXPTIME", "OBSGEO-L", "OBSGEO-B", "OBSGEO-H"]:
    if key in hdr:
        print(f"{key:10}: {hdr[key]}")

object_name = hdr.get("OBJECT", "Unknown")
obs_time = Time(hdr["DATE-OBS"], scale="utc")
lon, lat, height = hdr["OBSGEO-L"], hdr["OBSGEO-B"], hdr["OBSGEO-H"]
observer = EarthLocation(lat=lat, lon=lon, height=height)

#  Starlink-1008 TLE 
line1 = "1 44713U 19074A   25070.07810764  .00013954  00000-0  10203-2 0  9993"
line2 = "2 44713  53.0541  62.3301 0001629 137.3060  53.8953 15.70101947255788"
satellite = Satrec.twoline2rv(line1, line2)

# Propagate using SGP4 
jd, fr = jday(
    obs_time.datetime.year,
    obs_time.datetime.month,
    obs_time.datetime.day,
    obs_time.datetime.hour,
    obs_time.datetime.minute,
    obs_time.datetime.second + obs_time.datetime.microsecond * 1e-6,
)
error, r, v = satellite.sgp4(jd, fr)
r, v = np.array(r), np.array(v)

print("\n  SGP4 Propagation Results ")
print(f"ECI Position (km): {r}")
print(f"ECI Velocity (km/s): {v}")

velocity_mag = np.linalg.norm(v)
print(f"TLE-predicted velocity magnitude: {velocity_mag:.3f} km/s")

# Convert ECI to topocentric RA/Dec from observer location 
# Build SkyCoord for satellite in GCRS frame
sat_coord = SkyCoord(GCRS(x=r[0]*u.km, y=r[1]*u.km, z=r[2]*u.km, obstime=obs_time))

# Transform to RA/Dec as seen from Earth location
topo = sat_coord.transform_to(AltAz(obstime=obs_time, location=observer))
radec = sat_coord.transform_to('icrs')

print("\n Apparent Position (Predicted) ")
print(f"RA:  {radec.ra.deg:.6f} deg")
print(f"Dec: {radec.dec.deg:.6f} deg")
print(f"Altitude: {topo.alt.deg:.3f} deg")
print(f"Azimuth : {topo.az.deg:.3f} deg")

#  Compare to observed image data 
# Replace these with your detected streak RA/Dec (from cutout2d.py)
ra_obs_start, dec_obs_start = 165.245299, 53.028470
ra_obs_end, dec_obs_end = 164.946376, 53.075594

ra_obs_mean = (ra_obs_start + ra_obs_end) / 2
dec_obs_mean = (dec_obs_start + dec_obs_end) / 2

sep_ra = abs(ra_obs_mean - radec.ra.deg)
sep_dec = abs(dec_obs_mean - radec.dec.deg)
total_sep = np.sqrt(sep_ra**2 + sep_dec**2)

print("\n Comparison with Detected Streak ")
print(f"Observed mean RA,Dec: ({ra_obs_mean:.6f}, {dec_obs_mean:.6f})")
print(f"Predicted RA,Dec:     ({radec.ra.deg:.6f}, {radec.dec.deg:.6f})")
print(f"Angular separation:   {total_sep:.4f} degrees")

#  Velocity comparison 
obs_velocity = 7.5  # from image-derived estimate
diff = abs(obs_velocity - velocity_mag)
match_percent = 100 * (1 - diff / obs_velocity)
print(f"\nObserved velocity: {obs_velocity:.3f} km/s")
print(f"TLE-predicted velocity: {velocity_mag:.3f} km/s")
print(f"Difference: {diff:.3f} km/s ({match_percent:.1f}% match)")

print("\n End of Script ")
