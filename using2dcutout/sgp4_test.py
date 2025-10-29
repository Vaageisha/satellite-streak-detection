import sys
import numpy as np
from sgp4.api import Satrec, jday
from astropy.time import Time

print("Running SGP4 test...", flush=True)

# Starlink-1008 TLE
line1 = "1 44713U 19074A   25070.07810764  .00013954  00000-0  10203-2 0  9993"
line2 = "2 44713  53.0541  62.3301 0001629 137.3060  53.8953 15.70101947255788"

satellite = Satrec.twoline2rv(line1, line2)

# Observation time
obs_time = Time("2025-03-11T02:58:24.295", scale="utc")
jd, fr = jday(
    obs_time.datetime.year,
    obs_time.datetime.month,
    obs_time.datetime.day,
    obs_time.datetime.hour,
    obs_time.datetime.minute,
    obs_time.datetime.second
)

# Propagate orbit
error, r, v = satellite.sgp4(jd, fr)
r, v = np.array(r), np.array(v)

# Print results
print("ECI Position (km):", r.tolist(), flush=True)
print("ECI Velocity (km/s):", v.tolist(), flush=True)
print("Velocity magnitude (km/s):", np.linalg.norm(v), flush=True)
print("SGP4 propagation successful.", flush=True)
