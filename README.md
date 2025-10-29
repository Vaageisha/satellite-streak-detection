# Satellite Streak Detection

This project focuses on the automatic detection of satellite streaks in astronomical FITS images.  
It simulates satellite trails, detects them using image processing techniques, traces their RA/DEC coordinates, and verifies them against known satellite orbital data (TLE).

## Features
1.Automatic satellite streak detection using OpenCV and Hough Transform  
2.Coordinate mapping using Astropy WCS  
3.Velocity calculation and validation through SGP4 propagation  
4.Works with both simulated and real FITS data  

## Project Structure
/using2dcutout
-/make_synthetic_fits.py
-/cutout2d.py
-/comparetledata.py
-/sgp4_test.py

