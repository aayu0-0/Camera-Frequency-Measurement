# Camera-Frequency-Measurement

A computer vision application that estimates the vibration frequency of an oscillating object using a standard webcam and OpenCV.

---

## Overview

Traditional vibration measurement often requires specialized sensors. This project demonstrates that accurate frequency estimation can be achieved using only a camera.

An ArUco marker attached to the object is tracked frame-by-frame. The displacement data is converted into a time series, and Fast Fourier Transform (FFT) is applied to estimate the dominant vibration frequency.

---

## Features

* Real-time ArUco Marker Tracking
* Displacement Measurement
* FFT-Based Frequency Estimation
* Live Visualization
* Automatic Frequency Calculation

---

## Technologies Used

* Python
* OpenCV
* NumPy
* SciPy
* Matplotlib

---

## Project Structure

```
Camera-Frequency-Measurement/
│
├── assets/
├── src/
├── output/
├── requirements.txt
└── README.md
```

---

## Methodology

1. Detect ArUco marker
2. Track marker center
3. Record displacement
4. Apply FFT
5. Estimate dominant frequency
6. Display live results

---

## Applications

* Structural vibration analysis
* Educational demonstrations
* Mechanical systems
* Laboratory experiments
* Contactless frequency measurement

---

## Future Improvements

* Multi-axis vibration analysis
* Sub-pixel tracking
* Higher accuracy filtering
* Multiple marker tracking

---

## License

MIT License
