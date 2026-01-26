# RDVD 
This repository contains the MATLAB implementation of RDVD, a magnetic-signal–based algorithm for traffic flow detection and parking detection. The goal of RDVD is to accurately identify vehicle presence, entry, exit, and parking states from roadside magnetic sensor data with high robustness to noise and background interference.

## Key Features
Input: Raw magnetic sensor signals collected from roadside environments.
Output: Denoised signal and Vehicle detection results.

## Usage

Run `main.py` to execute the model and get the output:
   - **denoised signal**: Signal after noise reduction by our algorithm.
   - **detection result**: The vehicle detection result.


## File Descriptions

- **part1_sliding_entropy_analysis.m**: Extracts local complexity features of magnetic signals.
- **part2_vmd_decomposition.m**: Decomposes the signal into multiple intrinsic mode functions (IMFs).
- **part3_correlation_analysis.m**: Identifies vehicle-related components based on similarity measures.
- **part4_signal_reconstruction.m**: Reconstructs vehicle-induced magnetic disturbances.
- **part6_detection.m**: Performs adaptive threshold-based detection for traffic flow and roadside parking events.
- **part7_detection.m**: Performs adaptive threshold-based detection for traffic flow and roadside parking events.
