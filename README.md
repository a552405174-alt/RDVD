"# RDVD" 


Repository Structure

.
├── Datasets/
│ ├── parking/ # Parking detection datasets
│ └── traffic/ # Traffic flow detection datasets
│
├── main.m # Main entry script of RDVD
├── part1_sliding_entropy_analysis.m
├── part2_vmd_decomposition.m
├── part3_correlation_analysis.m
├── part4_signal_reconstruction.m
├── part6_detection.m
├── part7_detection.m
└── README.md


How to Run
1. Prepare the Dataset

Place your magnetic signal data in the corresponding folder:

Parking detection dataset
Datasets/parking/

Traffic flow detection dataset
Datasets/traffic/

Set the File Path in main.m

Open main.m and set the dataset path according to your task.

Parking detection:

dataPath = './Datasets/parking/';


Traffic flow detection:

dataPath = './Datasets/traffic/';


Run the Program

In MATLAB, simply execute: main

The complete RDVD processing pipeline will be executed automatically.


RDVD Method Overview

The RDVD algorithm consists of the following stages:

Sliding Entropy Analysis
Extracts local complexity features of magnetic signals.
(part1_sliding_entropy_analysis.m)

Variational Mode Decomposition (VMD)
Decomposes the signal into multiple intrinsic mode functions (IMFs).
(part2_vmd_decomposition.m)

Correlation Analysis
Identifies vehicle-related components based on similarity measures.
(part3_correlation_analysis.m)

Signal Reconstruction
Reconstructs vehicle-induced magnetic disturbances.
(part4_signal_reconstruction.m)

Vehicle Detection
Performs adaptive threshold–based detection for traffic flow and roadside parking events.
(part6_detection.m, part7_detection.m)
