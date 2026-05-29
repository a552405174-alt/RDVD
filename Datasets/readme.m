# RDVD Dataset

This repository provides the datasets used in the paper:

> RDVD: Recursive VMD-Based Denoising Framework for Magnetic Vehicle Detection Under Electromagnetic Interference

## Dataset Structure

```
Datasets/
├── parking/
└── traffic/
```

- **traffic/**: Dynamic vehicle detection dataset
- **parking/**: Parking detection dataset

## Dataset Description

Two magnetic vehicle detection datasets are provided:

### 1. Dynamic Vehicle Detection Dataset

The dynamic vehicle detection dataset contains **2,100 samples**. Each sample corresponds to the magnetic signal generated when multiple vehicles continuously pass over the magnetic sensor.

### 2. Parking Detection Dataset

The parking detection dataset contains **1,000 samples**. Each sample records the magnetic signal generated when a vehicle enters, occupies, or leaves a parking space.

All datasets are provided in **TXT format**.

## Data Format

Each TXT file contains four columns:

| Sequence | Time | Magnetic Field | Label |
|-----------|-----------|-----------|-----------|
| 1 | 37926725 | 180 | 0 |
| 2 | 37926806 | 181 | 0 |
| 3 | 37926887 | 100 | 1 |
| 4 | 37926968 | 97 | 1 |
| 5 | 37927049 | 185 | 0 |
| 6 | 37927130 | 178 | 0 |

### Column Description

| Column | Description |
|----------|----------|
| Sequence | Sample index |
| Time | Sampling timestamp |
| Magnetic Field | Measured magnetic field intensity |
| Label | Vehicle occupancy label |

For the label column:

- **0** indicates that no vehicle is present.
- **1** indicates that a vehicle is present.

An example label sequence is:

```
00000000111111111100000000
```

where consecutive `1`s correspond to the period during which a vehicle occupies the sensing region.

## Ground Truth Annotation

The ground-truth labels were manually annotated during on-site experiments according to the actual vehicle states observed in the field.

## Citation

If you use this dataset in your research, please cite our paper.

```bibtex
@article{RVDD,
  title={RVDD: Recursive VMD-Based Denoising Framework for Magnetic Vehicle Detection Under Electromagnetic Interference},
  author={...},
  journal={...},
  year={2025}
}
```

## License

This dataset is released for academic and research purposes only.
