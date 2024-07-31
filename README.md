
```markdown
# Joint Motion Estimation with Geometric Deformation Correction for Fetal Echo Planar Images Via Deep Learning

This repository contains the code and data for the paper **"Joint Motion Estimation with Geometric Deformation Correction for Fetal Echo Planar Images Via Deep Learning"**.

## Abstract

Motion estimation and geometric deformation correction are critical for accurate analysis of fetal echo planar images. This paper presents a deep learning framework that jointly addresses these tasks, improving the accuracy and robustness of fetal imaging.

## Requirements

- Python 3.8+
- NumPy
- PyTorch
- SimpleITK

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/JointMotionEstimation.git
    cd JointMotionEstimation
    ```

2. Install the required packages:

    ```bash
    pip install numpy torch SimpleITK
    ```

## Usage

### Training

To train the model, run:

```bash
python Eq_tracking.py 
```

### Evaluation

To evaluate the model, run:

```bash
python Eq_testing.py 
```


## Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{wang2024joint,
  title={Joint Motion Estimation with Geometric Deformation Correction for Fetal Echo Planar Images Via Deep Learning},
  author={Wang, Jian and Faghihpirayesh, Razieh and Erdogmus, Deniz and Gholipour, Ali},
  booktitle={Medical Imaging with Deep Learning},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
