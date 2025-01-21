# Digital Immune

This repository contains training scripts for various models on different datasets. The models are trained using PyTorch and support distributed training using `torch.distributed`.

## Training Scripts

### sEMG Training

The `train_sEMG.py` script is used for training models on sEMG datasets.

#### Usage
```bash
python train_sEMG.py
```

### Image Training

The `train_pic.py` script is used for training models on image datasets.

#### Usage
```bash
python train_pic.py
```

### MSA Training

The `train_msa.py` script is used for training models on MSA datasets.

#### Usage
```bash
python train_msa.py
```

### MRI Training

The `train_mri.py` script is used for training models on MRI datasets.

#### Usage
```bash
python train_mri.py
```

## Requirements

- Python 3.8+
- PyTorch 1.8+
- CUDA 10.2+

## Installation

Clone the repository:
```bash
git clone https://github.com/IRMVLab/Digital_Immune.git
cd Digital_Immune
```

## License

This project is licensed under the MIT License.