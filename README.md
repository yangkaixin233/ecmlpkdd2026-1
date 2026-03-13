# GA-MoE: Joint Geometry and Architecture Mixture of Experts for Graph Fraud Detection

This repository contains the official implementation of **GA-MoE**, a unified framework that jointly selects geometry and message-passing architecture at the node level for graph-based fraud detection.



## Datasets

We evaluate on four public fraud detection benchmarks:

| Dataset | #Nodes | #Edges | #Features | Fraud(%) | Homophily |
|---------|--------|--------|-----------|----------|-----------|
| FDCompCN | 5,317 | 30K | 57 | 10.51 | 0.96 |
| Amazon | 11,944 | 8.8M | 25 | 6.87 | 0.95 |
| YelpChi | 45,954 | 7.7M | 32 | 14.53 | 0.77 |
| T-Finance | 39,357 | 21.2M | 10 | 4.58 | 0.97 |

### Dataset Download

**YelpChi & Amazon:**
- Publicly available from [DGL](https://www.dgl.ai/) or [CARE-GNN repository](https://github.com/YingtongDou/CARE-GNN)
- Alternative: [GADBench benchmark](https://github.com/squareRoot3/GADBench)

**T-Finance:**
- Download from [BWGNN paper's Google Drive](https://drive.google.com/file/d/1VCOHkwaGU5d8JBkYXJCJGJKn-WGgLDXl/view)
- Reference: Tang et al., "Rethinking Graph Neural Networks for Anomaly Detection", ICML 2022

**FDCompCN:**
- Financial statement fraud dataset of Chinese companies from CSMAR database
- Contains three relations: C-I-C (investment), C-P-C (customer), C-S-C (supplier)
- Available from [SplitGNN repository](https://github.com/Yan-Xiaodi/SplitGNN)
- Reference: Wu et al., "SplitGNN: Spectral Graph Neural Network for Fraud Detection against Heterophily", CIKM 2023

### Data Format

Organize datasets as follows:

```
data/
├── amazon/
│   └── Amazon.mat
├── yelp/
│   └── YelpChi.mat
├── tfinance/
│   └── tfinance.npz
└── fdcompcn/
    └── comp.dgl
```

## Requirements

```
python >= 3.8
torch >= 1.11.0
dgl >= 0.9.1
numpy >= 1.22.4
scipy >= 1.4.1
scikit-learn >= 1.1.2
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Train on different datasets:

```bash
# Train on YelpChi
python train.py --dataset yelp --epochs 200 --patience 50

# Train on Amazon
python train.py --dataset amazon --epochs 200 --patience 50

# Train on T-Finance
python train.py --dataset tfinance --epochs 200 --patience 50

# Train on FDCompCN
python train.py --dataset fdcompcn --epochs 200 --patience 50
```

### Hyperparameters

Key hyperparameters (default values):

- `--hidden_dim`: Hidden dimension (default: 128)
- `--num_layers`: Number of GA-MoE layers (default: 2)
- `--lr`: Learning rate (default: 0.001)
- `--lambda1`: Contrastive loss weight (default: 0.1)
- `--lambda2`: Prototype loss weight (default: 0.05)
- `--lambda3`: Alignment loss weight (default: 0.1)
- `--lambda4`: Uniformity loss weight (default: 0.01)
- `--num_prototypes`: Number of prototypes per class (default: 3)
- `--temperature`: Temperature for contrastive learning (default: 0.5)


## Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{gamoe2026,
  title={GA-MoE: Joint Geometry and Architecture Mixture of Experts for Graph Fraud Detection},
  author={Anonymous},
  booktitle={European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases (ECML-PKDD)},
  year={2026}
}
```

## License

This project is licensed under the MIT License.

## Acknowledgments

We thank the authors of CARE-GNN, BWGNN, SplitGNN, and other baseline methods for making their datasets and code publicly available.
