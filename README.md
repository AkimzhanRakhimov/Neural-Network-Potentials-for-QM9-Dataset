# Neural Network Potentials for QM9 Dataset
This project implements a simple neural network potential (NNP) for small molecules using PyTorch and ASE. The goal is to predict molecular energies from atomic positions and types, and optionally run molecular dynamics simulations with forces computed via autograd.

# Features

MLP-based energy model (AtomMLP):

1. Takes atomic numbers and positions as input.
2. Computes pairwise distances up to 15 neighbors per atom.
3. Small, efficient model that can be trained on a subset of QM9 dataset.
4. Achieves low loss on small datasets (e.g., MSE ~0.86).

GNN-based energy model (AtomGNN):

1. Embeds atomic types.
2. Message passing layers propagate information between neighboring atoms within a cutoff.
3. Energy head sums atomic contributions.
4. Currently memory-intensive and hard to train on standard hardware.

# Molecular dynamics integration:

-Uses forces obtained via torch.autograd for simple velocity Verlet-like integration.
-Outputs trajectories as .xyz files for visualization with ASE.

Note: Atoms may "fly apart" due to unstable predicted forces from the network.

# Installation
```text
  pip install ase torch-geometric rdkit
```
## Project Structure

```text
.
├── src/
│   ├── data/         # Data loading, preprocessing, and batching
│   ├── models/       # Model definitions and training logic
│   ├── molecular_dynamics/    # Optional module for molecular dynamics simulation via autograd
│   └── requirements.txt/    # dependencies for the project
│
├── notebooks/
│   └── project.ipynb # Full experimental notebook
│
└── README.md
```

# Limitations

- Forces predicted by AtomMLP can be unstable, causing atoms to fly apart during dynamics.
- GNN model (AtomGNN) is memory-heavy, training on full QM9 molecules may not fit on GPU.
- Only a simple MLP has been successfully trained on a subset of QM9.
- Current implementation does not enforce energy conservation or long-term stability.

# Notes

This project is experimental. Use small datasets or simplified molecules for testing.
Recommended to start with AtomMLP before attempting GNN-based training.
Loss achieved on small MLP: ~0.86 (MSE).

# Future Improvements

- Stabilize forces for molecular dynamics.
- Optimize GNN for memory efficiency (batching, sparse edge representation).
- Incorporate periodic boundary conditions or long-range interactions.
- Extend to larger molecules and real simulations.

