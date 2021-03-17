# rbm_sampling
Restricted Boltzmann machine (RBM) implementation for the stationary state
of the isotropic 3D boundary-driven Heisenberg chain with incoherent dissipation and excitation on all sites
using accept-only and hybrid sampling strategies [arXiv:2012.10990 (2020)].

The provided code is the numerical realization of sampling strategies first introduced in https://arxiv.org/abs/2012.10990
to efficiently represent asymmetric open spin-1/2 quantum systems in the restricted Boltzmann machine neural network architecture.

OpenMP is used for parallelization. Required compile commands: -lm -fopenmp
