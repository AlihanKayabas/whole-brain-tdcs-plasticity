Whole-Brain Modeling of tDCS Effects on Epileptic Networks

A computational pipeline to simulate the multi-scale effects of Transcranial Direct Current Stimulation (tDCS) on drug-resistant epilepsy. This repository contains the pipeline to simulate tDCS electric fields (FEM) and a reference implementation of the neural mass model used for the population dynamics

**Project Overview**

The core objective of this project is to predict how specific tDCS montages influence the long-term dynamics of epileptic networks. It bridges the gap between structural impact (where the current flows) and dynamic impact (how synapses evolve via plasticity).

The Scientific Pipeline

The framework operates in four sequential phases:

Anatomy & Physics (Volume Conductor):

Uses Finite Element Modeling (SimNIBS) to solve the Laplace equation on a realistic head mesh.

Computes the average normal Electric Field ($E_n$) for each Region of Interest (ROI).

Structural Connectivity (Connectome):

Defines the network topology using standardized consensus data from the Human Connectome Project (HCP).

Parcellated via the Desikan-Killiany atlas (68 or 246 nodes).

Neural Engine (The Core Model):

Node Dynamics: Simulates cortical activity using Neural Mass Models (Jansen-Rit framework) with distinct subpopulations (Pyramidal, SST, PV, VIP).

Coupling: Implements the "Lambda-E" formalism ($V_0(t) = V_{rest} + \lambda \cdot E_{ext}$) to translate E-fields into membrane perturbations.

Plasticity: Implements Calcium-dependent Spike-Timing Dependent Plasticity (Graupner-Brunel formalism).

Observation (Forward Model):

Projects source-level Local Field Potentials (LFP) to scalp EEG using Lead Field matrices (Brainstorm/OpenMEEG).

Technical Highlights

1. Numba-Accelerated Solver

To achieve high-performance whole-brain simulations (246 coupled nodes) without relying on proprietary C++ backends (e.g., eCoalia), the differential equation solvers are implemented using Numba JIT (Just-In-Time) compilation. This allows for pure Python code that executes at native machine speeds.

2. Selective Plasticity Masking

Simulating plasticity on a dense whole-brain matrix ($246 \times 246$ connections) is computationally expensive and biologically redundant for focal pathologies.

This implementation features a Plasticity Masking Strategy:

The electrical dynamics of the entire brain are simulated to preserve global network effects.

The computationally intensive plasticity equations are evolved only on specific pathways of interest (e.g., Epileptogenic Zone $\to$ Propagation Zone).

This approach reduces simulation time by orders of magnitude while maintaining mathematical rigor regarding whole-brain feedback loops.

Repository Structure

The codebase is organized to separate data handling, core modeling, and analysis logic.

.
├── data/
│   ├── templates/          # MNI152 / SimNIBS "Ernie" templates (Mock data for GDPR)
│   └── connectivity/       # HCP Consensus Connectome (SC matrices)
├── src/
│   ├── models.py           # CORE: Numba implementation of WholeBrainPlasticity class
│   ├── simulation_runner.py# EXECUTION: Scripts to run specific scenarios (NE1/LOC1, etc.)
│   ├── preprocessing.py    # INPUTS: Loading SimNIBS results & computing ROI averages
│   ├── analysis.py         # OUTPUTS: Signal processing utils
│   │                       #  └─ Page-Hinkley Algorithm (IED detection)
│   │                       #  └─ Nonlinear Correlation h² (Functional Connectivity)
│   └── optimization.py     # INVERSE: Brute-force finder for optimal EZ-PZ configurations
├── notebooks/
│   └── pipeline_demo.ipynb # End-to-end workflow demonstration
├── requirements.txt        # Dependencies (numpy, numba, scipy, etc.)
└── README.md


Getting Started

Prerequisites

Python 3.8+

Numpy, Scipy, Matplotlib

Numba

Installation

git clone [https://github.com/yourusername/whole-brain-tdcs-plasticity.git](https://github.com/yourusername/whole-brain-tdcs-plasticity.git)
cd whole-brain-tdcs-plasticity
pip install -r requirements.txt


Usage

To run a selective plasticity simulation (demonstrating the masking strategy):

from src.simulation_runner import run_selective_plasticity_simulation

# Run a 10-second simulation with plasticity enabled only from Node 0 (EZ) to Nodes 1,2,3 (PZ)
run_selective_plasticity_simulation(n_nodes=246, ez_index=0, pz_indices=[1, 2, 3], duration=10.0)


Data Availability & GDPR

Patient Privacy: To comply with GDPR regulations and ensure reproducibility, this repository does not contain private patient data.

Anatomical models demonstrated here use standard templates (MNI152 / SimNIBS "Ernie").

Connectivity matrices are derived from the Human Connectome Project (HCP) consensus data.

References

This codebase is a standalone reference implementation based on the methodology described in:

[Your Name], Modélisation neuro-inspirée de masses neuronales pour l’interprétation de la transition vers la crise et des effets de la tDCS chez les patients épileptiques, PhD Thesis, Sorbonne Université, Chapter 4.

Scientific Foundations:

Neural Mass Model: Wendling et al. (2002), Jansen & Rit (1995).

Plasticity Rule: Graupner & Brunel (2012), Köksal-Ersöz et al. (2024).
