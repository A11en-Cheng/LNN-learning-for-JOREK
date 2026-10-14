# Hybrid Neural-Analytic ODE for Source-Aware Joint Dynamics

**Date:** 2026-10-13  
**Status:** Proposed Architecture  
**Context:** Resolution of timescale stiffness and spatial mapping overhead in JOREK-based Shattered Pellet Injection (SPI) surrogate modeling.

---

## 1. Context & Motivation

Simulating non-linear Magnetohydrodynamic (MHD) phenomena—specifically Shattered Pellet Injection (SPI) and thermal quench (TQ) dynamics within the JOREK framework—requires tracking complex, tightly coupled interactions between macroscopic plasma fluid dynamics, microscopic atomic processes, and solid pellet ablation. 

A central mathematical and computational bottleneck arises from the extreme timescale stiffness of the source terms and the spatial non-locality of moving impurity sources:

*   **The Failure of Classical Operator Splitting:** Traditional numerical schemes often rely on fractional step methods (e.g., Strang or Lie-Trotter operator splitting) to decouple fluid transport ODEs/PDEs from local atomic reaction-rate ODEs. However, when atomic ionization/recombination rates (governed by ADAS data) operate on nanosecond scales ($\tau_{\text{atomic}} \sim 10^{-9}$ s) while the global fluid time step operates on microsecond scales ($\tau_{\text{fluid}} \sim 10^{-6}$ s), operator splitting completely breaks the physical self-consistency. It introduces severe splitting errors, leading to non-physical numerical dissipation, phase lag in radiation collapse fronts, and spatial instabilities.
*   **The Computational Cost of Graph Neural Networks (GNNs):** To bypass Eulerian grid dependency for moving sources, previous iterations explored Graph Neural Networks for Lagrangian pellet tracking. However, GNNs incur an $\mathcal{O}(N^2)$ or high-constant $\mathcal{O}(N \cdot K)$ computational overhead during message passing over dense localized subgrids. Furthermore, standard autoregressive GNNs lack continuous-time mathematical guarantees, requiring artificial sub-stepping that exacerbates wall-clock overhead without fundamentally resolving structural stiffness.

This ADR proposes a **Hybrid Neural-Analytic ODE** architecture. By coupling Liquid Time-Constant (LTC) and Closed-form Continuous-depth (CfC) neural networks with exact analytical models for atomic physics, we construct a unified closed-form exponential integrator.

---

## 2. The Mathematical Framework: Joint Heterogeneous State Space

To restore the Markovian property and ensure a fully closed system, we define the complete system dynamics via a coupled, heterogeneous joint state space vector $Z(t)$.

The global state is defined as:
$$ Z(t) = \begin{bmatrix} U(t) \\ P(t) \\ C(t) \end{bmatrix} $$

Where the components represent:
*   **$U(t)$ (Macroscopic Fluid):** The Eulerian fluid state representation (9-field variables including $\psi, \Phi, \rho, T_e, T_i$, etc.) evaluated at the 4125 JOREK finite element nodes.
*   **$P(t)$ (Lagrangian Shards):** The discrete shard state vector encoding the active pellet fragments, containing their position, velocity, and radius/mass.
*   **$C(t)$ (Eulerian Cloud Moments):** The density vector of atomic charge states (e.g., $n_0 \dots n_{10}$ for Neon) projected onto the finite element grid, avoiding the tracking of millions of individual marker particles.

The governing continuous-time dynamics are formulated as a synchronized system:
$$ \frac{dZ}{dt} = \mathcal{F}_{trans}(U, C) + \mathcal{F}_{react}(C, U) + \mathcal{F}_{ablate}(P, U) $$

---

## 3. The Generalized Heterogeneous Closed-form Solution

To bypass explicit sub-stepping limits and resolve the extreme stiffness introduced by $\mathcal{F}_{react}$, we extend the CfC framework to operate directly over operator-split matrices derived from physical invariants.

### 3.1 The Piece-wise Autonomous Assumption (Quasi-Steady-State)
Over a discrete macroscopic temporal step $\Delta t$, we assume the local environmental parameters driving atomic rate coefficients (e.g., local $T_e$ and $n_e$) and the fluid transport coefficients remain piece-wise stationary relative to the ultra-fast atomic transitions. 

At $t=0$, we evaluate the neural transport rates and the physical ADAS matrices.

### 3.2 Exact Exponential Integration of Atomic Reactions (Analytic)
The atomic kinetic equation governing the charge states $C(t)$ is driven by a highly stiff reaction matrix $K$ (from ADAS) and an ablation source $B$. Under the autonomous assumption:
$$ \dot{C}(t) = K C(t) + B $$

This admits an exact analytical solution via matrix exponentiation (an Exponential Integrator):
$$ C(\Delta t) = \exp(K \Delta t) C_0 + K^{-1} \left( \exp(K \Delta t) - I \right) B_0 $$
*(Note: In implementation, singularity in $K$ due to mass conservation is handled via stable Taylor expansion of the integral term).* This guarantees unconditionally stable, non-negative, and mass-conserving charge-state evolution across arbitrarily large $\Delta t$, entirely eliminating the need for implicit Newton-Krylov iterations for the atomic physics.

### 3.3 Hadamard Gating for Fluid Transport (Neural)
To blend the non-linear continuous fluid transport with the exact atomic solution, the macroscopic fluid update is structured as a neural exponential integrator. The neural network predicts the local decay rate $\tau$ and target steady-state $U^*$:
$$ U(\Delta t) = U_0 \odot \exp(-\tau \Delta t) + U^* \odot (1 - \exp(-\tau \Delta t)) $$
Where $\odot$ denotes the Hadamard (element-wise) product. This serves as an adaptive gating mechanism, allowing each spatial node to independently relax toward its target state driven by the local source conditions.

---

## 4. Implementation Strategy: Spatial Mapping

To bridge the continuous Eulerian fields $U(\mathbf{x})$ and discrete Lagrangian particles $P$, we strictly reject learned MLP-based spatial mappings, which destroy geometric invariants and conservation laws. Instead, we enforce the geometric priors natively.

### 4.1 Field-to-Shard Mapping: Shape-Function Sparse Attention
When shards must read the local background plasma (e.g., $T_e, n_e$) to compute ablation rates, we utilize a highly constrained Cross-Attention mechanism. 
*   **Hard-coded Topology:** The attention adjacency matrix is restricted exclusively to the vertices of the specific finite element cell containing the shard.
*   **Analytic Weights:** The attention scores are completely overwritten by the exact JOREK Bicubic Hermite shape functions $N_i(\xi, \eta)$. 
This guarantees that the "neural" sampling of the local field matches the rigorous finite element interpolation to machine precision, with $\mathcal{O}(1)$ lookup cost per shard.

### 4.2 Shard-to-Field Projection: Eulerian Moments
Ablated mass and energy from the discrete shards must become source terms for the continuous Eulerian equations. Instead of generating millions of marker particles, the shard ablation rates are projected directly onto the 4125 Eulerian nodes as moment sources.
*   **Galerkin Projection:** We reuse the exact same shape functions (transpose mapping) multiplied by the corresponding Jacobian volume elements to scatter the discrete ablation source $B_{shard}$ back to the grid nodes: $B_{Eulerian} = W_{shape}^T B_{shard}$.
*   This strictly enforces global particle and energy conservation budgets natively within the architecture prior to the LTC/CfC time integration step.