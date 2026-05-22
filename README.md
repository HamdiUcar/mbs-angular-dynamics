# Angular Dynamics of Magnetically Bounded Bodies with Rotating Magnetic Fields

A JupyterLab application for symbolic derivation, solution, simulation, visualization, numerical analysis, and eigenvalue analysis of the angular dynamics of magnetically bounded or free bodies subjected to rotating magnetic fields.

## Overview

This repository contains a symbolic, numerical, and visual analysis framework for studying the angular dynamics of free or magnetically bounded rigid bodies subjected to rotating magnetic fields. The application is implemented as a JupyterLab-based research utility and combines symbolic derivation, numerical simulation, animation, plotting, linearization, and eigenvalue analysis.

Magnetic bounding by rotating fields is a relatively new phenomenon in which a rigid body with a magnetic moment can be trapped at a precise location by a rotating magnetic field with a gradient, without requiring a local minimum of magnetic potential. The effect is essentially realizable with dipole magnets. In this interaction, the body couples to the field through its magnetic moment, which is fixed in the body frame, and may undergo translational and angular oscillations of typically small amplitude. These oscillations allow the body to find a quasi-stable equilibrium with the driving field and the confinement mechanism, which would not be possible in a purely magnetostatic dipole system except through special mechanisms such as diamagnetism.

The model is intended to simulate the angular dynamics of a magnetic body under the magnetic bound-state condition in classical physics, with user-defined physical and numerical parameters. The related work is described in Ucar (2021): https://doi.org/10.3390/sym13030442

## Physical Model

The body is assumed, in the general case, to have a diagonal moment-of-inertia tensor. Its magnetic moment is aligned with one principal body axis, taken as the body’s magnetic axis, and is centered at the center of mass. The applied magnetic field has constant strength and rotates about the laboratory/reference z-axis with user-defined angular velocity, field inclination, and strength.

The angular degrees of freedom are described by the three Euler angles $\theta$, $\phi$, and $\psi$. The angle $\psi$ denotes rotation of the body about its magnetic axis, while $\phi$ and $\theta$ describe the azimuthal and zenithal orientation of the magnetic moment in a coordinate system whose z-axis is the rotation axis of the driving field.

This work focuses on the angular part of the dynamics, which is responsible for keeping the body in proper alignment with the rotating field. In a basic model, the angular oscillation consists of a symmetric conical motion of the body’s magnetic moment around the rotation axis of the driving field, synchronized with that field. This alignment mechanism can be associated with the phase-lag property of driven oscillators. In this case, the phase is held close to $\pi$, causing the body’s magnetic moment to be antiparallel to the field in the azimuthal components, while parallel alignment occurs in the zenithal components of the magnetic moment and the rotating field.

## Equations of Motion

The equations of motion are derived using the Euler–Lagrange formalism in the Euler-angle coordinates $(\theta, \phi, \psi)$, for a rigid body with a diagonal moment-of-inertia tensor, one principal axis of which coincides with the magnetic axis. Axisymmetric and isotropic cases are also covered as reductions of the general model.

Although the system is externally driven and therefore non-conservative, the Euler–Lagrange approach can still be used because the magnetic generalized torques are obtained from derivatives of the magnetic potential with respect to the generalized coordinates. Viscous damping is incorporated separately as an external generalized force/torque.

The symbolic derivations are implemented using SymPy. In this project, the derivations are generated in the `EOMDiag` class, and additional symbolic variables useful for further analysis are also made available there.

## Synchronized Solutions and Eigenvalue Analysis

For an axisymmetric body, the equations of motion admit synchronized steady-state solutions in which the vectors entering the dynamics remain time-invariant in the reference frame of the rotating magnetic field. In this rotating-field frame, the explicit time dependence cancels and the equations of motion become autonomous.

These equilibria, supported by numerical simulations and experimental observations, provide the operating points for local linearization. The equations of motion are expanded to first order about the equilibrium state defined by the generalized coordinates and their first derivatives, yielding the state-matrix representation used for eigenvalue analysis.

The eigenvalue analysis exposes the local stability characteristics and natural angular frequencies of the synchronized solution. In the absence of damping, the system can be marginally stable, with eigenvalues whose real parts are zero. In the presence of damping, stable solutions become asymptotically stable when the real parts of all eigenvalues are negative. When the field inclination angle $\gamma$ is reduced below the required stability threshold, one real part becomes positive, clearly indicating an unstable regime.

Section C analyzes the behavior of the state variables in the presence of damping. In particular, it is shown that, in the absence of electromagnetically induced spin-up torques, $\dot{\psi}$ converges to $-\omega \cos\theta$. This condition corresponds to a zero z-component of the body’s angular velocity in local coordinates and provides a useful reference state for setting up damped simulations.

## Application Sections

The JupyterLab utility is organized into several sections:

- **Section A — Derivation of Equations of Motion**  
  Derives the equations of motion from three Euler rotations using the Euler–Lagrange formalism.

- **Section B — Equations and Basic Vectors**  
  Summarizes the equations of motion for a body with a diagonal moment-of-inertia tensor, together with the axisymmetric and isotropic reductions. It also lists basic vectors used in the derivations.

- **Section C — State Variables with Damping**  
  Covers the behavior of the state variables in the presence of damping, which is useful for setting up equilibrium or near-equilibrium initial conditions in simulations.

- **Section D — Linearization and Eigenvalue Analysis**  
  Presents the linearization of the equations of motion, the Jacobians, the state-matrix representation, and eigenvalue analysis. This section can calculate and plot eigenvalues with respect to selected parameters, making it possible to visualize stability limits and eigenfrequencies.

- **Section E — Semi-Empirical Stability Criterion**  
  Reformulates and simplifies the semi-empirical stability criterion originally proposed by Ucar (2021), expressing it in terms of the field elevation/tilt angle $\gamma$.

## Visualization and Numerical Tools

The utility includes numerical simulation, animation, and plotting tools for examining the time evolution of relevant system variables. It also provides phase portraits, polar plots, and additional figures that help visualize synchronized motion, excited bounded modes, beat formation, nonlinear amplitude limitation, damping effects, and transitions toward instability.

By sweeping parameters such as field rotation rate, field inclination, damping, magnetic torque strength, inertia ratios, and initial conditions, the user can observe how stable, unstable, quasi-synchronized, and transitional regimes emerge.

## Purpose

The purpose of this repository is to provide a reproducible computational environment for investigating magnetic bound-state dynamics beyond simple numerical simulation. The symbolic derivation of the equations of motion allows the assumptions, coordinate definitions, and model reductions to be inspected directly. The numerical solver enables time-domain simulations, while the animation, plotting, phase-portrait, and eigenvalue tools help connect nonlinear simulations with local stability analysis.

This utility can be used as both a research notebook and an exploratory computational laboratory for angular magnetic dynamics, including magnetically bounded motion, rotating-field stabilization, resonance-like behavior, damping-dependent equilibrium states, non-axisymmetric instability mechanisms, and the relationship between nonlinear simulations and linearized eigenvalue spectra.

## Derivations and Related Equations

A screenshot of the application showing derivations of equations of motion, their linearization and related equations characterizing the angular dynamics is available here:
[View derivation PDF](derivations.pdf)

## Requirements
* Python 3.10 or higher. Highest version tested: 3.14

## Libraries needed
* jupyterlab
* ipython
* ipywidgets
* jupyterlab_widgets
* ipympl
* ipyvuetify
* anywidget
* numpy
* scipy
* sympy
* mpmath
* pandas
* numba
* matplotlib
* plotly
* pyperclip
* ffmpeg-python

## Installation

You can install all the requirements using the following command:

```bash
pip install -r requirements.txt
```

## License

This project is licensed under the Apache License 2.0.

The official version of the project is maintained in this repository. Forks and modified versions are permitted under the license, but they should not be represented as the official version of this project.

If you use the derivations, formulations, code, simulations, figures, or analysis results from this repository, please cite the archived release.

## Citation

If you use this software, symbolic derivations, simulations, figures, or analysis results, please cite the archived release:

Hamdi Ucar, *Angular Dynamics of Magnetically Bounded Bodies with Rotating Magnetic Fields*, version v1.0.0, Zenodo. DOI: 10.5281/zenodo.20315447
