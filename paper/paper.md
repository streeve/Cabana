---
title: 'Cabana: A Performance Portable Library for Particle-Based Simulations'
tags:
  - C++
  - Kokkos
  - particles
  - molecular dynamics
  - N-body cosmology
  - particle-in-cell
authors:
  - name: Stuart Slattery^[corresponding author]
    affiliation: 1
  - name: Samuel Temple Reeve
    orcid: 0000-0002-4250-9476
    affiliation: 1
  - name: Christoph Junghans
    affiliation: 2
  - name: Damien Lebrun-Grandié
    affiliation: 1
  - name: Robert Bird
    affiliation: 2
  - name: Guangye Chen
    affiliation: 2
  - name: Shane Fogerty
    affiliation: 2
  - name: Yuxing Qiu
    affiliation: 3
  - name: Aaron Scheinberg
    affiliation: 4
  - name: Stan Moore
    affiliation: 5
  - name: Steven Plimpton
    affiliation: 5
  - name: Timothy Germann
    affiliation: 2
  - name: James Belak
    affiliation: 6
  - name: Susan Mniszewski
    affiliation: 2
affiliations:
  - name: Oak Ridge National Laboratory
    index: 1
  - name: Los Alamos National Laboratory
    index: 2
  - name: University of California, Los Angeles
    index: 3
  - name: Jubilee Development
    index: 4
  - name: Sandia National Laboratories
    index: 5
  - name: Lawrence Livermore National Laboratory
    index: 6
date: 15 July 2021
bibliography: paper.bib
---

# Summary

Particle-based simulations are ubiquitous throughout many fields of
computational science and engineering, spanning the atomistic level with
molecular dynamics (MD), to mesoscale particle-in-cell (PIC) simulations for
solid mechanics, device-scale modeling with PIC methods for plasma physics, all
the way to N-body cosmology simulations of galaxy structures, with many other
methods in between [@hockney]. While these methods use particles to represent
significantly different entities with completely different physical models, many
low-level details are shared including performant algorithms for short- and/or
long-range particle interactions, multi-node particle communication patterns,
and other data management tasks such as particle sorting and neighbor list
construction.

`Cabana` is a performance portable library for particle-based simulations,
developed as part of the Co-Design Center for Particle Applications (CoPA)
within the Exascale Computing Project (ECP) [@ecp:2020]. The CoPA project and
its full development scope, including ECP partner applications, algorithm
development, and development of similar libraries for quantum MD, is described
in [@copa:2021]. `Cabana` uses the `Kokkos` library for on-node parallelism
[@kokkos:2014], enabling simulation on many-core CPU and GPU architectures, and
`MPI` for GPU-aware, multi-node communication. `Cabana` provides particle
simulation capabilities on all currently supported `Kokkos` backends, including
serial execution, `OpenMP` (including `OpenMP-Target` for GPUs), `CUDA` (NVIDIA
GPUs), `HIP` (AMD GPUs), and `SYCL` (Intel GPUs), providing a clear path for
future hardware. `Cabana` builds on `Kokkos` by providing new particle data
structures and particle algorithms resulting in a similar execution policy
based, node-level programming model. `Cabana` is an application and physics
agnostic, but particle-specific, toolkit intended to be used in conjunction with
`Kokkos` to generate an application or to be used as needed through interfaces
that wrap user memory.

# Statement of need

For the most part, particle simulation codes targeting high performance
computing have been developed specific to a given application area: examples
include `HACC` for cosmology [@hacc:2016], `LAMMPS` for atomic systems
[@lammps:1995], and `XGC` for plasma physics [@xgc:2018] (all ECP partner
applications for the CoPA project). In contrast, other areas of computational
science have successfully developed motif-based libraries, e.g. `AMReX` for
adaptive mesh based simulations [@amrex:2019], with many applications sharing
the development effort for common needs. Co-designing software for a simulation
"motif" such as particles is increasingly important as hardware for scientific
simulations continues to evolve, becoming more heterogeneous, requiring more
effort to extract performance, and otherwise likely requiring multiple versions
of a single application for each vendor-specific API. To address this need, our
objective is to provide scalable software services applicable to
high-performance scientific codes across numerous application domains through
general particle algorithms and data structures that are performant on a variety
of distributed memory and accelerated architectures in a single implementation.

## Particle capability

`Cabana` provides particle data structures to optimize performance across
hardware through an array-of-structs-of-arrays (AoSoA) concept. This directly
extends the `Kokkos::View` (portable multidimensional arrays) with an additional
vector length. Intermediate between struct-of-arrays (SoA) and array-of-structs
(AoS), the variable vector length makes the data layout configurable to achieve
the performance of AoS in SIMD-like settings and the memory locality of SoA.
This tunable layout was designed to enable optimal performance across multiple
kernels and across different hardware for a given application.

The main algorithmic functionality of the library includes particle neighbor
list generation and traversal, particle redistribution and halo communication
for domain decomposition, and particle sorting. Particle sorting currently
builds directly on `Kokkos` binning and sorting capabilities. Similarly,
parallel iteration within `Cabana` algorithms directly uses `Kokkos` options for
threaded parallelism. As with the AoSoA, `Cabana` extends the
`Kokkos::parallel_for` (portable parallel execution) with SIMD-parallel
capabilities for threading over the `AoSoA` data structures, as well as
neighbor-parallel iteration over both central particles and their neighbors
(potentially with many-body interactions and multiple levels of neighbors) with
user-configurable options for serial or threaded execution as needed for
application performance. Particle communication for multi-node simulation uses
GPU-aware `MPI`, with capabilities for migrating particles from one unique
owning rank to another, as well as to gather and scatter particle information
from owning ranks to neighboring ranks.

Existing packages are also leveraged for accelerating complex operations. For
example, an interface to `ArborX`, a library for performance portable geometric
search also built on `Kokkos`, has been included for neighbor list creation
which is more scalable for non-uniform particle distributions [@arborx:2020].

## Particle-grid capability

In addition to particle-specific algorithms, almost all particle-based codes use
grids in some way. To support these applications `Cabana` also provides
algorithms and data structures for many particle-grid motifs within the `Cajita`
subpackage. Distributed, logically rectilinear grid data structures and high
order, multidimensional spline kernels are available, together with the
requisite parallel communication to interpolate data between particles and
grids. While this is most relevant to PIC methods, grid structures are useful
even in simulations which are generally "mesh-free" (e.g. short-range MD), for
accelerating neighbor list generation and multi-node spatial decomposition.

As with the core particle package, `Cajita` includes interfaces to separate
libraries for complex particle-grid related motifs: distributed, performance
portable fast Fourier transforms are enabled for long-range MD and N-body
simulations with the `heFFTe` library [@heffte:2019]) and preconditioners and
solvers are provided for various flavors of PIC through the `HYPRE` library
[@hypre:2002].

## Tutorial, proxy applications, and Fortran support

An extensive set of documentation, testings, and examples are available for
`Cabana`, including unit tests, tutorial examples, and performance testing
examples across library functionality, described within the GitHub wiki and
`doxygen` API documentation. In addition, a separate repository exemplifies
using `Cabana` with Fortran application codes [@copa]. Many proxy applications
have also been developed using `Cabana`: `CabanaMD` for MD, `CabanaPIC` for
plasma PIC, and `ExaMPM` for the material point method (MPM) [@copa]. Proxy apps
are relatively simple representations of the main physics in production
applications and have proven useful within the `Cabana` development process for
demonstrating library needs, capability, and performance.

## Application adoption and future work

`Cabana` is designed for high-performance, large-scale particle simulations,
with early adoption by the `XGC` plasma physics code [@Scheinberg:2019], as well
as a new production MPM code for additive manufacturing, both a part of ECP. The
proxy apps developed thus far also demonstrate the potential for rapid
prototyping of particle codes on emerging hardware and interactions with
hardware vendors. One important aspect of continuing work is consistent
interaction with `Cabana`-based applications, contributing algorithms and data
structures back to `Cabana` where they could be useful across other particle
application. Similarly, interaction with the `Kokkos` team is critical to keep
`Cabana` up to date, but also to potentially contribute general parallel
approaches or data structures from `Cabana`, where appropriate. Other continuing
work includes tighter integration of the particle and particle-grid motifs, load
balancing, additional input/output capabilities, and performance optimizations
on early exascale systems.

# Acknowledgements

This work was performed as part of the Co-design Center for Particle
Applications, supported by the Exascale Computing Project (17-SC-20-SC), a
collaborative effort of the U.S. DOE Office of Science and the NNSA.

This work was performed at Lawrence Livermore National Laboratory under U.S.
Government Contract DE-AC52-07NA27344, Oak Ridge National Laboratory under U.S.
Government Contract DE-AC05-00OR22725, Los Alamos National Laboratory, and at
Sandia National Laboratories.

Los Alamos National Laboratory is operated by Triad National Security, LLC, for
the National Nuclear Security Administration of the U.S. Department of Energy
(Contract No. 89233218NCA000001).

Sandia National Laboratories is a multimission laboratory managed and operated
by National Technology and Engineering Solutions of Sandia, LLC., a wholly owned
subsidiary of Honeywell International, Inc., for the U.S. Department of Energy's
National Nuclear Security Administration under contract number  DE-NA-0003525.

This research used resources of the Oak Ridge Leadership Computing Facility
(OLCF),supported by DOE under the contract  DE-AC05-00OR22725.

# References
