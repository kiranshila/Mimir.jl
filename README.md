# Mimir
>
> ```
> I know where Othin's eye is hidden,
> Deep in the wide-famed well of Mimir;
> Mead from the pledge of Othin each morn
> Does Mimir drink: would you know yet more?
> ```

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://kiranshila.github.io/Mimir.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://kiranshila.github.io/Mimir.jl/dev)
[![Build Status](https://github.com/kiranshila/Mimir.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/kiranshila/Mimir.jl/actions/workflows/CI.yml?query=branch%3Amain)

## About

Mimir is a GPU-accelerated radio transient detection and processing pipeline, optimized for use in the Galactic Radio Explorer radio telescope.

TODO: Write more

## Stages

1. Read incoming spectrum data from UDP payloads into a ring buffer
2. Perform first-pass RFI mitigation and flagging
3. Dedisperse with [Dedisp.jl](https://github.com/kiranshila/Dedisp.jl)
4. Find SNR maxima for each possible pulse window size
5. Cluster with [Clustering.jl](https://github.com/JuliaStats/Clustering.jl)
6. Classify dedispersed dynamic spectra