#!/bin/bash

MV_DIR=/home/4pf/MiniVATES.jl

module load nvhpc

#export CUDA_HOME=/sw/wombat/Nvidia_HPC_SDK/Linux_aarch64/24.5/cuda/12.4

# rm -f $MV_DIR/Manifest.toml
# rm -f $MV_DIR/LocalPreferences.toml

julia --project=$MV_DIR -e 'using Pkg; Pkg.instantiate()'

# MPI + CUDA
julia --project=$MV_DIR -e ' \
    using Pkg; \
    if !haskey(Pkg.project().dependencies, "MPIPreferences"); \
        Pkg.add("MPIPreferences"); \
    end; \
    using MPIPreferences; \
    MPIPreferences.use_jll_binary(); \
    using MPI; \
    MPI.install_mpiexecjl(force = true); \
    if !haskey(Pkg.project().dependencies, "CUDA"); \
        Pkg.add("CUDA"); \
    end; \
    using CUDA; \
    '

# JACC
julia --project=$MV_DIR -e ' \
    using Pkg; \
    jaccInfo = Pkg.dependencies()[Pkg.project().dependencies["JACC"]]; \
    if jaccInfo.git_revision != "v0.3.1"; \
        Pkg.add(; name="JACC", rev = "v0.3.1"); \
    end; \
    using JACC; \
    JACC.set_backend("cuda"); \
    '

# Verify the packages are installed correctly
julia --project=$MV_DIR -e 'using Pkg; Pkg.instantiate()'
julia --project=$MV_DIR -e ' \
    using Pkg; \
    Pkg.status(); \
    using CUDA; \
    CUDA.versioninfo(); \
    '
