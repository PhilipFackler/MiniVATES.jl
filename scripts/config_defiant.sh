#!/bin/bash

# Change the first 3 lines appropriately
PROJ_DIR=/lustre/polis/csc266/scratch/4pf
export JULIA_DEPOT_PATH=$PROJ_DIR/julia_depot
MV_DIR=$PROJ_DIR/MiniVATES.jl
echo $MV_DIR

# good practice to avoid conflicts with existing default modules
module purge

# load required modules
module load PrgEnv-cray-amd
module load cray-mpich
module load julia

# remove existing generated Manifest.toml
rm -f $MV_DIR/Manifest.toml
rm -f $MV_DIR/LocalPreferences.toml

# Required to point at underlying modules above
export JULIA_AMDGPU_DISABLE_ARTIFACTS=1

julia --project=$MV_DIR -e 'using Pkg; Pkg.instantiate()'

# cray-mpich
julia --project=$MV_DIR -e 'using Pkg; Pkg.add("MPIPreferences")'
julia --project=$MV_DIR -e 'using MPIPreferences; MPIPreferences.use_system_binary(; library_names=["libmpi_cray"], mpiexec="srun")'

# amdgpu
julia --project=$MV_DIR -e 'using Pkg; Pkg.add("AMDGPU")'
julia --project=$MV_DIR -e 'using JACC; JACC.JACCPreferences.set_backend("amdgpu")'

# Verify the packages are installed correctly
julia --project=$MV_DIR -e 'using Pkg; Pkg.instantiate()'
