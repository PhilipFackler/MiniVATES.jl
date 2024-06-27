#!/bin/bash
#SBATCH -A CSC266
#SBATCH -J MiniVATES.jl
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -t 0:10:00
#SBATCH -p batch-gpu
#SBATCH -N 1
#SBATCH -G 4

export JULIA_DEPOT_PATH=/lustre/polis/csc266/scratch/4pf/julia_depot
cd /lustre/polis/csc266/scratch/4pf/MiniVATES.jl

srun -n 4 --gpus-per-task=1 julia --project test/benzil_corelli.jl
