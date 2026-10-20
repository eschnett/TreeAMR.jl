#!/bin/bash
# `bench/symmetry_nested.sh gpu` under an H200 job's directives: four
# GPUs of one node, one rank a GPU. See that script's header.
#
#     TREEAMR_REPO=<this checkout> TREEAMR_BASE=<baseline checkout> \
#         sbatch bench/symmetry_nested_gpu.sh
#
# A first run on h200debugq, two GPUs, to set the environments up and
# check them at a tiny size:
#
#     TREEAMR_NESTED_STAGES="setup smoke" sbatch --partition=h200debugq \
#         --gres=gpu:h200:2 --ntasks=2 --time=1:00:00 bench/symmetry_nested_gpu.sh

#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:h200:4
#SBATCH --mem=128G
#SBATCH --time=2:00:00
#SBATCH --job-name=treeamr-nested-gpu
#SBATCH --output=treeamr-nested-gpu-%j.out

set -euo pipefail
exec bash "${TREEAMR_REPO:?set TREEAMR_REPO to this checkout}/bench/symmetry_nested.sh" gpu
