#!/bin/bash

#SBATCH --job-name=ckws_adapted_refined_score_attack_GPU
#SBATCH --output=ckws_adapted_refined_score_attack_GPU-%j.out
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu_p100
#SBATCH --mem=242G
#SBATCH --gres=gpu:1
#SBATCH --constraint=avx2

module load nvidia/cuda-11.0
module load nvidia/cuda-11.0_cudnn-8.1
module load nvidia/cuda-11.0_tensorrt-7.2
module load nvidia/cuda-11.0_nccl-2.8
module load nvidia/nvtop

export TF_FORCE_GPU_ALLOW_GROWTH=true

srun python3 -m ckws_adapted_score_attack.generate_results
