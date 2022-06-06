#!/bin/bash
#
#SBATCH --job-name=ckws_adapted_refined_score_attack_preprocess
#SBATCH --output=ckws_adapted_refined_score_attack_preprocess-%j.out
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --constraint=avx2

srun python3 -m ckws_adapted_score_attack.dataset_preprocessor
