#!/bin/bash
#
#SBATCH --job-name=ckws_adapted_refined_score_attack
#SBATCH --output=ckws_adapted_refined_score_attack-%j.out
#
#SBATCH --ntasks=1
#SBATCH --partition=main
#SBATCH --cpus-per-task=64
#SBATCH --constraint=avx2
#SBATCH --mem=485G

srun python3 -m ckws_adapted_score_attack.generate_results
