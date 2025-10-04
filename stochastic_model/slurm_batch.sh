#!/bin/bash
#SBATCH --job-name=ccdA_sim
#SBATCH --output=logs/job_%A_%a.out
#SBATCH --error=logs/job_%A_%a.err
#SBATCH --array=0-999         # 1000 parallel jobs (change as needed)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=3G
#SBATCH --account=virginia_igem
#SBATCH --time=00:50:00      # Adjust depending on runtime
#SBATCH --partition=standard # Change depending on your cluster config

module load python/3.8.20  # or the Python module on your system

# Create a directory for outputs if it doesn't exist
mkdir -p outputs

# Run your script with the array index as an argument
python3 mech_mod.py $SLURM_ARRAY_TASK_ID