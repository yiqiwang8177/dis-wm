#!/bin/bash 
#SBATCH --job-name=swm_tworoom		# Name of your job 
#SBATCH --output=logs/job_%j.out	# Standard output log (%j = Job ID) 
#SBATCH --error=logs/job_%j.err 	# Error log 
#SBATCH --partition=general 		# (General, Debug, Preempt or Cpu)
#SBATCH --qos=qos_general 		# Matches the partition for guaranteed priority
#SBATCH --ntasks=1 			# Number of tasks 
#SBATCH --cpus-per-task=8 		# CPU cores per task 
#SBATCH --mem=64G 			# Memory (RAM) limit 
#SBATCH --time=08:00:00 			# Time limit (D-HH:MM:SS) 
#SBATCH --gres=gpu:a5000:1 		# Request 1 a5000 GPU 

#Envs  
source $(conda info --base)/etc/profile.d/conda.sh
conda activate swm

# Your commands:  
srun python train.py \
    data=tworoom \
    loader.batch_size=128 \
    loader.prefetch_factor=3 \
    trainer.default_root_dir='/zfsauton/scratch/yiqiw2/logs' \
    wandb.enabled=True


