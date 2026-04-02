#!/bin/bash 
#SBATCH --job-name=swm_tworoom		# Name of your job 
#SBATCH --output=/zfsauton2/home/yiqiw2/slurmlogs/job_%j.out	# Standard output log (%j = Job ID) 
#SBATCH --error=/zfsauton2/home/yiqiw2/slurmlogs/job_%j.err 	# Error log 
#SBATCH --partition=debug 		# (General, Debug, Preempt or Cpu)
#SBATCH --qos=qos_debug 		# Matches the partition for guaranteed priority
#SBATCH --ntasks=1 			# Number of tasks 
#SBATCH --cpus-per-task=8 		# CPU cores per task 
#SBATCH --mem=64G 			# Memory (RAM) limit 
#SBATCH --time=00:05:00 			# Time limit (D-HH:MM:SS) 
#SBATCH --gres=gpu:a6000:1 		# Request 1 a6000 GPU 

#Envs  
source $(conda info --base)/etc/profile.d/conda.sh
conda activate swm

# Your commands:  
srun python train.py \
    data=tworoom \
    loader.batch_size=256 \
    trainer.default_root_dir='/zfsauton/scratch/yiqiw2/logs' \
    wandb.enabled=True

