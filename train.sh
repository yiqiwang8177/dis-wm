source $(conda info --base)/etc/profile.d/conda.sh
conda activate swm

HYDRA_FULL_ERROR=1 python train.py \
    data=tworoom \
    loader.batch_size=128 \
    trainer.default_root_dir='/zfsauton/scratch/yiqiw2/logs' \
    wandb.enabled=True