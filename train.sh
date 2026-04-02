source $(conda info --base)/etc/profile.d/conda.sh
conda activate swm

python train.py \
    data=tworoom \
    loader.batch_size=256 \
    trainer.default_root_dir='/zfsauton/scratch/yiqiw2/logs' \
    wandb.enabled=False