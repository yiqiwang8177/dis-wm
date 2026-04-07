source $(conda info --base)/etc/profile.d/conda.sh
conda activate swm

# lewm:
# HYDRA_FULL_ERROR=1 python train.py \
#     data=tworoom \
#     loader.batch_size=128 \
#     trainer.default_root_dir='/zfsauton/scratch/yiqiw2/logs' \
#     wandb.enabled=True

# our own diswm: 
HYDRA_FULL_ERROR=1 python train.py \
    output_model_name=diswm \
    dino_features=True \
    diswm=True \
    data=tworoom \
    loader.batch_size=32 \
    trainer.default_root_dir='/zfsauton/scratch/yiqiw2/logs' \
    wandb.enabled=False