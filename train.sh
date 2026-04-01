 HYDRA_FULL_ERROR=1  python train.py \
    data=tworoom \
    dino_features=True \
    wandb.enabled=True \
    diswm=True \
    output_model_name=lewm_dino_diswm