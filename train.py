import os
from functools import partial
from pathlib import Path
from xml.parsers.expat import model

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from jepa import JEPA
from module import ARPredictor, Embedder, MLP, SIGReg, StateExtractor
from utils import get_column_normalizer, get_img_preprocessor, ModelObjectCallBack
import timm

torch.hub.set_dir('/zfsauton/scratch/yiqiw2/.cache')

def lejepa_forward(self, batch, stage, cfg, static_weights=None):
    """encode observations, predict next states, compute losses."""

    ctx_len = cfg.wm.history_size
    n_preds = cfg.wm.num_preds
    lambd = cfg.loss.sigreg.weight
    static_sigreg = cfg.static_sigreg # if True, static state has sigreg

    # Replace NaN values with 0 (occurs at sequence boundaries)
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    output = self.model.encode(batch)

    emb = output["emb"]  # (B, T, D)
    emb_static = output["emb_static"] if "emb_static" in output else None 
    act_emb = output["act_emb"]

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, : ctx_len]
    # Teacher-forcing loss
    tgt_emb = emb[:, n_preds:] # label
    pred_emb = self.model.predict(ctx_emb, ctx_act, static_emb=emb_static[:, n_preds:] if emb_static is not None else None) # pred

    # LeWM loss
    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
    output["sigreg_loss"]= self.sigreg(emb.transpose(0, 1))
    output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"]  
    if "emb_static" in output:
        assert static_weights is not None, "static_weights must be provided for DisWM"
        if static_sigreg:
            output["sigreg_loss_static"]= self.sigreg(emb_static.transpose(0, 1))
            output["loss"] += lambd * output["sigreg_loss_static"]
        # static loss: pairwise loss 
        diff = emb_static.unsqueeze(2) - emb_static.unsqueeze(1) 
        pairwise_loss = diff.pow(2).mean(dim=-1) * static_weights.to(diff.device).unsqueeze(0)
        output["static_loss"] = pairwise_loss.mean()
        output["loss"] += output["static_loss"]
        static_emb_std = emb_static.std(dim=1).mean()
        output["static_emb_std"] = static_emb_std
        
        # -------------------------------------------------------------- #
        # Rollout loss                                                     #
        #                                                                  #
        # Starting from the context window, we autoregressively predict   #
        # `rollout_steps` steps ahead, reusing each predicted dynamic     #
        # embedding as the next input while keeping static_emb fixed.     #
        #                                                                  #
        # Diagram:                                                         #
        #   dyn:    emb0 -> emb_1 ->emb_2 -> ... #
        #   static: [fixed emb_static[:, 0]] broadcast across all steps   #
        #   target: emb1, emb2, emb3, ...,              #
        # -------------------------------------------------------------- #
        
        current_dyn = output["emb"][:, 0:1]  # (B, 1, D_static)
        rollout_preds = []
        max_rollout = output["emb"].shape[1]-1
        fixed_static = emb_static[:, 0:1].repeat(1, max_rollout, 1) # (B, 1, D_static)-> (B, max_rollout, D_static)
        rollout_act_emb  = output["act_emb"][:, :max_rollout]

        for step in range(max_rollout):
            act_step = rollout_act_emb[:, :step+1]  # (B, step, A)

            pred_step = self.model.predict(
                current_dyn,          # (B, step, D) 
                act_step,             # (B, step, A)
                static_emb=fixed_static[:,:step+1 ],  # (B, step, D_static)
            )[:,-1:]  # (B, 1, D)
            
            rollout_preds.append(pred_step)
            current_dyn = torch.cat([current_dyn, pred_step], dim=1)

        rollout_preds = torch.cat(rollout_preds, dim=1)      # (B, T-1, D)
        rollout_targets = output["emb"][:, n_preds:]               # (B, T-1, D)
        
        output["rollout_loss"] = (rollout_preds - rollout_targets).pow(2).mean()
        output["loss"] += output["rollout_loss"]   
        
    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    if "static_emb_std" in output:
        losses_dict[f"{stage}/static_emb_std"] = output["static_emb_std"].detach()
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output

@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    #########################
    ##       dataset       ##
    #########################

    dataset = swm.data.HDF5Dataset(**cfg.data.dataset, transform=None)
   
    # What I did here only match dino resize but interpolation is different, check timm doc for details
    transforms = [get_img_preprocessor(source='pixels', target='pixels', img_size=cfg.img_size)] if not cfg.dino_features else [get_img_preprocessor(source='pixels', target='pixels', img_size=256)]
    
    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels") or 'feature' in col:
                continue

            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)

            setattr(cfg.wm, f"{col}_dim", dataset.get_dim(col))
    
    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform
   

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )

    train = torch.utils.data.DataLoader(train_set, **cfg.loader,shuffle=True, drop_last=True, generator=rnd_gen)
    val = torch.utils.data.DataLoader(val_set, **cfg.loader, shuffle=False, drop_last=False)
    
    ##############################
    ##       model / optim      ##
    ##############################
    
    if not cfg.dino_features:
        encoder = spt.backbone.utils.vit_hf(
            cfg.encoder_scale,
            patch_size=cfg.patch_size,
            image_size=cfg.img_size,
            pretrained=False,
            use_mask_token=False,
        ) 
    else:
        encoder =  timm.create_model('vit_small_patch16_dinov3.lvd1689m', pretrained=True, num_classes=0,)
        encoder.eval()
        # set requires_grad to False for all parameters in the encoder
        for param in encoder.parameters():
            param.requires_grad = False

    diswm = cfg.get("diswm", False)
    state_extractor, state_encoding, static_weights = None, None, None
    if diswm:
        print("Using DisWM architecture with 2 branches.")
        state_extractor = StateExtractor(
            embed_dim=384 if cfg.dino_features else encoder.config.hidden_size,
            num_queries=2,
            hidden_dim=256,
            num_heads=1,
            num_layers=1,
        )
        state_encoding = MLP(
            input_dim=384 if cfg.dino_features else encoder.config.hidden_size,
            output_dim=cfg.wm.get("embed_dim", 384),
            hidden_dim=512,
            norm_fn=torch.nn.BatchNorm1d,
        )
        T = cfg.wm.history_size + 1 
        i = torch.arange(T).unsqueeze(1)  # shape (T, 1)
        j = torch.arange(T).unsqueeze(0)  # shape (1, T)
        diff = j - i
        """
        0, 1, 1/2,  1/3,        ...
        0, 0, 1,    1/2, 1/3,   ...
        0, 0, 0,    1,   1/2,   ...
        """
        static_weights = torch.where(diff > 0, 1.0 / diff, torch.zeros_like(diff, dtype=torch.float))
      
    hidden_dim = encoder.config.hidden_size if not cfg.dino_features else 384
    embed_dim = cfg.wm.get("embed_dim", hidden_dim)
    effective_act_dim = cfg.data.dataset.frameskip * cfg.wm.action_dim

    predictor = ARPredictor(
        num_frames=cfg.wm.history_size,
        input_dim=embed_dim,
        hidden_dim=hidden_dim,
        output_dim=hidden_dim,
        **cfg.predictor,
    )

    action_encoder = Embedder(input_dim=effective_act_dim, emb_dim=embed_dim)
    
    projector = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=torch.nn.BatchNorm1d,
    )

    predictor_proj = MLP(
        input_dim=hidden_dim if not cfg.diswm else hidden_dim + embed_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=torch.nn.BatchNorm1d,
    )

    world_model = JEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=predictor_proj,
        use_dino=cfg.dino_features,
        diswm=cfg.diswm,
        state_extractor=state_extractor,
        state_encoding=state_encoding,
    )
  
    optimizers = {
        'model_opt': {
            "modules": 'model',
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model = world_model,
        sigreg = SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(lejepa_forward, cfg=cfg, static_weights=static_weights),
        optim=optimizers,
    )

    ##########################
    ##       training       ##
    ##########################

    run_id = cfg.get("subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(), run_id)

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    object_dump_callback = ModelObjectCallBack(
        dirpath=run_dir, filename=cfg.output_model_name, epoch_interval=1,
    )
   
    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[object_dump_callback],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=run_dir / f"{cfg.output_model_name}_weights.ckpt",
    )

    manager()
    return


if __name__ == "__main__":
    run()
