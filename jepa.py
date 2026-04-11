"""JEPA Implementation"""

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn

def detach_clone(v):
    return v.detach().clone() if torch.is_tensor(v) else v

class JEPA(nn.Module):

    def __init__(
        self,
        encoder,
        predictor,
        action_encoder,
        projector=None,
        pred_proj=None,
        use_dino=False,
        diswm=False,
        state_extractor=None,
        state_encoding=None,
    ):
        super().__init__()

        self.encoder = encoder
        self.use_dino = use_dino
        self.diswm = diswm # 2 branches
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.state_encoding = state_encoding
        self.projector = projector or nn.Identity()
        self.pred_proj = pred_proj or nn.Identity()
        self.state_extractor = state_extractor

    def encode(self, info):
        """Encode observations and actions into embeddings.
        info: dict with pixels and action keys
        """
       
        if not self.use_dino:
            pixels = info['pixels'].float()
            b = pixels.size(0)
            pixels = rearrange(pixels, "b t ... -> (b t) ...") # flatten for encoding
            output = self.encoder(pixels, interpolate_pos_encoding=True)
            pixels_emb = output.last_hidden_state[:, 0]  # cls token
        else:
            # use dino feature 
            pixels = info['pixels'].float()
            b = pixels.size(0)
            with torch.no_grad():
                feats = self.encoder.forward_features(rearrange(pixels, "b t ... -> (b t) ..."))
            if not self.diswm:
                pixels_emb = feats[:, 0]  
            else:
                pixels_emb = self.state_extractor(feats[:, 5:])   # get rid of cls token and 4  registers 
                
        if not self.diswm:
            emb = self.projector(pixels_emb)
        else:
            emb = self.projector(pixels_emb[:, 0])  # dynamic branch
            emb_static = self.state_encoding(pixels_emb[:, 1])  # static branch
            info["emb_static"] = rearrange(emb_static, "(b t) d -> b t d", b=b)
        info["emb"] = rearrange(emb, "(b t) d -> b t d", b=b)

        if "action" in info:
            info["act_emb"] = self.action_encoder(info["action"])

        return info

    def predict(self, emb, act_emb, static_emb=None):
        """Predict next state embedding
        emb: (B, T, D)
        act_emb: (B, T, A_emb)
        static_emb: (B, T, D_static) or None
        """
        preds = self.predictor(emb, act_emb)
        if self.diswm:
            try:
                preds = torch.cat([preds, static_emb], dim=-1)
            except:
                assert False, f"{preds.shape} {static_emb.shape}"
        preds = self.pred_proj(rearrange(preds, "b t d -> (b t) d"))
        preds = rearrange(preds, "(b t) d -> b t d", b=emb.size(0))
        return preds

    ####################
    ## Inference only ##
    ####################

    def rollout(self, info, action_sequence, history_size: int = 3):
        """Rollout the model given an initial info dict and action sequence.
        pixels: (B, S, T, C, H, W). E.g., 1x300x1x3x224x224
        action_sequence: (B, S, T, action_dim). E.g., 1x300x5x10
         - S is the number of action plan samples
         - T is the time horizon
        """

        assert "pixels" in info, "pixels not in info_dict"
        H = info["pixels"].size(2)
        B, S, T = action_sequence.shape[:3]
       
        act_0, act_future = torch.split(action_sequence, [H, T - H], dim=2)
        info["action"] = act_0
        n_steps = T - H

        # copy and encode initial info dict
        _init = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
        _init = self.encode(_init)
        emb = info["emb"] = _init["emb"].unsqueeze(1).expand(B, S, -1, -1)
        _init = {k: detach_clone(v) for k, v in _init.items()}

        # if diswm, fix static embedding from first encoded frame
        has_static, emb_static_flat = "emb_static" in _init, None
        if has_static:
            # _init["emb_static"]: (B, H, D_static) -> take first frame, expand over S
            emb_static_init = _init["emb_static"][:, 0:1]  # (B, 1, D_static)
            emb_static_init = emb_static_init.unsqueeze(1).expand(B, S, -1, -1)  # (B, S, 1, D_static)
            emb_static_flat = rearrange(emb_static_init, "b s ... -> (b s) ...").clone()  # (BS, 1, D_static)
            emb_static_flat = repeat(emb_static_flat, 'b t d -> b (T t) d', T = history_size) # (BS, history_size, D_static)
        # flatten batch and sample dimensions for rollout
        emb = rearrange(emb, "b s ... -> (b s) ...").clone()
        act = rearrange(act_0, "b s ... -> (b s) ...")
        act_future = rearrange(act_future, "b s ... -> (b s) ...")

        # rollout predictor autoregressively for n_steps
        HS = history_size
        for t in range(n_steps):
            act_emb = self.action_encoder(act)
            emb_trunc = emb[:, -HS:]  # (BS, HS, D)
            act_trunc = act_emb[:, -HS:]  # (BS, HS, A_emb)
            pred_emb = self.predict(emb_trunc, act_trunc, static_emb=emb_static_flat,
            )[:, -1:]  # (BS, 1, D)
            emb = torch.cat([emb, pred_emb], dim=1)  # (BS, T+1, D)

            next_act = act_future[:, t : t + 1, :]  # (BS, 1, action_dim)
            act = torch.cat([act, next_act], dim=1)  # (BS, T+1, action_dim)

        # predict the last state
        act_emb = self.action_encoder(act)  # (BS, T, A_emb)
        emb_trunc = emb[:, -HS:]  # (BS, HS, D)
        act_trunc = act_emb[:, -HS:]  # (BS, HS, A_emb)
        pred_emb = self.predict(emb_trunc, act_trunc, static_emb=emb_static_flat,)[:, -1:]  # (BS, 1, D)
        emb = torch.cat([emb, pred_emb], dim=1)

        # unflatten batch and sample dimensions
        pred_rollout = rearrange(emb, "(b s) ... -> b s ...", b=B, s=S)
        info["predicted_emb"] = pred_rollout
        if has_static:
            info["emb_static"] = rearrange(emb_static_flat, "(b s) ... -> b s ...", b=B, s=S)
        return info

    def criterion(self, info_dict: dict):
        """Compute the cost between predicted embeddings and goal embeddings."""
        pred_emb = info_dict["predicted_emb"]  # (B,S, T-1, dim)
        goal_emb = info_dict["goal_emb"]  # (B, S, T, dim)

        goal_emb = goal_emb[..., -1:, :].expand_as(pred_emb)

        # return last-step cost per action candidate
        cost = F.mse_loss(
            pred_emb[..., -1:, :],
            goal_emb[..., -1:, :].detach(),
            reduction="none",
        ).sum(dim=tuple(range(2, pred_emb.ndim)))  # (B, S)

        return cost

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor):
        """ Compute the cost of action candidates given an info dict with goal and initial state."""

        assert "goal" in info_dict, "goal not in info_dict"

        device = next(self.parameters()).device
        for k in list(info_dict.keys()):
            if torch.is_tensor(info_dict[k]):
                info_dict[k] = info_dict[k].to(device)

        goal = {k: v[:, 0] for k, v in info_dict.items() if torch.is_tensor(v)}
        goal["pixels"] = goal["goal"]

        for k in info_dict:
            if k.startswith("goal_"):
                goal[k[len("goal_") :]] = goal.pop(k)

        goal.pop("action")
        goal = self.encode(goal)

        info_dict["goal_emb"] = goal["emb"]
        info_dict = self.rollout(info_dict, action_candidates)

        cost = self.criterion(info_dict)
        
        return cost
