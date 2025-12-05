import torch
import torch.nn as nn
from typing import List, Literal, Optional
from dataclasses import dataclass
import torch.optim as optim
import numpy as np
import pickle
import math
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import torch.nn.functional as F

# importing nets and buffer
from TD3Agent.critic import Critic
from SACAgent.gaussian_actor import GaussianActor
from TD3Agent.replay_buffer import PERRecentReplayBuffer, ReplayBuffer

import os
from datetime import datetime


# ----------------
# Utilities
# ----------------
def get_device() -> torch.device:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device


def col(x: torch.Tensor) -> torch.Tensor:
    """
    Enforce column shape [B, 1].
    """
    if x.ndim == 1:
        return x.view(-1, 1)
    return x

@torch.no_grad()
def hard_update(target: nn.Module, online: nn.Module) -> None:
    target.load_state_dict(online.state_dict())


@torch.no_grad()
def soft_update(target: nn.Module, online: nn.Module, tau: float) -> None:
    for tp, p in zip(target.parameters(), online.parameters()):
        tp.data.mul_(1.0 - tau).add_(tau * p.data)


class SACAgent(nn.Module):
    """
    Soft Actor Critic Agent
    """
    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            actor_hidden: List[int],
            critic_hidden: List[int],
            # learning
            gamma: float = 0.99,
            actor_lr: float = 1e-4,
            critic_lr: float = 1e-3,
            alpha_lr: float = 1e-4,
            batch_size: int = 256,
            grad_clip_norm: Optional[float] = 10.0,
            # entropy/ temperature
            init_alpha: float = 0.2,
            learn_alpha: bool = True,
            target_entropy: Optional[float] = None,
            # targets update
            target_update: Literal["soft", "hard"] = "soft",
            tau: float = 0.005,
            hard_update_interval: int = 10_000,
            # architecture of the actor and critic
            activation: str = "relu",
            use_layernorm: bool = False,
            dropout: float = 0.0,
            max_action: float = 1.0,
            # buffer
            buffer_size: int = 1_000_000,
            use_per: bool = True,
            # device
            device: Optional[torch.device] = None,
            use_adamw: bool = True,
            actor_freeze: int = 0,
    ):
        super(SACAgent, self).__init__()
        self.device = device if device is not None else get_device()

        # hparams
        self.gamma = gamma
        self.batch_size = batch_size
        self.grad_clip_norm = grad_clip_norm
        self.target_update = target_update
        self.tau = tau
        self.hard_update_interval = hard_update_interval
        self.max_action = float(max_action)
        self.use_per = use_per
        self.actor_freeze = int(actor_freeze)

        # ---- critic (double Q) ----
        self.critic = Critic(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=critic_hidden,
            activation=activation,
            use_layernorm=use_layernorm,
            dropout=dropout,
        ).to(self.device)

        self.critic_target = Critic(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=critic_hidden,
            activation=activation,
            use_layernorm=use_layernorm,
            dropout=dropout,
        ).to(self.device)

        hard_update(self.critic_target, self.critic)

        self.loss_fn_critic = nn.SmoothL1Loss(reduction="none")

        # ---- actor (Gaussian policy) ----
        self.actor = GaussianActor(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=actor_hidden,
            activation=activation,
            use_layer_norm=use_layernorm,
            dropout=dropout,
            max_action=max_action,
        ).to(self.device)

        # ---- temperature (alpha) ----
        self.learn_alpha = learn_alpha
        # log_alpha will be parameterized so alpha = exp(log_alpha) > 0
        self.log_alpha = torch.tensor(
            np.log(init_alpha), dtype=torch.float32, device=self.device, requires_grad=True
        )

        # --- optimizers and loss function ---
        if use_adamw:
            self.actor_optimizer = optim.AdamW(self.actor.parameters(), lr=actor_lr, weight_decay=0.0)
            self.critic_optimizer = optim.AdamW(self.critic.parameters(), lr=critic_lr, weight_decay=0.0)
            self.alpha_optimizer = optim.AdamW([self.log_alpha], lr=alpha_lr, weight_decay=0.0)
        else:
            self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
            self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)

        if target_entropy is None:
            # default heuristic
            self.target_entropy = - float(action_dim)
        else:
            self.target_entropy = float(target_entropy)

        # ---- replay buffer ----
        if use_per:
            self.buffer = PERRecentReplayBuffer(buffer_size, state_dim, action_dim)
        else:
            self.buffer = ReplayBuffer(buffer_size, state_dim, action_dim)

        # ---- counters / logs ----
        self.train_steps = 0
        self.total_it = 0
        self.actor_losses = []
        self.critic_losses = []
        self.alpha_losses = []
        self.alphas = []

    # ---- interactions ----
    @torch.no_grad()
    def act(self, state: np.ndarray, eval_mode: bool = False) -> np.ndarray:
        """
        Map a state to an action
        - eval_mode=True: deterministic mean
        - eval_mode=False: stochastic
        """
        self.actor.eval()
        s = torch.as_tensor(state, device=self.device, dtype=torch.float32).unsqueeze(0)
        if eval_mode:
            a = self.actor.deterministic_action(s)
        else:
            a, log_prob, mean_a = self.actor.sample(s)
        self.actor.train()
        return a.squeeze(0).cpu().numpy()

    @torch.no_grad()
    def take_action(self, state: np.ndarray, explore: bool = False) -> np.ndarray:
        """
        In same style with TD3
        """
        return self.act(state, eval_mode=not explore)

    @torch.no_grad()
    def act_eval(self, state: np.ndarray, sigma_eval: float = 0.0) -> np.ndarray:
        a = self.act(state, eval_mode=True)
        if sigma_eval > 0.0:
            noise = np.random.randn(*a.shape) * sigma_eval
            a = np.clip(a + noise, -self.max_action, self.max_action)
        return a

    def push(self, s, a, r, ns, done):
        """
        store an experience into the buffer
        """

        self.buffer.push(s, a, r, ns, done)

    def pretrain_push(self, s, a, r, ns):
        """
        Pretraining: bulk insert (s, a, r, ns) from MPC into the buffer.
        """
        if not hasattr(self.buffer, "pretrain_add"):
            raise RuntimeError("Replay buffer does not support pretrain_add (set use_per=False).")

        self.buffer.pretrain_add(s, a, r, ns)


    def pretrain_from_buffer(
            self,
            num_updates: int = 50000,
            log_interval: int = 1000,
    ):
        """
        Offline pretraining on the MPC dataset.

        - Critic: SAC target y = r + gamma * (Q_tgt - alpha * log_pi).
        - Actor: pure behavioral cloning of MPC actions (MSE on deterministic action).
        - Alpha: optional update using the current policy entropy.
        """

        if len(self.buffer) < self.batch_size:
            raise RuntimeError("Buffer is less than the batch size")

        self.actor.train()
        self.critic.train()

        logs = {
            "actor_bc_loss": [],
            "critic_td_loss": [],
            "alpha_loss": [],
        }

        for it in range(1, num_updates + 1):
            # sample batch (PER or uniform)
            if not self.use_per:
                s, a, r, ns, done = self.buffer.sample(self.batch_size, device=self.device)
                idx = None
                is_w = torch.ones((self.batch_size, 1), device=self.device)
            else:
                s, a, r, ns, done, idx, is_w = self.buffer.sample(self.batch_size, device=self.device)
                is_w = col(is_w.to(self.device, non_blocking=True).float())

            s = s.to(self.device, non_blocking=True).float()
            a = a.to(self.device, non_blocking=True).float()
            r = col(r.to(self.device, non_blocking=True).float())
            ns = ns.to(self.device, non_blocking=True).float()
            done = col(done.to(self.device, non_blocking=True).float())

            # ----- critic target -----
            with torch.no_grad():
                next_action, logp_next, mean_next = self.actor.sample(ns)
                q_next = self.critic_target.combined_forward(ns, next_action, mode="min")
                q_next = col(q_next)
                alpha = self.log_alpha.exp()
                y = r + self.gamma * (1.0 - done) * (q_next - alpha * logp_next)

            # critic update
            q1, q2 = self.critic(s, a)
            q1 = col(q1)
            q2 = col(q2)
            l1 = self.loss_fn_critic(q1, y)
            l2 = self.loss_fn_critic(q2, y)
            critic_loss = (is_w * (l1 + l2)).mean()

            self.critic_optimizer.zero_grad(set_to_none=True)
            critic_loss.backward()
            if self.grad_clip_norm is not None:
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip_norm)
            self.critic_optimizer.step()

            # PER priorities based on TD error
            if idx is not None and hasattr(self.buffer, "update_priorities"):
                td_errors = (y - q1).detach().abs().view(-1)
                self.buffer.update_priorities(idx, td_errors)

            # ----- actor: behavioral cloning of MPC actions -----
            mean_action = self.actor.deterministic_action(s)
            bc_loss = F.mse_loss(mean_action, a)

            self.actor_optimizer.zero_grad(set_to_none=True)
            bc_loss.backward()
            if self.grad_clip_norm is not None:
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip_norm)
            self.actor_optimizer.step()

            # ----- temperature (alpha) update -----
            if self.learn_alpha:
                # only update alpha, not the policy, so detach logp
                with torch.no_grad():
                    _, logp_new, _ = self.actor.sample(s)
                alpha_loss = -(self.log_alpha.exp() * (logp_new.detach() + self.target_entropy)).mean()

                self.alpha_optimizer.zero_grad(set_to_none=True)
                alpha_loss.backward()
                self.alpha_optimizer.step()
            else:
                alpha_loss = torch.tensor(0.0, device=self.device)

            # ----- target critic update -----
            if self.target_update == "soft":
                soft_update(self.critic_target, self.critic, self.tau)
            else:
                if it % self.hard_update_interval == 0:
                    hard_update(self.critic_target, self.critic)

            self.train_steps += 1
            self.total_it += 1

            logs["actor_bc_loss"].append(float(bc_loss.item()))
            logs["critic_td_loss"].append(float(critic_loss.item()))
            logs["alpha_loss"].append(float(alpha_loss.item()))

            if log_interval and (it % log_interval == 0):
                print(f"[SAC pretrain] it={it}  bc={bc_loss.item():.4e}  q={critic_loss.item():.4e}  alpha={self.log_alpha.exp().item():.3f}")

        # keep critic target synced at the end
        hard_update(self.critic_target, self.critic)

        return logs


    # ---- train step ----
    def train_step(self):
        """
        One SAC update:
            1) sample batch
            2) critic update
            3) actor update
            4) alpha update
            5) target update
        """
        # check if the buffer has enough info
        if len(self.buffer) < self.batch_size:
            return None

        if not self.use_per:
            s, a, r, ns, done = self.buffer.sample(self.batch_size, device=self.device)
            idx = None
            is_w = torch.ones((self.batch_size, 1), device=self.device)
        else:
            s, a, r, ns, done, idx, is_w = self.buffer.sample(self.batch_size, device=self.device)
            is_w = col(is_w.to(self.device, non_blocking=True).float())  # [B, 1]

        s = s.to(self.device, non_blocking=True).float()  # [B, S]
        a = a.to(self.device, non_blocking=True).float()  # [B, A]
        r = col(r.to(self.device, non_blocking=True).float())  # [B, 1]
        ns = ns.to(self.device, non_blocking=True).float()  # [B, S]
        done = col(done.to(self.device, non_blocking=True).float())  # [B, 1]

        # ----- critic target y: r + gamma * (Q_tgt - alpha * log_pi) -----
        with torch.no_grad():
            # sample a' and log_pi(a' | s')
            next_action, logp_next, mean_next = self.actor.sample(ns)
            # Q_target(s', a')
            q_next = self.critic_target.combined_forward(ns, next_action, mode="min")
            q_next = col(q_next)
            alpha = self.log_alpha.exp()
            # soft target
            y = r + self.gamma * (1.0 - done) * (q_next - alpha * logp_next)

        # critic update
        q1, q2 = self.critic(s, a)
        q1 = col(q1)
        q2 = col(q2)
        l1 = self.loss_fn_critic(q1, y)
        l2 = self.loss_fn_critic(q2, y)
        critic_loss = (is_w * (l1 + l2)).mean()

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        if self.grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip_norm)
        self.critic_optimizer.step()

        # TD errors for PER prority
        if idx is not None and hasattr(self.buffer, "update_priorities"):
            td_errors = (y - q1).detach().abs().view(-1)
            self.buffer.update_priorities(idx, td_errors)

        # ---- actor update ----
        # sample action from current policy for states s
        new_actions, logp_new, mean_new = self.actor.sample(s)
        # Q(s, a_new) using critic 1
        q_new = self.critic.combined_forward(s, new_actions, mode="q1")
        q_new = col(q_new)

        alpha = self.log_alpha.exp().detach()
        # policy loss: E[alpha * log_pi(a|s) - Q(s,a)]
        actor_loss = (alpha * logp_new - q_new).mean()

        if self.train_steps >= getattr(self, "actor_freeze", 0):
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            if self.grad_clip_norm is not None:
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip_norm)
            self.actor_optimizer.step()

        # ---- temperature (alpha) update ----
        if self.learn_alpha:
            # L(alpha) = E[-alpha * (log_pi + H_target)]
            alpha_loss = -(self.log_alpha.exp() * (logp_new + self.target_entropy).detach()).mean()

            self.alpha_optimizer.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_optimizer.step()
        else:
            alpha_loss = torch.tensor(0.0, device=self.device)

        # ----- target nets update -----
        if self.target_update == "soft":
            soft_update(self.critic_target, self.critic, self.tau)
        else:
            if self.train_steps % self.hard_update_interval == 0:
                hard_update(self.critic_target, self.critic)

        self.train_steps += 1
        self.total_it += 1

        # logging
        self.actor_losses.append(float(actor_loss.item()))
        self.critic_losses.append(float(critic_loss.item()))
        self.alpha_losses.append(float(alpha_loss.item()))
        self.alphas.append(float(self.log_alpha.exp().item()))

        return float(critic_loss.item()), float(actor_loss.item()), float(self.log_alpha.exp().item())

    def save(
            self,
            directory: str,
            prefix: str = None,
            include_optim: bool = False,):
        """
        Save a checkpoint to `directory` with filename
        f"{prefix}_{timestamp}.pkl".
        """
        os.makedirs(directory, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(directory, f"{prefix}_{timestamp}.pkl")

        payload = {}

        if hasattr(self, "actor"):
            payload["actor_state_dict"] = self.actor.state_dict()
        if hasattr(self, "critic"):
            payload["critic_state_dict"] = self.critic.state_dict()

        # --- save targets if they exist ---
        if hasattr(self, "actor_target"):
            payload["actor_target_state_dict"] = self.actor_target.state_dict()
        if hasattr(self, "critic_target"):
            payload["critic_target_state_dict"] = self.critic_target.state_dict()

        # --- hyperparams ---
        hparams = {}
        for name in [
            "gamma",
            "actor_lr",
            "critic_lr",
            "batch_size",
            "grad_clip_norm",
            "policy_delay",
            "target_update",
            "tau",
            "hard_update_interval",
            "target_combine",
            "t_std",
            "noise_clip",
            "max_action",
            "actor_freeze",
            "mode",
            "steps",
            "train_steps",
            "total_it",
            # SAC-specific
            "alpha",
            "alpha_lr",
            "target_entropy",
        ]:
            if hasattr(self, name):
                val = getattr(self, name)
                if hasattr(val, "item"):
                    val = float(val.item())
                hparams[name] = val

        if hasattr(self, "log_alpha"):
            # store as a scalar float
            hparams["log_alpha"] = float(self.log_alpha.detach().cpu().item())

        payload["hparams"] = hparams

        # --- optimizer states ---
        if include_optim:
            if hasattr(self, "actor_optimizer"):
                payload["actor_optimizer_state_dict"] = self.actor_optimizer.state_dict()
            if hasattr(self, "critic_optimizer"):
                payload["critic_optimizer_state_dict"] = self.critic_optimizer.state_dict()
            # SAC-specific: alpha_optimizer
            if hasattr(self, "alpha_optimizer"):
                payload["alpha_optimizer_state_dict"] = self.alpha_optimizer.state_dict()

        with open(path, "wb") as f:
            pickle.dump(payload, f)

        print(f"Saved checkpoint to: {path}")

    def load(self, path: str, load_optim: bool = False, load_alpha: bool = True):

        with open(path, "rb") as f:
            d = pickle.load(f)

        # ---- main nets ----
        if "actor_state_dict" in d and hasattr(self, "actor"):
            self.actor.load_state_dict(d["actor_state_dict"])
        else:
            print("[load] WARNING: no 'actor_state_dict' or no self.actor")

        if "critic_state_dict" in d and hasattr(self, "critic"):
            self.critic.load_state_dict(d["critic_state_dict"])
        else:
            print("[load] WARNING: no 'critic_state_dict' or no self.critic")

        # ---- targets (if class has them) ----
        if hasattr(self, "actor_target"):
            if "actor_target_state_dict" in d:
                self.actor_target.load_state_dict(d["actor_target_state_dict"])
            else:
                hard_update(self.actor_target, self.actor)

        if hasattr(self, "critic_target"):
            if "critic_target_state_dict" in d:
                self.critic_target.load_state_dict(d["critic_target_state_dict"])
            else:
                hard_update(self.critic_target, self.critic)

        # ---- hyperparams from checkpoint, if any ----
        hparams = d.get("hparams", {})

        actor_lr = getattr(self, "actor_lr", None)
        critic_lr = getattr(self, "critic_lr", None)

        if actor_lr is None:
            actor_lr = hparams.get("actor_lr", 3e-4)
            self.actor_lr = actor_lr
        if critic_lr is None:
            critic_lr = hparams.get("critic_lr", 3e-4)
            self.critic_lr = critic_lr

        # ---- (re)init optimizers ----
        self.actor_optimizer = torch.optim.AdamW(
            self.actor.parameters(), lr=actor_lr, weight_decay=0.0
        )
        self.critic_optimizer = torch.optim.AdamW(
            self.critic.parameters(), lr=critic_lr, weight_decay=0.0
        )

        if load_optim:
            if "actor_optimizer_state_dict" in d:
                self.actor_optimizer.load_state_dict(d["actor_optimizer_state_dict"])
            if "critic_optimizer_state_dict" in d:
                self.critic_optimizer.load_state_dict(d["critic_optimizer_state_dict"])

            # SAC-specific: alpha optimizer if present
            if hasattr(self, "alpha_optimizer") and "alpha_optimizer_state_dict" in d:
                self.alpha_optimizer.load_state_dict(d["alpha_optimizer_state_dict"])

        # ---- SAC-specific alpha stuff (safe for TD3; just checks) ----
        if "alpha" in hparams and hasattr(self, "alpha"):
            self.alpha = hparams["alpha"]

        if load_alpha and "log_alpha" in hparams and hasattr(self, "log_alpha"):
            with torch.no_grad():
                self.log_alpha.copy_(torch.tensor(hparams["log_alpha"], device=self.log_alpha.device))

        print(f"Agent loaded successfully from: {path}")
