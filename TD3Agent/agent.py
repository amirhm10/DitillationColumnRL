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
from TD3Agent.actor import Actor
from TD3Agent.replay_buffer import PERRecentReplayBuffer, ReplayBuffer

import os
from datetime import datetime


# ----------------
# Utilities
# ----------------
def get_device() -> torch.device:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device


def hard_update(target: nn.Module, online: nn.Module) -> None:
    target.load_state_dict(online.state_dict())


@torch.no_grad()
def soft_update(target: nn.Module, online: nn.Module, tau: float) -> None:
    for tp, p in zip(target.parameters(), online.parameters()):
        tp.data.mul_(1.0 - tau).add_(tau * p.data)


# exploration schedule for Gaussian action noise
@dataclass
class GaussianNoiseSchedule:
    std_start: float = 0.2
    std_end: float = 0.02
    mode: Literal["linear", "exp", "cosine"] = "exp"
    decay_steps: int = 200_000
    decay_rate: float = 0.99995

    def value(self, step: int) -> float:
        if self.mode == "linear":
            t = min(1.0, step / max(1, self.decay_steps))
            return self.std_start + (self.std_end - self.std_start) * t
        if self.mode == "exp":
            return self.std_end + (self.std_start - self.std_end) * (self.decay_rate ** step)
        if self.mode == "cosine":
            t = min(1.0, step / max(1, self.decay_steps))
            return self.std_end + 0.5 * (self.std_start - self.std_end) * (1 + math.cos(math.pi * t))
        raise ValueError("mode must be 'linear' | 'exp' | 'cosine'")


def col(x: torch.Tensor) -> torch.Tensor:
    return x if x.ndim == 2 else x.view(-1, 1)


def copy_params_by_order(new_model: nn.Module, old_model: nn.Module):
    with torch.no_grad():
        vec = parameters_to_vector(list(old_model.parameters()))
        vector_to_parameters(vec, list(new_model.parameters()))


class TD3Agent(nn.Module):
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
            batch_size: int = 256,
            grad_clip_norm: Optional[float] = 10.0,
            # TD3
            policy_delay: int = 2,
            target_policy_smoothing_noise_std: float = 0.2,
            noise_clip: float = 0.5,
            max_action: float = 1.0,
            # targets update
            target_update: Literal["soft", "hard"] = "soft",
            tau: float = 0.005,
            hard_update_interval: int = 10_000,
            target_combine: Literal["min", "max", "mean", "q1"] = "min",
            # architecture of the actor and critic
            activation: str = "relu",
            use_layernorm: bool = False,
            dropout: float = 0.0,
            squash: str = "tanh",
            # exploration
            exploration_schedule: Optional[GaussianNoiseSchedule] = None,
            std_start: float = 1.0,
            std_end: float = 0.05,
            std_decay_rate: float = 0.99,
            std_decay_steps: int = 100_000,
            std_decay_mode: Literal["linear", "exp", "cosine"] = "exp",
            # buffer
            buffer_size: int = 1_000_000,
            # device/opt
            device: Optional[torch.device] = None,
            use_adamw: bool = True,
            # actor freeze
            actor_freeze: int = 0,
            mode: str = None,
    ):
        super(TD3Agent, self).__init__()
        self.device = device if device is not None else get_device()

        # --- hparams ---
        self.gamma = gamma
        self.batch_size = batch_size
        self.grad_clip_norm = grad_clip_norm
        self.policy_delay = policy_delay
        self.t_std = target_policy_smoothing_noise_std
        self.noise_clip = noise_clip
        self.max_action = float(max_action)
        self.target_update = target_update
        self.tau = tau
        self.hard_update_interval = hard_update_interval
        self.target_combine = target_combine
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr

        self.steps = 0
        self.train_steps = 0
        self.total_it = 0

        self.actor_freeze = actor_freeze

        # --- actors and critic networks ---
        self.actor = Actor(state_dim, action_dim, actor_hidden,
                           activation=activation, use_layernorm=use_layernorm,
                           dropout=dropout, max_action=max_action, squash=squash).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, actor_hidden,
                                  activation=activation, use_layernorm=use_layernorm,
                                  dropout=dropout, max_action=max_action, squash=squash).to(self.device)
        self.critic = Critic(state_dim, action_dim, critic_hidden,
                             activation=activation, use_layernorm=use_layernorm,
                             dropout=dropout).to(self.device)
        self.critic_target = Critic(state_dim, action_dim, critic_hidden,
                                    activation=activation, use_layernorm=use_layernorm,
                                    dropout=dropout).to(self.device)

        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target, self.critic)

        # --- optimizers and loss function ---
        if use_adamw:
            self.actor_optimizer = optim.AdamW(self.actor.parameters(), lr=actor_lr, weight_decay=0.0)
            self.critic_optimizer = optim.AdamW(self.critic.parameters(), lr=critic_lr, weight_decay=0.0)
        else:
            self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
            self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.loss_fn_critic = nn.SmoothL1Loss(reduction="none")  # per-sample Huber

        # Train with MPC reward or not
        self.mode = mode

        # Buffer initialization
        if self.mode == "mpc":
            self.buffer = ReplayBuffer(buffer_size, state_dim, action_dim)
        else:
            self.buffer = PERRecentReplayBuffer(buffer_size, state_dim, action_dim)

        # logs
        self.actor_losses, self.critic_losses = [], []

        # decay scheduler
        self.expl_sched = exploration_schedule if exploration_schedule is not None else GaussianNoiseSchedule(
            std_start=std_start, std_end=std_end,
            decay_steps=std_decay_steps, mode=std_decay_mode,
            decay_rate=std_decay_rate,
        )


    # -------- interactions ------
    @torch.no_grad()
    def act_eval(self, state: np.ndarray, sigma_eval: float = 0.0) -> np.ndarray:
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        a = self.actor_target(s)
        if sigma_eval > 0.0:
            a = a + torch.randn_like(a) * sigma_eval
        return a.clamp(-self.max_action, self.max_action).cpu().numpy()

    @torch.no_grad()
    def take_action(self, state: np.ndarray, explore: bool = False) -> np.ndarray:
        self.steps += 1
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        a = self.actor(s).detach().cpu().numpy()
        if explore:
            self._expl_sigma = self.expl_sched.value(self.steps)
            a = a + np.random.randn(*a.shape) * self._expl_sigma
        return np.clip(a, -self.max_action, self.max_action)


    def push(self, s, a, r, ns, done):
        self.buffer.push(s, a, r, ns, done)

    def pretrain_push(self, s, a, r, ns,):
        self.buffer.pretrain_add(s, a, r, ns)

    def set_actor_lr(self, new_lr: float):
        for g in self.actor_optimizer.param_groups:
            g['lr'] = new_lr
        if self.total_it % 800 == 1:
            print(f"Actor learning rate changed to: {new_lr:.2e}")


    # ------ training ------
    def train_step(self) -> Optional[float]:
        # check if the buffer has enough info
        if len(self.buffer) < self.batch_size:
            return None

        if self.mode == "mpc":
            s, a, r, ns, done = self.buffer.sample(self.batch_size, device=self.device)
        else:
            s, a, r, ns, done, idx, is_w = self.buffer.sample(self.batch_size, device=self.device)
            is_w = col(is_w.to(self.device, non_blocking=True).float())  # [B, 1]

        s = s.to(self.device, non_blocking=True).float()  # [B, S]
        a = a.to(self.device, non_blocking=True).float()  # [B, A]
        r = col(r.to(self.device, non_blocking=True).float())  # [B, 1]
        ns = ns.to(self.device, non_blocking=True).float()  # [B, S]
        done = col(done.to(self.device, non_blocking=True).float())  # [B, 1]


        with torch.no_grad():
            # target policy smoothing
            base_next = self.actor_target(ns)
            noise = torch.empty_like(base_next).normal_(0.0, self.t_std)
            noise.clamp_(-self.noise_clip, self.noise_clip)
            next_action = (base_next + noise).clip(-self.max_action, self.max_action)

            # Next q value
            q_next = self.critic_target.combined_forward(ns, next_action, mode=self.target_combine)
            if self.mode == "mpc":
                y = r + self.gamma * q_next
            else:
                y = r + self.gamma * (1.0 - done) * q_next



        # critic update
        q1, q2 = self.critic(s, a)
        q1 = col(q1)
        q2 = col(q2)
        td = (y - q1).detach().abs().view(-1)
        l1 = self.loss_fn_critic(q1, y)
        l2 = self.loss_fn_critic(q2, y)
        if self.mode == "mpc":
            critic_loss = (l1 + l2).mean()
        else:
            critic_loss = (is_w * (l1 + l2)).mean()

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        if self.grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip_norm)
        self.critic_optimizer.step()
        self.critic_losses.append(float(critic_loss.item()))

        # ------ Delayed actor + target update -------
        if self.total_it % self.policy_delay == 0:
            curr = self.actor(s)
            q_for_actor = self.critic.q1_forward(s, curr)
            actor_loss = -torch.mean(q_for_actor)

            if self.total_it >= self.actor_freeze:
                self.actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                if self.grad_clip_norm is not None:
                    nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip_norm)
                self.actor_optimizer.step()
            self.actor_losses.append(float(actor_loss.item()))

            if self.target_update == "soft":
                soft_update(self.actor_target, self.actor, self.tau)
                soft_update(self.critic_target, self.critic, self.tau)
            else:
                if self.train_steps % self.hard_update_interval == 0:
                    hard_update(self.actor_target, self.actor)
                    hard_update(self.critic_target, self.critic)

        self.total_it += 1
        self.train_steps += 1

        # --------- PER: update priorities from |TD| ----------
        if self.mode != "mpc":
            if hasattr(self.buffer, "update_priorities"):
                self.buffer.update_priorities(idx, td.abs())

        return float(critic_loss.item())

    def pretrain_from_buffer(self,
                             num_updates: int=50000,
                             use_target_noise: bool=True,
                             log_interval: int=1000,
                             mode: str="mpc",):

        self.mode = mode

        if len(self.buffer) < self.batch_size:
           raise RuntimeError("Buffer is less than the batch size")

        self.actor.train()
        self.critic.train()

        logs = {
            "actor_bc_loss": [],
            "critic_td_loss": []
        }

        for it in range(1, num_updates+1):
            if self.mode == "mpc":
                s, a, r, ns, done = self.buffer.sample(self.batch_size, device=self.device)
            else:
                s, a, r, ns, done, idx, is_w = self.buffer.sample(self.batch_size, device=self.device)
                is_w = col(is_w.to(self.device, non_blocking=True).float())  # [B, 1]

            s = s.to(self.device, non_blocking=True).float()  # [B, S]
            a = a.to(self.device, non_blocking=True).float()  # [B, A]
            r = col(r.to(self.device, non_blocking=True).float())  # [B, 1]
            ns = ns.to(self.device, non_blocking=True).float()  # [B, S]
            done = col(done.to(self.device, non_blocking=True).float())  # [B, 1]

            with torch.no_grad():
                # target policy smoothing
                base_next = self.actor_target(ns)
                if use_target_noise:
                    noise = torch.empty_like(base_next).normal_(0.0, self.t_std)
                    noise.clamp_(-self.noise_clip, self.noise_clip)
                    next_action = (base_next + noise).clip(-self.max_action, self.max_action)
                else:
                    next_action = base_next.clip(-self.max_action, self.max_action)

                # Next q value
                q_next = self.critic_target.combined_forward(ns, next_action, mode=self.target_combine)
                if self.mode == "mpc":
                    y = r + self.gamma * q_next
                else:
                    y = r + self.gamma * (1.0 - done) * q_next

            # critic update
            q1, q2 = self.critic(s, a)
            q1 = col(q1)
            q2 = col(q2)
            l1 = self.loss_fn_critic(q1, y)
            l2 = self.loss_fn_critic(q2, y)
            critic_loss = (l1 + l2).mean()

            self.critic_optimizer.zero_grad(set_to_none=True)
            critic_loss.backward()
            if self.grad_clip_norm is not None:
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip_norm)
            self.critic_optimizer.step()

            # ----- Actor behavioral cloning
            pred_actions = self.actor(s)
            bc_loss = F.mse_loss(pred_actions, a)
            self.actor_optimizer.zero_grad(set_to_none=True)
            bc_loss.backward()
            if self.grad_clip_norm is not None:
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip_norm)
            self.actor_optimizer.step()

            if self.target_update == "soft":
                soft_update(self.actor_target, self.actor, self.tau)
                soft_update(self.critic_target, self.critic, self.tau)
            else:
                if it % self.hard_update_interval == 0:
                    hard_update(self.actor_target, self.actor)
                    hard_update(self.critic_target, self.critic)

            # logging and printing
            logs["actor_bc_loss"].append(float(bc_loss.item()))
            logs["critic_td_loss"].append(float(critic_loss.item()))

            if log_interval and (it % log_interval == 0):
                print(f"[pretrain] it={it}  bc={bc_loss.item():.4e}  q={critic_loss.item():.4e}")

        # keep targets synced at the end
        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target, self.critic)



    def load(self, path: str):
        with open(path, 'rb') as f:
            d = pickle.load(f)

        self.actor.load_state_dict(d['actor_state_dict'])
        self.critic.load_state_dict(d['critic_state_dict'])
        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target, self.critic)

        # re-init optimizers then load states
        self.actor_optimizer = torch.optim.AdamW(self.actor.parameters(), lr=self.actor_lr, weight_decay=0)
        self.critic_optimizer = torch.optim.AdamW(self.critic.parameters(), lr=self.critic_lr, weight_decay=0)

        print(f"Agent loaded successfully from: {path}")

    def save(
            self,
            directory: str,
            prefix: str = "td3",
            include_optim: bool = False,
    ) -> str:
        """
        Save a checkpoint to `directory` with filename based on current time.
        Returns the full path to the saved pickle.

        Notes:
        - Your existing `load(...)` only reads 'actor_state_dict' and 'critic_state_dict'.
          Extra keys saved here are harmless and simply ignored by that loader.
        - Set `include_optim=True` if you later add a loader that restores optimizers.
        """
        os.makedirs(directory, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(directory, f"{prefix}_{timestamp}.pkl")

        payload = {
            # what your current `load(...)` expects:
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),

            # nice-to-have extras (ignored by your current loader):
            "actor_target_state_dict": self.actor_target.state_dict(),
            "critic_target_state_dict": self.critic_target.state_dict(),
            "hparams": {
                "gamma": self.gamma,
                "actor_lr": self.actor_lr,
                "critic_lr": self.critic_lr,
                "batch_size": self.batch_size,
                "grad_clip_norm": self.grad_clip_norm,
                "policy_delay": self.policy_delay,
                "target_update": self.target_update,
                "tau": self.tau,
                "hard_update_interval": self.hard_update_interval,
                "target_combine": self.target_combine,
                "t_std": self.t_std,
                "noise_clip": self.noise_clip,
                "max_action": self.max_action,
                "actor_freeze": self.actor_freeze,
                "mode": self.mode,
                "steps": self.steps,
                "train_steps": self.train_steps,
                "total_it": self.total_it,
            },
        }

        if include_optim:
            payload["actor_optimizer_state_dict"] = self.actor_optimizer.state_dict()
            payload["critic_optimizer_state_dict"] = self.critic_optimizer.state_dict()

        with open(path, "wb") as f:
            pickle.dump(payload, f)

        print(f"Saved TD3 checkpoint to: {path}")
        return path

