import numpy as np
import torch


# ----------------------
# Simple Uniform Replay Buffer
# ----------------------
class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int, action_dim: int):
        self.capacity = int(capacity)
        self.ptr = 0
        self.size = 0
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)  # 0/1 floating

    def push(self, s, a, r, ns, done: bool):
        i = self.ptr
        self.states[i] = s
        self.actions[i] = a
        self.rewards[i] = r
        self.next_states[i] = ns
        self.dones[i] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device="cpu",):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.from_numpy(self.states[idx]).to(device),
            torch.from_numpy(self.actions[idx]).to(device),
            torch.from_numpy(self.rewards[idx]).to(device),
            torch.from_numpy(self.next_states[idx]).to(device),
            torch.from_numpy(self.dones[idx].astype(np.float32)).to(device),
        )

    def pretrain_add(self, states_p, actions_p, rewards_p, next_states_p):
        n = states_p.shape[0]
        if n + self.ptr > self.capacity:
            raise ValueError("Pretraining Data exceeds buffer capacity")

        self.states[self.ptr: self.ptr + n] = states_p
        self.actions[self.ptr: self.ptr + n] = actions_p
        self.rewards[self.ptr: self.ptr + n] = rewards_p
        self.next_states[self.ptr: self.ptr + n] = next_states_p
        self.ptr += n
        self.size = min(self.ptr, self.capacity)

    def __len__(self):
        return self.size


# ----------------------
# PER Recent Replay Buffer
# ----------------------
class PERRecentReplayBuffer:
    def __init__(
            self,
            capacity: int,
            state_dim: int,
            action_dim: int,
            eps: float = 1e-6,
            alpha: float = 0.6,  # PER exponent (0 = uniform, 1 = full PER)
            beta_start: float = 0.4,  # IS correction starts small, anneals to 1.0
            beta_end: float = 1.0,
            beta_steps: int = 50_000,  # steps over which beta anneals
    ):

        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.ptr = 0
        self.size = 0
        self.step_counter = 0

        self.states = np.zeros((self.capacity, self.state_dim), np.float32)
        self.actions = np.zeros((self.capacity, self.action_dim), np.float32)
        self.rewards = np.zeros((self.capacity,), np.float32)
        self.next_states = np.zeros((self.capacity, self.state_dim), np.float32)
        self.dones = np.zeros((self.capacity,), np.float32)  # 0/1 floats

        # PER + recency
        self.birth_step = np.zeros(self.capacity, np.int64)
        self.priorities = np.zeros(self.capacity, np.float32)
        self.eps = eps
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_steps = beta_steps
        self.beta_t = 0
        self._max_priority = 1.0

    # ----- helpers ------
    def _beta(self):
        frac = min(1.0, self.beta_t / max(1, self.beta_steps))
        return self.beta_start + frac * (self.beta_end - self.beta_start)

    def push(self, s, a, r, ns, done, p0=None):
        i = self.ptr
        self.states[i] = s
        self.actions[i] = a
        self.rewards[i] = r
        self.next_states[i] = ns
        self.dones[i] = float(done)

        pri = float(p0) if (p0 is not None and p0 > 0) else self._max_priority
        self.priorities[i] = pri
        self._max_priority = max(self._max_priority, pri)

        self.birth_step[i] = self.step_counter
        self.step_counter += 1

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
            self,
            batch_size: int,
            device="cpu",
            frac_per: float = 0.5,
            frac_recent: float = 0.2,
            recent_window=1000,
    ):

        assert self.size > 0
        k_per = int(batch_size * frac_per)
        k_recent = int(batch_size * frac_recent)
        k_uniform = batch_size - k_per - k_recent

        idx_all = np.arange(self.size)

        # recent pool
        recent_window = min(self.size, int(recent_window))
        if recent_window <= 0: recent_window = self.size
        cutoff = np.partition(self.birth_step[:self.size], -recent_window)[-recent_window]
        pool_recent = idx_all[self.birth_step[:self.size] >= cutoff]
        if pool_recent.size == 0:
            pool_recent = idx_all

        # Recent sampling
        idx_recent = np.random.choice(pool_recent, size=k_recent, replace=(pool_recent.size < k_recent))
        # Uniform sampling
        idx_uni = np.random.choice(self.size, size=k_uniform, replace=True)

        # PER draw
        if k_per > 0:
            pr = np.maximum(self.priorities[:self.size], self.eps)
            probs = pr ** self.alpha
            probs /= probs.sum()
            idx_per = np.random.choice(self.size, size=k_per, replace=True, p=probs)
            beta = self._beta()
            self.beta_t += 1
            w = (self.size * probs[idx_per]) ** (-beta)
            w /= w.max()
            isw_per = w.astype(np.float32)
        else:
            idx_per = np.array([], dtype=np.int64)
            isw_per = np.array([], dtype=np.float32)

        # merge
        idx = np.concatenate([idx_per, idx_recent, idx_uni])
        is_w = np.concatenate([isw_per, np.ones(k_recent + k_uniform, dtype=np.float32)], axis=0)

        # tensors
        s = torch.from_numpy(self.states[idx]).to(device)
        a = torch.from_numpy(self.actions[idx]).to(device)
        r = torch.from_numpy(self.rewards[idx]).to(device)
        ns = torch.from_numpy(self.next_states[idx]).to(device)
        d = torch.from_numpy(self.dones[idx]).to(device)
        is_w = torch.from_numpy(is_w).to(device)

        return s, a, r, ns, d, idx, is_w

    def update_priorities(self, idx, td_errors):
        # td errors
        if isinstance(td_errors, torch.Tensor):
            td = td_errors.detach().abs().view(-1).cpu().numpy()
        else:
            td = np.abs(td_errors).reshape(-1)
        p = td + self.eps
        # Clip to avoid extreme skew
        p = np.clip(p, 1e-4, 1e4)
        self.priorities[idx] = p.astype(np.float32)
        self._max_priority = max(self._max_priority, float(p.max()))

    def __len__(self):
        return self.size
