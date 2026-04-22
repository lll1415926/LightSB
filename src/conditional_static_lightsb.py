import math
from typing import Optional

import torch
from torch import nn
from torch.distributions import Categorical, Independent, MixtureSameFamily, Normal


class StaticParentChildLightSB(nn.Module):
    """
    Static 2d -> d entropic-OT / LightSB-style model with cost

        c_{B,M}(x, y) = 1/2 (Bx - y)^T M (Bx - y),

    where:
        - x is the joint parent latent in R^{2d}
        - y is the child latent in R^d
        - B maps R^{2d} -> R^d
        - M is diagonal positive-definite by default

    Adjusted potential:
        eta_theta(y) = sum_k alpha_k N(y ; r_k, epsilon S_k)

    with diagonal S_k. This yields closed forms for:
        - log eta_theta(y)
        - log Z_theta(x)
        - pi_theta(y | x)
    """

    def __init__(
        self,
        latent_dim: int,
        n_components: int,
        epsilon: float,
        init_scale: float = 1.0,
        b_mode: str = "average",
        freeze_b: bool = False,
        learn_m: bool = False,
    ) -> None:
        super().__init__()
        assert latent_dim > 0
        assert n_components > 0
        assert epsilon > 0.0

        self.latent_dim = latent_dim
        self.n_components = n_components
        self.source_dim = 2 * latent_dim
        self.b_mode = b_mode

        self.register_buffer("epsilon", torch.tensor(float(epsilon)))

        self.log_alpha_raw = nn.Parameter(
            self.epsilon * torch.log(torch.ones(n_components) / n_components)
        )
        self.r = nn.Parameter(0.05 * torch.randn(n_components, latent_dim))
        self.S_log_diagonal = nn.Parameter(
            torch.full((n_components, latent_dim), math.log(init_scale))
        )

        self.register_buffer("B0_weight", self._make_B0_weight())
        if b_mode == "diag_gate":
            self.B = None
            self.alpha_logits = nn.Parameter(torch.zeros(latent_dim))
            self.alpha_logits.requires_grad_(not freeze_b)
        else:
            self.alpha_logits = None
            self.B = nn.Linear(self.source_dim, latent_dim, bias=False)
            self._init_B(b_mode)
            self.B.weight.requires_grad_(not freeze_b)

        self.log_M_diag = nn.Parameter(torch.zeros(latent_dim), requires_grad=learn_m)

    def _init_B(self, b_mode: str) -> None:
        with torch.no_grad():
            self.B.weight.zero_()
            d = self.latent_dim
            if b_mode == "first":
                self.B.weight[:, :d] = torch.eye(d)
            elif b_mode == "average":
                self.B.weight[:, :d] = 0.5 * torch.eye(d)
                self.B.weight[:, d:2 * d] = 0.5 * torch.eye(d)
            elif b_mode == "random":
                self.B.weight.normal_(mean=0.0, std=0.02)
            elif b_mode == "diag_gate":
                pass
            else:
                raise ValueError(f"Unknown b_mode: {b_mode}")

    def _make_B0_weight(self) -> torch.Tensor:
        d = self.latent_dim
        B0 = torch.zeros(d, self.source_dim)
        B0[:, :d] = 0.5 * torch.eye(d)
        B0[:, d:2 * d] = 0.5 * torch.eye(d)
        return B0

    def _build_joint_source(self, z_f: torch.Tensor, z_m: torch.Tensor) -> torch.Tensor:
        return torch.cat([z_f, z_m], dim=-1)

    def init_r_by_samples(self, samples: torch.Tensor) -> None:
        assert samples.shape[0] == self.n_components
        self.r.data.copy_(samples.to(self.r.device))

    def get_log_alpha(self) -> torch.Tensor:
        return (self.log_alpha_raw - self.log_alpha_raw.mean()) / self.epsilon

    def get_S(self) -> torch.Tensor:
        return torch.exp(self.S_log_diagonal)

    def get_M_diag(self) -> torch.Tensor:
        return torch.exp(self.log_M_diag)

    def get_M_inv_diag(self) -> torch.Tensor:
        return torch.exp(-self.log_M_diag)

    def get_alpha(self) -> torch.Tensor:
        if self.alpha_logits is None:
            raise RuntimeError("alpha is only defined for b_mode='diag_gate'")
        return torch.sigmoid(self.alpha_logits)

    def get_B_weight(self) -> torch.Tensor:
        if self.b_mode == "diag_gate":
            alpha = self.get_alpha()
            weight = torch.zeros(
                self.latent_dim,
                self.source_dim,
                device=alpha.device,
                dtype=alpha.dtype,
            )
            idx = torch.arange(self.latent_dim, device=alpha.device)
            weight[idx, idx] = alpha
            weight[idx, idx + self.latent_dim] = 1.0 - alpha
            return weight
        return self.B.weight

    def projected_source(self, z_f: torch.Tensor, z_m: torch.Tensor) -> torch.Tensor:
        if self.b_mode == "diag_gate":
            alpha = self.get_alpha().unsqueeze(0)
            return alpha * z_f + (1.0 - alpha) * z_m
        x = self._build_joint_source(z_f, z_m)
        return self.B(x)

    def get_log_potential(self, y: torch.Tensor) -> torch.Tensor:
        S = self.get_S()
        log_alpha = self.get_log_alpha()

        mix = Categorical(logits=log_alpha)
        comp = Independent(
            Normal(loc=self.r, scale=torch.sqrt(self.epsilon * S)),
            1,
        )
        gmm = MixtureSameFamily(mix, comp)
        return gmm.log_prob(y) + torch.logsumexp(log_alpha, dim=-1)

    def log_Z(
        self,
        z_f: torch.Tensor,
        z_m: torch.Tensor,
        a: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del a
        bx = self.projected_source(z_f, z_m)
        S = self.get_S()
        M_inv = self.get_M_inv_diag()
        log_alpha = self.get_log_alpha()

        total_var = self.epsilon * (S + M_inv.unsqueeze(0))
        diff = bx.unsqueeze(1) - self.r.unsqueeze(0)

        log_prob = -0.5 * (
            ((diff.square()) / total_var.unsqueeze(0)).sum(dim=-1)
            + torch.log(2.0 * math.pi * total_var).sum(dim=-1).unsqueeze(0)
        )
        return torch.logsumexp(log_alpha.unsqueeze(0) + log_prob, dim=-1)

    def loss(
        self,
        z_f: torch.Tensor,
        z_m: torch.Tensor,
        y: torch.Tensor,
        a: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return (self.log_Z(z_f, z_m, a) - self.get_log_potential(y)).mean()

    def b_deviation_penalty(self) -> torch.Tensor:
        return (self.get_B_weight() - self.B0_weight).square().sum()

    @torch.no_grad()
    def sample_child(
        self,
        z_f: torch.Tensor,
        z_m: torch.Tensor,
        a: Optional[torch.Tensor] = None,
        n_samples: int = 1,
    ) -> torch.Tensor:
        del a
        bx = self.projected_source(z_f, z_m)
        S = self.get_S()
        S_inv = 1.0 / S
        M = self.get_M_diag()
        M_inv = self.get_M_inv_diag()
        log_alpha = self.get_log_alpha()

        total_var = self.epsilon * (S + M_inv.unsqueeze(0))
        diff = bx.unsqueeze(1) - self.r.unsqueeze(0)
        mix_logits = log_alpha.unsqueeze(0) - 0.5 * (
            ((diff.square()) / total_var.unsqueeze(0)).sum(dim=-1)
            + torch.log(2.0 * math.pi * total_var).sum(dim=-1).unsqueeze(0)
        )

        sigma = 1.0 / (S_inv + M.unsqueeze(0))
        posterior_mean = sigma.unsqueeze(0) * (
            S_inv.unsqueeze(0) * self.r.unsqueeze(0)
            + M.unsqueeze(0).unsqueeze(0) * bx.unsqueeze(1)
        )
        cov_diag = self.epsilon * sigma.unsqueeze(0)

        mix = Categorical(logits=mix_logits)
        comp = Independent(
            Normal(loc=posterior_mean, scale=torch.sqrt(cov_diag)),
            1,
        )
        gmm = MixtureSameFamily(mix, comp)

        if n_samples == 1:
            return gmm.sample()
        samples = gmm.sample((n_samples,))
        return samples.permute(1, 0, 2).contiguous()
