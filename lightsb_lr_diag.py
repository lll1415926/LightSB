from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class LowRankDiagLightSBPotential(nn.Module):
    """
    Unified LightSB potential using the original rank-0 formula as the reference.

    We define, for each component k,

        S_k = Diag(exp(delta_k)) + U_k U_k^T

    and use the same formulas for all ranks:

        z_theta(y)
        = sum_k exp(log_alpha_k) * N(y ; m_k, epsilon * S_k)

        C_theta(x)
        = sum_k exp(log_alpha_k)
              * exp((x^T S_k x + 2 x^T m_k) / (2 epsilon))

        pi_theta(y | x, k)
        = N(y ; m_k + S_k x, epsilon * S_k)

    Therefore:
        - r = 0 gives the original diagonal LightSB model exactly.
        - r > 0 gives a low-rank extension of the same model.

    Notes
    -----
    1. `alpha_scale` is used as the diffusion parameter epsilon for compatibility
       with the original LightSB API.
    2. `sigma` and `inv_sigma2` are kept only for backward compatibility with
       older code paths, but are not used by the unified formulas below.
    3. `eps` is used only as numerical jitter when needed; it is NOT part of
       the model definition.
    """

    def __init__(
        self,
        d: int,
        K: int,
        r: int,
        sigma: float = 3.0,
        eps: float = 1e-4,
        init_delta: float = -2.0,
        init_mean_scale: float = 0.2,
        alpha_scale: float = 1.0,
    ) -> None:
        super().__init__()
        assert d > 0 and K > 0 and r >= 0
        assert sigma > 0.0
        assert eps >= 0.0
        assert alpha_scale > 0.0

        self.d = d
        self.K = K
        self.r = r
        self.eps = float(eps)

        # kept only for backward compatibility with external code
        sigma = float(sigma)
        self.register_buffer("sigma", torch.tensor(sigma))
        self.register_buffer("inv_sigma2", torch.tensor(1.0 / (sigma * sigma)))

        # In the unified model, alpha_scale plays the role of epsilon.
        self.register_buffer("alpha_scale", torch.tensor(float(alpha_scale)))

        # Original LightSB-style storage:
        # log_alpha_raw = epsilon * log_alpha
        self.log_alpha_raw = nn.Parameter(self.alpha_scale * torch.log(torch.ones(K) / K))

        self.m = nn.Parameter(torch.randn(K, d) * init_mean_scale)   # [K, d]
        self.delta = nn.Parameter(torch.full((K, d), init_delta))    # [K, d]

        # Important: do NOT initialize U at exactly zero, or it may stay stuck.
        if r > 0:
            self.U = nn.Parameter(1e-3 * torch.randn(K, d, r))       # [K, d, r]
        else:
            self.U = nn.Parameter(torch.zeros(K, d, 0), requires_grad=False)

    # ============================================================
    # Initialization helpers
    # ============================================================
    @torch.no_grad()
    def initialize_means_from_samples(self, y_samples: torch.Tensor) -> None:
        """
        Initialize mixture centers from target samples.

        Args:
            y_samples: [N, d]
        """
        n = y_samples.shape[0]
        if n < self.K:
            idx = torch.randint(0, n, (self.K,), device=y_samples.device)
        else:
            idx = torch.randperm(n, device=y_samples.device)[: self.K]
        self.m.copy_(y_samples[idx])

    @torch.no_grad()
    def init_r_by_samples(self, samples: torch.Tensor) -> None:
        """Compatibility alias with original LightSB API."""
        self.initialize_means_from_samples(samples)

    @torch.no_grad()
    def recenter_alpha(self) -> None:
        """
        Remove the global shift freedom in alpha_raw.
        """
        self.log_alpha_raw.sub_(self.log_alpha_raw.mean())

    def get_log_alpha(self) -> torch.Tensor:
        """
        Return centered/scaled log-alpha.

        Original LightSB-style storage:
            log_alpha_raw = epsilon * log_alpha
        so
            log_alpha = log_alpha_raw / epsilon
        """
        return (self.log_alpha_raw - self.log_alpha_raw.mean()) / self.alpha_scale

    def get_S(self) -> torch.Tensor:
        """
        Diagonal part of S_k: diag(exp(delta_k)).
        """
        return torch.exp(self.delta)

    def get_r(self) -> torch.Tensor:
        """
        Compatibility API with original LightSB (`r` centers).
        """
        return self.m

    # ============================================================
    # Low-rank S_k helpers
    # S_k = D_k + U_k U_k^T,  D_k = diag(exp(delta_k))
    # ============================================================
    def _prepare_S_cache(self) -> Dict[str, torch.Tensor]:
        """
        Cache useful quantities for:
            S_k = Diag(exp(delta_k)) + U_k U_k^T

        Returns:
            dict with:
                S_diag: [K, d]
                S_inv_diag: [K, d]
                L: [K, r, r] or None
                logdet_S: [K]
        """
        S_diag = self.get_S()                 # [K, d]
        S_inv_diag = 1.0 / S_diag             # [K, d]

        if self.r == 0:
            L = None
            logdet_S = torch.log(S_diag).sum(dim=-1)
        else:
            V = S_inv_diag.unsqueeze(-1) * self.U
            eye_r = torch.eye(self.r, device=self.U.device, dtype=self.U.dtype)

            # M_k = I + U_k^T D_k^{-1} U_k
            M = eye_r.unsqueeze(0) + self.U.transpose(-1, -2) @ V   # [K, r, r]
            L, info = torch.linalg.cholesky_ex(M)

            if torch.any(info != 0):
                jitter = max(self.eps, 1e-8)
                M = M + jitter * eye_r.unsqueeze(0)
                L = torch.linalg.cholesky(M)

            logdet_S = (
                torch.log(S_diag).sum(dim=-1)
                + 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(dim=-1)
            )

        return {
            "S_diag": S_diag,
            "S_inv_diag": S_inv_diag,
            "L": L,
            "logdet_S": logdet_S,
        }

    def _solve_S(
        self,
        v: torch.Tensor,
        cache: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute S^{-1} v for v shaped [B, K, d] using Woodbury.

        Args:
            v: [B, K, d]

        Returns:
            [B, K, d]
        """
        S_inv_diag = cache["S_inv_diag"]          # [K, d]
        u1 = v * S_inv_diag.unsqueeze(0)          # [B, K, d]

        if self.r == 0:
            return u1

        L = cache["L"]                            # [K, r, r]
        U = self.U                                # [K, d, r]

        t = torch.einsum("kdr,bkd->bkr", U, u1)   # [B, K, r]
        L_b = L.unsqueeze(0).expand(v.shape[0], -1, -1, -1)
        s = torch.cholesky_solve(t.unsqueeze(-1), L_b).squeeze(-1)   # [B, K, r]

        Us = torch.einsum("kdr,bkr->bkd", U, s)   # [B, K, d]
        u2 = Us * S_inv_diag.unsqueeze(0)         # [B, K, d]

        return u1 - u2

    def _matvec_S(
        self,
        x: torch.Tensor,
        cache: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute S_k x for all k.

        Args:
            x: [B, d]

        Returns:
            [B, K, d]
        """
        out = x.unsqueeze(1) * cache["S_diag"].unsqueeze(0)   # D x

        if self.r == 0:
            return out

        Utx = torch.einsum("kdr,bd->bkr", self.U, x)          # [B, K, r]
        return out + torch.einsum("kdr,bkr->bkd", self.U, Utx)

    def _quad_x_S_x(
        self,
        x: torch.Tensor,
        cache: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute x^T S_k x for all k.

        Args:
            x: [B, d]

        Returns:
            [B, K]
        """
        out = (x.unsqueeze(1).square() * cache["S_diag"].unsqueeze(0)).sum(dim=-1)

        if self.r == 0:
            return out

        Utx = torch.einsum("kdr,bd->bkr", self.U, x)
        return out + (Utx * Utx).sum(dim=-1)

    # ============================================================
    # log z_theta(y)
    # ============================================================
    def get_log_potential(
        self,
        y: torch.Tensor,
        cache: Dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """
        Unified LightSB endpoint potential:

            log z_theta(y)
            = logsumexp_k [
                log_alpha_k
                + log N(y ; m_k, epsilon * S_k)
            ]

        Args:
            y: [B, d]

        Returns:
            [B]
        """
        if cache is None:
            cache = self._prepare_S_cache()
        epsilon = self.alpha_scale.to(dtype=y.dtype)

        diff = y.unsqueeze(1) - self.m.unsqueeze(0)      # [B, K, d]
        S_inv_diff = self._solve_S(diff, cache)          # [B, K, d]
        quad = (diff * S_inv_diff).sum(dim=-1)           # [B, K]

        log_norm = 0.5 * (
            self.d * (math.log(2.0 * math.pi) + torch.log(epsilon))
            + cache["logdet_S"]
        )                                                # [K]

        logits = (
            self.get_log_alpha().unsqueeze(0)
            - 0.5 * quad / epsilon
            - log_norm.unsqueeze(0)
        )
        return torch.logsumexp(logits, dim=-1)

    def potential_log_z(self, y: torch.Tensor) -> torch.Tensor:
        """Compatibility alias."""
        return self.get_log_potential(y)

    # ============================================================
    # Conditional component parameters for pi(y|x)
    # ============================================================
    def compute_component_params(
        self,
        x: torch.Tensor,
        cache: Dict[str, torch.Tensor] | None = None,
    ) -> Dict[str, torch.Tensor]:
        """
        For the unified LightSB model:

            log w_k(x)
            = log_alpha_k + (x^T S_k x + 2 x^T m_k) / (2 epsilon)

            mu_k(x)
            = m_k + S_k x

            y | x, k ~ N(mu_k(x), epsilon * S_k)

        Args:
            x: [B, d]

        Returns:
            dict with:
                mu: [B, K, d]
                log_w: [B, K]
                cache: S-side cache
        """
        if cache is None:
            cache = self._prepare_S_cache()
        epsilon = self.alpha_scale.to(dtype=x.dtype)

        Sx = self._matvec_S(x, cache)                    # [B, K, d]
        mu = self.m.unsqueeze(0) + Sx                    # [B, K, d]

        x_S_x = self._quad_x_S_x(x, cache)               # [B, K]
        x_m = torch.einsum("bd,kd->bk", x, self.m)       # [B, K]

        log_w = (
            self.get_log_alpha().unsqueeze(0)
            + (x_S_x + 2.0 * x_m) / (2.0 * epsilon)
        )

        return {
            "mu": mu,
            "log_w": log_w,
            "cache": cache,
        }

    # ============================================================
    # log C_theta(x)
    # ============================================================
    def get_log_C(
        self,
        x: torch.Tensor,
        cache: Dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """
        Unified LightSB log C:

            log C_theta(x)
            = logsumexp_k [
                log_alpha_k
                + (x^T S_k x + 2 x^T m_k) / (2 epsilon)
            ]

        Args:
            x: [B, d]

        Returns:
            [B]
        """
        params = self.compute_component_params(x, cache=cache)
        return torch.logsumexp(params["log_w"], dim=-1)

    def log_Z(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compatibility alias.

        In the unified/original-LightSB formulation, this matches get_log_C(x).
        """
        return self.get_log_C(x)

    # ============================================================
    # Conditional mixture weights
    # ============================================================
    def conditional_mixture_weights(self, x: torch.Tensor) -> torch.Tensor:
        params = self.compute_component_params(x)
        return F.softmax(params["log_w"], dim=-1)        # [B, K]

    # ============================================================
    # Conditional sampling from pi_theta(y|x)
    # ============================================================
    @torch.no_grad()
    def conditional_sample(self, x: torch.Tensor, n_samples: int = 1) -> torch.Tensor:
        """
        Sample from:

            pi_theta(y|x)
            = sum_k w_k(x) N(m_k + S_k x, epsilon * S_k)

        using the factorization:

            S_k = Diag(S_diag,k) + U_k U_k^T

        so a sample from N(0, S_k) can be generated as:

            sqrt(S_diag,k) * xi + U_k eta,

        where xi ~ N(0, I_d), eta ~ N(0, I_r).

        Args:
            x: [B, d]
            n_samples: int

        Returns:
            y: [B, n_samples, d]
        """
        params = self.compute_component_params(x)
        mu = params["mu"]               # [B, K, d]
        log_w = params["log_w"]         # [B, K]
        cache = params["cache"]

        B, _, d = mu.shape

        probs = F.softmax(log_w, dim=-1)                              # [B, K]
        comp_idx = torch.multinomial(probs, n_samples, replacement=True)  # [B, n_samples]

        gather_idx = comp_idx.unsqueeze(-1).expand(-1, -1, d)
        mu_sel = torch.gather(mu, 1, gather_idx)                      # [B, n_samples, d]

        # exact sampling from epsilon * S_k
        S_diag_sel = cache["S_diag"][comp_idx]                        # [B, n_samples, d]
        noise = torch.sqrt(S_diag_sel) * torch.randn(
            B, n_samples, d, device=x.device, dtype=x.dtype
        )

        if self.r > 0:
            U_sel = self.U[comp_idx]                                  # [B, n_samples, d, r]
            eta = torch.randn(B, n_samples, self.r, device=x.device, dtype=x.dtype)
            noise = noise + torch.einsum("bndr,bnr->bnd", U_sel, eta)

        return mu_sel + torch.sqrt(self.alpha_scale).to(dtype=x.dtype) * noise

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compatibility forward: sample one mapped latent like original LightSB."""
        return self.conditional_sample(x, n_samples=1)[:, 0, :]

    # ============================================================
    # Regularization
    # ============================================================
    def regularization(
        self,
        lambda_diag: float = 0.0,
        lambda_U: float = 1e-4,
        lambda_m: float = 0.0,
    ) -> torch.Tensor:
        """
        Optional L2 regularization on the shape and center parameters.
        """
        reg = torch.zeros((), device=self.m.device, dtype=self.m.dtype)

        if lambda_diag > 0:
            reg = reg + lambda_diag * (self.delta ** 2).mean()
        if lambda_m > 0:
            reg = reg + lambda_m * (self.m ** 2).mean()
        if self.r > 0 and lambda_U > 0:
            reg = reg + lambda_U * (self.U ** 2).mean()

        return reg

    # ============================================================
    # LightSB objective
    # ============================================================
    def training_loss(self, x_src: torch.Tensor, y_tgt: torch.Tensor) -> torch.Tensor:
        """
        Unified LightSB objective:

            E_{x~mu}[log C_theta(x)] - E_{y~nu}[log z_theta(y)]
        """
        cache = self._prepare_S_cache()
        return (-self.get_log_potential(y_tgt, cache=cache) + self.get_log_C(x_src, cache=cache)).mean()


# =====================================================================
# Toy data
# =====================================================================
def make_toy(d: int, device: torch.device, seed: int = 0):
    g = torch.Generator(device="cpu").manual_seed(seed)

    src_means = torch.randn(3, d, generator=g) * 2.5   # [3, d]

    shift = torch.zeros(d)
    shift[: min(3, d)] = torch.tensor([3.0, -2.5, 2.0][: min(3, d)])
    tgt_means = (src_means + shift)[torch.tensor([1, 2, 0])]   # permuted target modes

    src_means = src_means.to(device)
    tgt_means = tgt_means.to(device)

    src_std = 0.6
    tgt_std = 0.5

    def sample_src(n: int) -> torch.Tensor:
        idx = torch.randint(0, 3, (n,), device=device)
        return src_means[idx] + src_std * torch.randn(n, d, device=device)

    def sample_tgt(n: int) -> torch.Tensor:
        idx = torch.randint(0, 3, (n,), device=device)
        return tgt_means[idx] + tgt_std * torch.randn(n, d, device=device)

    return sample_src, sample_tgt, src_means, tgt_means


# =====================================================================
# Training demo
# =====================================================================
def train_demo() -> None:
    torch.manual_seed(0)
    device = torch.device("cpu")   # change to "cuda" if available

    d, K, r = 8, 16, 2
    epsilon = 1.0   # unified LightSB diffusion parameter

    model = LowRankDiagLightSBPotential(
        d=d,
        K=K,
        r=r,
        eps=1e-6,
        init_delta=-2.0,
        alpha_scale=epsilon,
    ).to(device)

    sample_src, sample_tgt, src_means, tgt_means = make_toy(d, device)

    with torch.no_grad():
        y_init = sample_tgt(K * 8)
        model.initialize_means_from_samples(y_init)
        model.log_alpha_raw.zero_()
        model.recenter_alpha()

    opt = torch.optim.Adam(
        [
            {"params": [model.log_alpha_raw, model.m], "lr": 1e-3},
            {"params": [model.delta, model.U], "lr": 3e-4},
        ]
    )

    batch = 512
    n_steps = 400

    print(f"training on d={d}, K={K}, r={r}, epsilon={epsilon}")
    print("step | objective | reg | total")
    print("-----+-----------+-----------+-----------")

    for step in range(n_steps):
        x = sample_src(batch)
        y = sample_tgt(batch)

        objective = model.training_loss(x, y)
        # reg = model.regularization(lambda_U=1e-4)
        reg = 0.0
        loss = objective + reg

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        with torch.no_grad():
            model.recenter_alpha()

        if step % 50 == 0 or step == n_steps - 1:
            print(
                f"{step:4d} | {objective.item():+9.4f} | "
                f"{reg.item():.4e} | {loss.item():+9.4f}"
            )

    # ------------------------------------------------------------
    # Probe conditional samples
    # ------------------------------------------------------------
    with torch.no_grad():
        x_probe = src_means                           # [3, d]
        weights = model.conditional_mixture_weights(x_probe)
        y_samples = model.conditional_sample(x_probe, n_samples=128)   # [3, 128, d]
        y_mean = y_samples.mean(dim=1)               # [3, d]

        print("\nProbe source centers:")
        print(src_means)

        print("\nMean of conditional samples:")
        print(y_mean)

        print("\nTarget centers:")
        print(tgt_means)

        print("\nConditional mixture weight row sums:")
        print(weights.sum(dim=-1))

        log_z = model.potential_log_z(y_samples.reshape(-1, d))
        log_Z = model.log_Z(x_probe)
        print("\nSanity check:")
        print("log_z finite:", torch.isfinite(log_z).all().item())
        print("log_Z finite:", torch.isfinite(log_Z).all().item())


if __name__ == "__main__":
    train_demo()
