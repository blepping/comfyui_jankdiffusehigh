from __future__ import annotations

from typing import TYPE_CHECKING

from .utils import ensure_model, sigma_to_float

if TYPE_CHECKING:
    import torch


class GuidedModel:
    def __init__(
        self,
        dh_sampler_object,
        guidance_sigmas: torch.Tensor,
        guidance_steps: int,
    ):
        self.dhso = dh_sampler_object
        self.allow_guidance = True
        self.force_guidance: None | int = None
        self.set_guidance_range(guidance_sigmas, guidance_steps)

    def set_guidance_range(self, guidance_sigmas, guidance_steps):
        if len(guidance_sigmas) >= 2:
            self.guidance_start_sigma = sigma_to_float(guidance_sigmas[0]) + 1e-05
            self.guidance_end_sigma = sigma_to_float(guidance_sigmas[-1]) + 1e-05
        else:
            self.guidance_start_sigma = None
            self.guidance_end_sigma = None
        self.guidance_sigmas_list = tuple(guidance_sigmas.detach().cpu().tolist())
        self.guidance_steps = guidance_steps

    def find_guidance_step_(self, sigma_float: float) -> None | int:
        return (
            next(
                (
                    idx
                    for idx, gsigma in enumerate(self.guidance_sigmas_list)
                    if gsigma <= sigma_float
                ),
                None,
            )
            if sigma_float <= self.guidance_start_sigma
            else 0
        )

    def get_guidance_step(self, sigma: torch.Tensor) -> None | int:
        sigma_float = sigma_to_float(sigma)
        if not self.allow_guidance:
            return None
        sigma_float = sigma_to_float(sigma)
        if self.force_guidance is None and (
            self.guidance_start_sigma is None or sigma_float < self.guidance_end_sigma
        ):
            return None
        if self.force_guidance is not None and self.force_guidance >= 0:
            return self.force_guidance
        step_idx = (
            next(
                (
                    idx
                    for idx, gsigma in enumerate(self.guidance_sigmas_list)
                    if gsigma <= sigma_float
                ),
                None,
            )
            if sigma_float <= self.guidance_start_sigma
            else 0
        )
        if step_idx is None:
            if not self.force_guidance:
                return None
            step_idx = max(0, self.guidance_steps - 1)
        return min(step_idx, self.guidance_steps - 1)

    def make_wrapper(self):
        guided_model = self

        class DiffuseHighModelWrapper:
            def __getattr__(self, k):
                try:
                    return getattr(guided_model, k)
                except AttributeError:
                    raise AttributeError(k) from None

            def __call__(self, *args: list, **kwargs: dict) -> torch.Tensor:
                return guided_model(*args, **kwargs)

        return DiffuseHighModelWrapper()

    def __call__(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        **extra_args: dict,
    ) -> torch.Tensor:
        dhso = self.dhso
        model = dhso.model
        dhso.seed_offset += 1
        ensure_model(model)
        guidance_step = self.get_guidance_step(sigma)
        sigma_offset = max(
            1e-05,
            1.0
            + (
                dhso.sigma_dishonesty_factor_guidance
                if guidance_step is not None
                else dhso.sigma_dishonesty_factor
            ),
        )
        denoised = model(
            x,
            sigma * sigma_offset if sigma_offset != 1 else sigma,
            **extra_args,
        )
        return (
            denoised
            if guidance_step is None
            else dhso.apply_guidance(guidance_step, denoised)
        )
