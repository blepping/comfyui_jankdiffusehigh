from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any

from comfy import latent_formats, samplers
from comfy.k_diffusion import sampling as kds
from comfy_extras import nodes_gits
from comfy_extras.nodes_align_your_steps import AlignYourStepsScheduler

from .utils import fallback

if TYPE_CHECKING:
    import torch


class Schedule:
    schedule_default_kwargs = {  # noqa: RUF012
        "alignyoursteps": {},
        "beta": {"alpha": 0.6, "beta": 0.6},
        "ddim_uniform": {},
        "exponential": {"sigma_min": -1, "sigma_max": -1},
        "gits": {"coeff": 1.2},
        "karras": {"sigma_min": -1, "sigma_max": -1, "rho": 7.0},
        "laplace": {"sigma_min": -1, "sigma_max": -1, "mu": 0.0, "beta": 0.5},
        "normal": {},
        "polyexponential": {"sigma_min": -1, "sigma_max": -1, "rho": 1.0},
        "sgm_uniform": {},
        "simple": {},
        "vp": {"beta_d": 19.9, "beta_min": 0.1, "eps_s": 0.001},
        "kl_optimal": {"sigma_min": -1, "sigma_max": -1},
    }

    def __init__(
        self,
        model_sampling: object,
        latent_format: object,
        schedule_name: str,
        steps: int,
        *,
        denoise: float = 1.0,
        **kwargs: dict,
    ):
        if denoise <= 0:
            raise ValueError("Bad denoise value: must be greater than 0")
        if denoise > 1:
            raise ValueError("Bad denoise value: must be less or equal to 1")
        self.denoise = denoise
        self.model_sampling = model_sampling
        if isinstance(latent_format, latent_formats.SD15):
            self.model_type = "SD1"
        elif isinstance(latent_format, latent_formats.SDXL):
            self.model_type = "SDXL"
        else:
            self.model_type = None
        self.steps = steps
        self.total_steps = int(steps / denoise)
        self.schedule_name = schedule_name
        self.schedule_kwargs = self.get_schedule_kwargs(schedule_name, kwargs)
        self._sigmas: torch.Tensor | None = None

    def get_schedule_kwargs(
        self,
        schedule_name: str,
        schedule_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, int | float | str]:
        schedule_name = schedule_name.lower().strip()
        if schedule_name not in self._make_sigmas_handlers:
            errstr = f"Unknown schedule: {schedule_name}"
            raise ValueError(errstr)
        schedule_default_kwargs = self.schedule_default_kwargs.get(schedule_name, {})
        schedule_kwargs = fallback(schedule_kwargs, {})
        bad_keys = ",".join({
            k for k in schedule_kwargs if k not in schedule_default_kwargs
        })
        if bad_keys:
            errstr = (
                f"Schedule {schedule_name} passed unsupported arguments: {bad_keys}"
            )
            raise ValueError(errstr)
        schedule_kwargs = schedule_default_kwargs | schedule_kwargs
        sigma_min = schedule_kwargs.get("sigma_min")
        sigma_max = schedule_kwargs.get("sigma_max")
        if sigma_min is not None and sigma_min < 0:
            schedule_kwargs["sigma_min"] = (
                self.model_sampling.sigma_min.detach().cpu().item()
            )
        if sigma_max is not None and sigma_max < 0:
            schedule_kwargs["sigma_max"] = (
                self.model_sampling.sigma_max.detach().cpu().item()
            )
        if schedule_name == "alignyoursteps":
            model_type = schedule_kwargs.get("model_type")
            if model_type is None:
                if self.model_type is None:
                    raise RuntimeError(
                        "alignyoursteps schedule specified with no model_type and we can't guess the model type",
                    )
                schedule_kwargs["model_type"] = self.model_type
            elif model_type not in {"SD1", "SDXL"}:
                raise ValueError(
                    "alignyoursteps model_type must be one of SD15 or SDXL",
                )
        return schedule_kwargs

    @property
    def sigmas(self) -> torch.Tensor:
        if self._sigmas is None:
            self._sigmas = self.make_sigmas()[-(self.steps + 1) :]
        return self._sigmas.detach().clone()

    def make_sigmas_alignyoursteps(self) -> torch.Tensor:
        return AlignYourStepsScheduler().get_sigmas(
            model_type=self.schedule_kwargs["model_type"],
            steps=self.total_steps,
            denoise=1.0,
        )[0]

    def make_sigmas_gits(self) -> torch.Tensor:
        coeff = round(self.schedule_kwargs["coeff"], 2)
        if coeff not in nodes_gits.NOISE_LEVELS:
            raise ValueError(
                "GITS scheduler only supports coeff values between 0.8 and 1.5 (inclusive) in increments of 0.05",
            )
        return nodes_gits.GITSScheduler().get_sigmas(
            coeff=self.schedule_kwargs["coeff"],
            steps=self.total_steps,
            denoise=1.0,
        )[0]

    def make_sigmas_no_model_sampling(self, f: callable) -> torch.Tensor:
        return f(n=self.total_steps, **self.schedule_kwargs)

    def make_sigmas_model_sampling(self, f: callable) -> torch.Tensor:
        return f(self.model_sampling, self.total_steps, **self.schedule_kwargs)

    _make_sigmas_handlers = {  # noqa: RUF012
        "alignyoursteps": make_sigmas_alignyoursteps,
        "beta": partial(make_sigmas_model_sampling, f=samplers.beta_scheduler),
        "ddim_uniform": partial(
            make_sigmas_model_sampling,
            f=samplers.ddim_scheduler,
        ),
        "exponential": partial(
            make_sigmas_no_model_sampling,
            f=kds.get_sigmas_exponential,
        ),
        "gits": make_sigmas_gits,
        "karras": partial(make_sigmas_no_model_sampling, f=kds.get_sigmas_karras),
        "laplace": partial(make_sigmas_no_model_sampling, f=kds.get_sigmas_laplace),
        "normal": partial(make_sigmas_model_sampling, f=samplers.normal_scheduler),
        "polyexponential": partial(
            make_sigmas_no_model_sampling,
            f=kds.get_sigmas_polyexponential,
        ),
        "sgm_uniform": partial(
            make_sigmas_model_sampling,
            f=partial(samplers.normal_scheduler, sgm=True),
        ),
        "simple": partial(
            make_sigmas_model_sampling,
            f=samplers.simple_scheduler,
        ),
        "vp": partial(
            make_sigmas_no_model_sampling,
            f=kds.get_sigmas_vp,
        ),
    }
    if hasattr(samplers, "kl_optimal_scheduler"):
        _make_sigmas_handlers["kl_optimal"] = partial(
            make_sigmas_no_model_sampling,
            f=samplers.kl_optimal_scheduler,
        )

    def make_sigmas(self) -> torch.Tensor:
        handler = self._make_sigmas_handlers.get(self.schedule_name)
        if handler is None:
            errstr = f"Internal error, please bug report: could not get handler for schedule {self.schedule_name}"
            raise RuntimeError(errstr)
        return handler(self)
