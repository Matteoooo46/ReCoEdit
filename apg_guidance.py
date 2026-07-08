"""
Standalone APG (Adaptive Projected Guidance) module for diffsynth pipelines.

APG paper: "Eliminating Oversaturation and Artifacts of High Guidance Scales
in Diffusion Models" (https://huggingface.co/papers/2410.02416)

This module monkey-patches a diffsynth pipeline instance at runtime to replace
standard CFG with APG, without modifying any library files.

=== Key design ===
- Qwen-Image-Edit uses a Flow Matching scheduler → model output is velocity v.
- x0 = latents - sigma * v   (flow matching identity)
- APG operates in x0 space:
    diff = x0_cond - x0_uncond
    parallel = proj(diff, x0_cond)
    orthogonal = diff - parallel
    apg_update = orthogonal + eta * parallel
    x0_guided = x0_cond + (cfg_scale - 1) * apg_update
    v_guided = (latents - x0_guided) / sigma
- Reverse momentum (paper eq.): running = diff + beta * running  (beta < 0)

Usage:
    from apg_guidance import patch_pipeline_for_apg

    pipe = QwenImagePipeline.from_pretrained(...)
    patch_pipeline_for_apg(pipe, eta=0.0, beta=-0.5)

    # Now pipe() calls use APG instead of CFG
    result = pipe(prompt=..., cfg_scale=4.0, ...)
"""

import torch
from typing import Optional


class ReverseMomentumBuffer:
    """
    Reverse momentum buffer from APG paper.
    Update rule: running = diff + beta * running
    where beta is negative (e.g., -0.5, -0.75).
    """

    def __init__(self, beta: float):
        self.beta = beta
        self.running = 0

    def update(self, diff: torch.Tensor) -> torch.Tensor:
        self.running = diff + self.beta * self.running
        return self.running

    def reset(self):
        self.running = 0


def _get_sigma(pipe, timestep: torch.Tensor) -> torch.Tensor:
    """
    Get sigma for the current timestep from the FlowMatchScheduler.
    Returns a scalar tensor on the same device as timestep.
    """
    scheduler = pipe.scheduler
    timestep_cpu = timestep.cpu()
    timestep_id = torch.argmin((scheduler.timesteps - timestep_cpu).abs())
    sigma = scheduler.sigmas[timestep_id].to(dtype=timestep.dtype, device=timestep.device)
    return sigma


def _apg_x0_guidance(
    pred_cond: torch.Tensor,        # velocity v_cond
    pred_uncond: torch.Tensor,      # velocity v_uncond
    latents: torch.Tensor,          # current noisy latents x_t
    sigma: torch.Tensor,            # current noise level
    cfg_scale: float,
    eta: float,
    momentum_buffer: Optional[ReverseMomentumBuffer],
    norm_threshold: Optional[float],
) -> torch.Tensor:
    """
    APG in x0 space, following the paper formulation.

    Args:
        pred_cond: conditional velocity prediction
        pred_uncond: unconditional velocity prediction
        latents: current noisy latents
        sigma: current noise level
        cfg_scale: guidance scale (paper's 's')
        eta: parallel component weight (eta=0 removes all parallel, recommended default)
        momentum_buffer: reverse momentum buffer (None = disabled)
        norm_threshold: norm clipping threshold (None = disabled)

    Returns:
        pred_guided: APG-guided velocity prediction (ready for scheduler step)
    """
    # Safety: fall back to standard CFG in velocity space if sigma ≈ 0
    # (happens at the very last denoising step)
    if sigma.item() < 1e-8:
        diff = pred_cond - pred_uncond
        return pred_uncond + cfg_scale * diff

    # Convert velocity to x0:  x0 = latents - sigma * v
    x0_cond = latents - sigma * pred_cond
    x0_uncond = latents - sigma * pred_uncond

    # diff in x0 space
    diff = x0_cond - x0_uncond

    # Reverse momentum (paper: running = diff + beta * running, beta < 0)
    if momentum_buffer is not None:
        diff = momentum_buffer.update(diff)

    # Optional norm clipping
    if norm_threshold is not None and norm_threshold > 0:
        dim = [-i for i in range(1, len(diff.shape))]
        ones = torch.ones_like(diff)
        diff_norm = diff.norm(p=2, dim=dim, keepdim=True)
        scale_factor = torch.minimum(ones, norm_threshold / diff_norm)
        diff = diff * scale_factor

    # Project diff onto x0_cond direction (double precision for numerical stability)
    dim = [-i for i in range(1, len(diff.shape))]
    v0, v1 = diff.double(), x0_cond.double()
    v1_norm = torch.nn.functional.normalize(v1, dim=dim)
    v0_parallel = (v0 * v1_norm).sum(dim=dim, keepdim=True) * v1_norm
    v0_orthogonal = v0 - v0_parallel
    parallel = v0_parallel.type_as(diff)
    orthogonal = v0_orthogonal.type_as(diff)

    # APG update: push mainly in orthogonal direction
    apg_update = orthogonal + eta * parallel

    # x0_guided = x0_cond + (cfg_scale - 1) * apg_update
    # (Standard CFG in x0 space would be: x0_cond + (cfg_scale-1) * diff)
    x0_guided = x0_cond + (cfg_scale - 1.0) * apg_update

    # Convert back to velocity:  v = (latents - x0) / sigma
    pred_guided = (latents - x0_guided) / sigma

    return pred_guided


def patch_pipeline_for_apg(
    pipe,
    eta: float = 0.0,
    beta: Optional[float] = -0.5,
    norm_threshold: Optional[float] = None,
):
    """
    Replace the pipeline's cfg_guided_model_fn with an APG version at runtime.

    The patched function:
    1. Converts model velocity output to x0 via x0 = latents - sigma * v
    2. Projects diff = x0_cond - x0_uncond onto x0_cond
    3. Separates parallel / orthogonal components
    4. apg_update = orthogonal + eta * parallel
    5. x0_guided = x0_cond + (cfg_scale - 1) * apg_update
    6. Converts back: v_guided = (latents - x0_guided) / sigma

    Args:
        pipe: A diffsynth BasePipeline instance (e.g., QwenImagePipeline).
        eta: Parallel component weight.
             0.0 = complete suppression of parallel (most aggressive, recommended).
             0.25 = mild suppression.
             1.0 = degenerates to standard CFG (no APG benefit).
        beta: Reverse momentum coefficient (paper eq.: running = diff + beta * running).
              Negative values: -0.5 (mild), -0.75 (aggressive).
              None = disable momentum.
        norm_threshold: Norm clipping threshold. None or 0.0 = disabled.
    """
    # Save APG config on the pipe instance
    pipe._apg_eta = eta
    pipe._apg_beta = beta
    pipe._apg_norm_threshold = norm_threshold
    pipe._apg_momentum_buffer = None

    # Keep reference to original CFG method for potential fallback
    pipe._original_cfg_guided_model_fn = pipe.cfg_guided_model_fn

    def apg_guided_model_fn(model_fn, cfg_scale, inputs_shared, inputs_posi, inputs_nega, **inputs_others):
        # Reset momentum buffer at the start of each new pipe() call
        progress_id = inputs_others.get("progress_id", None)
        if progress_id is not None and progress_id == 0:
            pipe._apg_momentum_buffer = None

        # Handle positive-only LoRA (preserved from original CFG logic)
        if inputs_shared.get("positive_only_lora", None) is not None:
            pipe.clear_lora(verbose=0)
            pipe.load_lora(pipe.dit, state_dict=inputs_shared["positive_only_lora"], verbose=0)

        noise_pred_posi = model_fn(**inputs_posi, **inputs_shared, **inputs_others)

        if cfg_scale != 1.0:
            if inputs_shared.get("positive_only_lora", None) is not None:
                pipe.clear_lora(verbose=0)
            noise_pred_nega = model_fn(**inputs_nega, **inputs_shared, **inputs_others)

            # Get current sigma and latents
            timestep = inputs_others["timestep"]
            sigma = _get_sigma(pipe, timestep)
            latents = inputs_shared["latents"]

            # Initialize momentum buffer on first step of each denoising loop
            if pipe._apg_beta is not None and pipe._apg_momentum_buffer is None:
                pipe._apg_momentum_buffer = ReverseMomentumBuffer(pipe._apg_beta)

            if isinstance(noise_pred_posi, tuple):
                noise_pred = tuple(
                    _apg_x0_guidance(
                        n_posi, n_nega, latents, sigma, cfg_scale,
                        eta=pipe._apg_eta,
                        momentum_buffer=pipe._apg_momentum_buffer,
                        norm_threshold=pipe._apg_norm_threshold,
                    )
                    for n_posi, n_nega in zip(noise_pred_posi, noise_pred_nega)
                )
            else:
                noise_pred = _apg_x0_guidance(
                    noise_pred_posi, noise_pred_nega, latents, sigma, cfg_scale,
                    eta=pipe._apg_eta,
                    momentum_buffer=pipe._apg_momentum_buffer,
                    norm_threshold=pipe._apg_norm_threshold,
                )
        else:
            noise_pred = noise_pred_posi

        return noise_pred

    # Replace the method on the instance
    pipe.cfg_guided_model_fn = apg_guided_model_fn

    print(f"[APG] Pipeline patched (x0-space projection): eta={eta}, beta={beta}, "
          f"norm_threshold={norm_threshold}")
    print(f"[APG] Formula: x0_cond + (cfg_scale-1) * (orthogonal + {eta}*parallel)")
    print(f"[APG] Reverse momentum: running = diff + ({beta}) * running")
