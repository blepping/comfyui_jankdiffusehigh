# ComfyUI jank DiffuseHigh
Janky implementation of [DiffuseHigh](https://github.com/yhyun225/DiffuseHigh/) for ComfyUI.

Facilitates generating directly to resolutions higher than the model was trained for, similar to Kohya Deep Shrink, HiDiffusion, etc.

This is a best-effort attempt at implementation. If you experience poor results, please don't let it reflect on the official version. There's a good chance it's something I did wrong.

## Current Status

Alpha - early implementation. Many rough edges but the core functionality is there. Mainly targetted at advanced users who can deal with some weird stuff and frequent workflow-breaking changes.

See the [changelog](changelog.md) for recent user visible changes.

**Known issues/caveats**

* There will be frequent workflow-breaking changes for a while yet.
* Progress and previews are pretty wonky (you can look at the log for some progress information).
* Using VAE or upscale models may result in the main model getting repeatedly unloaded/reloaded. Try using `latent` as the `guidance_mode`. If you actually have enough VRAM, maybe disabling smart memory (via ComfyUI commandline parameter) would help.
* Currently only tested on SD15 and SDXL, may not work with models like Flux. (Not much testing in general as of yet.)
* Brownian noise-based (AKA SDE) samplers may be a bit weird here, there is a workaround in place but it might not be enough. Also don't use with prompt-control's PCSplitSampling stuff.

## Description

The DiffuseHigh approach is similar to an iterative upscale/run some more steps at low denoise approach with a twist: it mixes in guidance from a reference image for a number of steps at the beginning of each sampling iteration. The guidance is derived from the low frequency parts of the reference and also gets sharpened first to increase detail.

My approach implements it as a sampler which means it's _mostly_ model-agnostic and avoids some common issues with alternative approaches like Deep Shrink and HiDiffusion that require model patches. It's also possible to generate a low or mid-resolution image to see if you like the results and then increase the number of iterations to get a similar result where with Deep Shrink/HiDiffusion type effects enabling/disabling the patch will effectively change the seed.

The main disadvantage compared to the alternatives I mentioned is that it is relatively slow and VRAM hungry since it requires multiple iterations at high res while Deep Shrink/HiDiffusion actually speed up generation while the scaling effect is active.

## Nodes

### `DiffuseHighSampler`

#### Inputs

* `highres_sigmas`: Optional: Sigmas used for everything other than the initial reference image. **Note**: Should be around 0.3-0.5 denoise. You won't get good results connecting something like `KarrasScheduler` here without splitting the sigmas. If not specified, will use the last 15 steps of a 50 step Karras schedule like the official implementation.
* `sampler`: Optional: Default sampler used for steps. If not specified the sampler will default to non-ancestral Euler.
* `reference_image_opt`: Optional: Image used for the initial pass. If not connected, a low-res initial reference will be generated using the schedule from the normal sigmas (i.e. the sigmas attached to `SamplerCustom` or whatever actual sampler node you're using).
* `guidance_sampler_opt`: Optional: Sampler used for guidance steps. If not specified, will fallback to the base sampler. Note: The sampler is called on individual steps, samplers that keep history will not work well here.
* `reference_sampler_opt`: Optional: Sampler used to generate the initial low-resolution reference. Only used if reference_image_opt is not connected.
* `vae_opt`: Optional when vae_mode is set to `taesd`, otherwise this is the VAE that will be used for encoding/decoding images. If using TAESD, you will require the corresponding encoder (which I believe ComfyUI does not install by default). TAESD models go in `models/vae_approx`, you can find them here: https://github.com/madebyollin/taesd
* `upscale_model_opt`: Optional: Model used for upscaling. When not attached, simple image scaling will be used. Regardless, the image will be scaled to match the size expected based on `scale_factor`. For example, if you use scale_factor 2 and a 4x upscale model, the image will get scaled down after the upscale model runs.
* `yaml_parameters`: Optional: Allows specifying custom parameters via YAML. You can also override any of the normal parameters by key. This input can be converted into a multiline text widget. Note: When specifying paramaters this way, there is very little error checking. See below for some information about advanced parameters.

#### Parameters

* `guidance_steps`: Number of guidance steps after an upscale.
* `guidance_mode`: The original implementation uses `image` guidance. This requires a VAE encode/decode per guidance step. Alternatively, you can try using guidance via the latent instead which is much faster. Personally I recommend setting this to `latent`.
* `guidance_factor`: Mix factor used on guidance steps. 1.0 means use 100% DiffuseHigh guidance for those steps (like the original implementation).
* `fadeout_factor`: Can be enabled to fade out guidance_factor. For example, if `guidance_factor` is 1 and guidance_steps is 4 then `fadeout_factor` would use these `guidance_factor`s for the guidance steps: 1.00, 0.75, 0.50, 0.25
* `scale_factor`: Upscale factor per iteration. The scaled size will be rounded to increments of 64 by default (can be adjusted via YAML parameters).
* `renoise_factor`: Strength of noise added at the start of each iteration. The default of 1.0 (100%) is the normal amount, but you can increase this slightly to add more detail. Something like `1.02` seems pretty good.
* `iterations`: Number of upscale iterations to run. Be careful, this can add up fast - if you start at 512x512 with a 2.0 scale factor then 3 iterations will get you to 4096x4096.
* `vae_mode`: Mode used for encoding/decoding images. TAESD is fast/low VRAM but may reduce quality (you will also need the TAESD encoders installed in `models/vae_approx`). Normal will just use the normal VAE node, tiled with use the tiled VAE node. Alternatively, if you have [ComfyUI-TiledDiffusion](https://github.com/shiimizu/ComfyUI-TiledDiffusion) installed you can use `tiled_diffusion` here.

#### YAML Parameters

<details>

<summary>Expand for advanced parameters</summary>

Note: JSON is also valid YAML so you can use that instead if you prefer.

You can also override normal parameters from the node. For example:
```yaml
iterations: 3
scale_factor: 1.5
```

*Note*: A lot of these parameters are experimental/just stuff to try for a different effect. Their existence doesn't necessarily mean enabling/changing the parameter will be better than the default.

Default advanced parameter values:

```yaml
# Mode used for blending the normal model prediction with the guidance during guidance steps.
# Only has an effect when guidance_factor is less than 1.0
# One of: image, latent, wavelets
# "image" can only be used when guidance_mode is also "image" - will fall back to "wavelets" otherwise.
blend_by_mode: "image"

# Multiplier on the denoised wavelets. This would be the high frequency component by default.
denoised_wavelet_multiplier: 1.0

# See: https://pytorch-wavelets.readthedocs.io/en/latest/index.html
# dtcwt_mode enables using DTCWT rather than the default DWT.
dtcwt_biort: "near_sym_a"
dtcwt_mode: false
dtcwt_qshift: "qshift_a"
dwt_level: 1
dwt_mode: "symmetric"
dwt_wave: "db4"

# Flips the highpass/lowpass filters. Normally the reference lowpass and denoised highpass parts
# get used. If you flip them, you'll be using denoised for structural guidance and the reference
# for the high-frequency part.
dwt_flip_filters: false

# Enables agressively triggering Python's garbage collection to
# free up memory. May make out of memory issues less likely.
enable_gc: true

# Enables aggressively clearing the CUDA cache. May make out of memory issues less likely.
# It's same to enable this on non-Nvidia GPUs, it just won't do anything.
enable_cache_clearing: true

# Number of times to restart guidance steps. (Does a restart back like restart sampling.)
guidance_restart: 0
# Factor for noise added during guidance restarts.
guidance_restart_s_noise: 1.0

# Multiplier on the reference wavelets. This would be the low frequency component by default.
reference_wavelet_multiplier: 1.0

# Mode used for simple image rescales. Probably the main alternative here is setting it to lanczos.
# See: https://pillow.readthedocs.io/en/stable/handbook/concepts.html#filters-comparison-table
resample_mode: "bicubic"

# Increment image sizes are rounded to. Must be at least 8 and a multiple of 8.
rescale_increment: 64

# Workaround for long standing ComfyUI bug. See:
#  https://github.com/comfyanonymous/ComfyUI/issues/2833
#  https://github.com/comfyanonymous/ComfyUI/pull/4518
#  https://github.com/comfyanonymous/ComfyUI/pull/2841
seed_rng: true

# Only has an effect when seed_rng is enabled. Advances the
# RNG to avoid a case where you use the same noise during sampling
# as the initial noise.
seed_rng_offset: 1

# Mode used for sharpening. Can be one of: gaussian, contrast_adaptive
# If using contrast_adaptive, I'd recommend setting sharpen_strength a bit lower.
sharpen_mode: "gaussian"

sharpen_gaussian_kernel_size: 3
sharpen_gaussian_sigma: [0.1, 2.0]

# Strength of the sharpen effect. Set to 0 to disable.
sharpen_strength: 1.0

# Disables the callback function (basically disables previews).
skip_callback: false

# Allows specifying an offset into highres_sigmas.
# You can use a negative number here, in which case we count from the end.
sigma_offset: 0

# When enabled, uses an upscale model if connected. Mainly useful with
# iteration overrides.
use_upscale_model: true

# Allows passing extra arguments to the VAE encoder/decoder. Must be null or an object.
# Mainly useful with tiled_diffusion where you could do something like:
#   vae_decode_kwargs: { fast: false }
vae_decode_kwargs: null
vae_encode_kwargs: null

# Either null or an object.
# Allows overriding the sigma used for highres steps. See description below.
schedule_override: null

# Either null or an object.
# Allows overriding parameters per iteration. See description below.
iteration_override: null
```

**Iteration Overrides**

Example:

```yaml
iteration_override:
    0:
        scale_factor: 2.0
    1:
        scale_factor: 1.5
        skip_callback: true
```

You can override most parameters this way. Exceptions: Node inputs, `iteration_override` itself and `iterations`.

The `iteration_overrides` should either be `null` (disabled) or a YAML object with the iteration number (note: zero-based) as the key which contains an object with parameters in the same format as the main YAML parameters. Can be used to vary `scale_factor` across iterations, switched to tiled VAE only when the image is large enough for it to be worthwhile, disable previews (via `skip_callback: false`) if you're running out of memory at high res, etc.

**Schedule Overrides**

```yaml
# Schedule overrides can be specified at the top level.
schedule_override:
    schedule_name: karras
    steps: 20
    denoise: 0.3
    sigma_max: 14.614632
    sigma_min: 0.0291675

# Or in an interation override:
iteration_override:
    1:
        schedule_override:
            schedule_name: sgm_uniform
            steps: 15
            denoise: 0.35

# Note: Example only, not a recommendation.
```

`schedule_name` and `steps` are required, `denoise` is optional and defaults to `1.0` (not recommended for actual use). You may also specify additional parameters if the scheduler node supports them. For example, `karras` supports `sigma_min`, `sigma_max` and `rho`. `sigma_min` and `sigma_max` will default to the model's values which may be different from the node.

Supported schedules: `alignyoursteps`, `beta`, `ddim_uniform`, `exponential`, `gits`, `karras`, `laplace`, `normal`, `polyexponential`, `sgm_uniform`, `simple`, `vp`

Schedule overrides may also be combined with the `sigma_offset` parameter. The official DiffuseHigh uses the last 15 steps of a 50 step Karras schedule which would look like:

```yaml
schedule_override:
    schedule_name: karras
    steps: 50
    # denoise defaults to 1.0 here.

# Negative values count from the end.
# Note that this is 16 because steps are from a -> b, b -> c, etc.
sigma_offset: -16
```

</details>

***

## Tips/Recommendations

I tried to set the node defaults to align with the official implementation. These are my personal recommendations mainly based on usage with SD15:

* If you're using SD15 (possibly SDXL also), MSW-MSA attention from my [jankhidiffusion](https://github.com/blepping/comfyui_jankhidiffusion) is a significant speed increase. I feel like it really is a performance free lunch.
* Using `latent` guidance mode is about twice as fast as `image`. You may need to reduce the guidance factor a bit and/or enable fadeout.
* Using a fast upscale model like RealESRGAN_x2 may increase quality without much of a performance cost.
* It's very important that the initial reference is as close to flawless as possible. Unlike the normal highres fix approach which can sometimes fix issues when you set the denoise relatively high, DiffuseHigh guidance keeps the model from diverging from the reference too much. This can be a double edged sword in some cases.
* The sampler has a workaround for a [long standing bug in ComfyUI](https://github.com/comfyanonymous/ComfyUI/issues/2833) where generations aren't deterministic when `add_noise` is disabled in the sampler. However, this may change seeds. You can disable the workaround via the advanced YAML options - see `seed_rng` and `seed_rng_offset`.
* For `taesd` VAE mode, you will need the TAESD encoder models available at https://github.com/madebyollin/taesd - put them in `models/vae_approx`.

***

## Credits

Heavily referenced from the official implementation: [DiffuseHigh](https://github.com/yhyun225/DiffuseHigh/)
