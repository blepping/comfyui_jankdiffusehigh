# ComfyUI jank DiffuseHigh
Janky implementation of [DiffuseHigh](https://github.com/yhyun225/DiffuseHigh/) for ComfyUI.

Facilitates generating directly to resolutions higher than the model was trained for, similar to Kohya Deep Shrink, HiDiffusion, etc.

This is a best-effort attempt at implementation. If you experience poor results, please don't let it reflect on the official version. There's a good chance it's something I did wrong.

## Current Status

Beta - lightly tested but the main features are in place. Mainly targeted at advanced users who can deal with some weird stuff and frequent workflow-breaking changes.

See the [changelog](changelog.md) for recent user visible changes.

**Known issues/caveats**

* There will be frequent workflow-breaking changes for a while yet.
* Progress and previews are pretty wonky (you can look at the log for some progress information).
* Using VAE or upscale models may result in the main model getting repeatedly unloaded/reloaded. Try using `latent` as the `guidance_mode`. If you actually have enough VRAM, maybe disabling smart memory (via ComfyUI commandline parameter) would help.
* Brownian noise-based (AKA SDE) samplers may be a bit weird here, there is a workaround in place but it might not be enough. Also don't use with prompt-control's PCSplitSampling stuff.

**Rectified Flow models note**: Should now work with RF models. SD3.5 apparently cannot handle high res images (even img2img) at all, so I don't recommend trying that. Flux seems to work pretty well. `image` guidance mode seems noticeably better than `latent` for Flux (based on my very limited testing) although it is slow. I haven't tested SD3.0 or other RF models, jank DiffuseHigh should handle them correctly but whether the results are actually decent I really couldn't say. Using `guidance_restart` probably won't work correctly.

## Description

The DiffuseHigh approach is similar to an iterative upscale/run some more steps at low denoise approach with a twist: it mixes in guidance from a reference image for a number of steps at the beginning of each sampling iteration. The guidance is derived from the low frequency parts of the reference and also gets sharpened first to increase detail.

My approach implements it as a sampler which means it's _mostly_ model-agnostic and avoids some common issues with alternative approaches like Deep Shrink and HiDiffusion that require model patches. It's also possible to generate a low or mid-resolution image to see if you like the results and then increase the number of iterations to get a similar result where with Deep Shrink/HiDiffusion type effects enabling/disabling the patch will effectively change the seed.

The main disadvantage compared to the alternatives I mentioned is that it is relatively slow and VRAM hungry since it requires multiple iterations at high res while Deep Shrink/HiDiffusion actually speed up generation while the scaling effect is active.

## Nodes

### `DiffuseHighSampler`

This is the main DiffuseHigh sampler node.

I recommend expanding the YAML Parameters section and at least skimming through it so you can see what your options are. Most advanced features are controlled there - you can do stuff like switch VAEs, upscale models or other parameters per iteration which can be a very powerful tool.

**Input Parameters**: You can connect stuff like VAEs, upscale models and masks using this input.

**Mask Usage**: Masks can be connected via `input_params_opt`. There are currently two ways they can be used: as a global mask or to mask the guidance. If the mask has no name, it is by default a global mask. If you name it `guidance` then it will be treated as a guidance mask. You don't have to stick to those names, `mask_name` and `guidance_mask_name` in the YAML parameters can be used to control what masks are used. Where global masks are set, the model is allowed to change the image - where they aren't set, it will be the reference image. Guidance masks apply guidance where the mask is set and you get the model's normal prediction otherwise. Non-binary masks work the way you'd expect: you'll get a blend based on the mask strength in a particular area.

#### Inputs

* `highres_sigmas`: Optional: Sigmas used for everything other than the initial reference image. **Note**: Should be around 0.3-0.5 denoise. You won't get good results connecting something like `KarrasScheduler` here without splitting the sigmas. If not specified, will use the last 15 steps of a 50 step Karras schedule like the official implementation.
* `sampler`: Optional: Default sampler used for steps. If not specified the sampler will default to non-ancestral Euler.
* `reference_image_opt`: Optional: Image used for the initial pass. If not connected, a low-res initial reference will be generated using the schedule from the normal sigmas (i.e. the sigmas attached to `SamplerCustom` or whatever actual sampler node you're using).
* `guidance_sampler_opt`: Optional: Sampler used for guidance steps. If not specified, will fallback to the base sampler.
* `reference_sampler_opt`: Optional: Sampler used to generate the initial low-resolution reference. Only used if reference_image_opt is not connected.
* `vae_opt`: Optional when vae_mode is set to `taesd`, otherwise this is the VAE that will be used for encoding/decoding images. If using TAESD, you will require the corresponding encoder (which I believe ComfyUI does not install by default). TAESD models go in `models/vae_approx`, you can find them here: https://github.com/madebyollin/taesd
* `upscale_model_opt`: Optional: Model used for upscaling. When not attached, simple image scaling will be used. Regardless, the image will be scaled to match the size expected based on `scale_factor`. For example, if you use scale_factor 2 and a 4x upscale model, the image will get scaled down after the upscale model runs.
* `input_params_opt`: Optional: Output from a `DiffuseHighParam` node. Allows connecting additional inputs that can't be specified by text (i.e. VAEs, upscale models and the like).
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

<summary>★ Click to expand for information on YAML parameters ★</summary>

Note: JSON is also valid YAML so you can use that instead if you prefer.

You can also override normal parameters from the node. For example:
```yaml
iterations: 3
scale_factor: 1.5
```

*Note*: A lot of these parameters are experimental/just stuff to try for a different effect. Their existence doesn't necessarily mean enabling/changing the parameter will be better than the default.

Default advanced parameter values:

```yaml
# Mainly useful in iteration overrides, allows skipping an iteration. When defined at
# the toplevel it will skip everything which probably isn't what you want.
skip: false

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

# Mode used for sharpening. Can be one of:
#   gaussian, contrast_adaptive, contrast_adaptive_raw
# If using contrast_adaptive_raw, I'd recommend setting sharpen_strength a bit lower.
sharpen_mode: "gaussian"

sharpen_gaussian_kernel_size: 3
sharpen_gaussian_sigma: [0.1, 2.0]

# Strength of the sharpen effect. Set to 0 to disable.
sharpen_strength: 1.0

# Disables the callback function (basically disables previews).
skip_callback: false

# Offset to sigmas passed to the model, -0.05 would mean reduce the sigma by 5%.
# If unset, sigma_dishonesty_factor_guidance will use the value from sigma_dishonesty_factor
# for guidance steps.
# Telling the model there's less noise than there actually is can increase detail
# (and conversely telling it there's more will reduce detail/smooth things out).
# A little goes a long way. Start with something like -0.03 to increase detail.
sigma_dishonesty_factor: 0.0
sigma_dishonesty_factor_guidance: null

# When enabled, uses an upscale model if connected. Mainly useful with
# iteration overrides.
use_upscale_model: true

# Only has an effect if the upscale model is connected and enabled. This
# will force it to run even if the scale factor is 1 or the size already
# matches the scale. This is to allow use of 1x upscale models that just
# add an effect like film grain.
force_upscale_model: false

# Allows passing extra arguments to the VAE encoder/decoder. Must be null or an object.
# Mainly useful with tiled_diffusion where you could do something like:
#   vae_decode_kwargs: { fast: false }
vae_decode_kwargs: null
vae_encode_kwargs: null

# Can be used to access named parameters connected with a DiffuseHigh Param
# node. Also serves as a reference for the default names. For example, if
# you want to connect highres sigmas with the DiffuseHigh Param node, you
# would set the type to "sigmas" and the name to "highres".
vae_name: ""
upscale_model_name: ""
highres_sigmas_name: "highres"
reference_image_name: "reference"
sampler_name: ""
reference_sampler_name: "reference"
guidance_sampler_name: "guidance"
custom_noise_name: ""
restart_custom_noise_name: "restart"
mask_name: ""
guidance_mask_name: "guidance"

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

Supported schedules: `alignyoursteps`, `beta`, `ddim_uniform`, `exponential`, `gits`, `karras`, `laplace`, `normal`, `polyexponential`, `sgm_uniform`, `simple`, `vp`, `kl_optimal` (once/if support is merged into ComfyUI)

</details>

### `DiffuseHighParam`

This node allows you to connect additional inputs to the `DiffuseHighSampler` node, such as VAEs, upscale models, custom noise samplers. You can chain these nodes together to specify multiple parameters at once.

List of main sampler inputs and corresponding parameter names:

* `guidance_sampler_opt`: type `sampler`, name `guidance`
* `highres_sigmas`: type `sigmas`, name `highres`
* `reference_image_opt`: type `image`, name `reference`
* `reference_sampler_opt`: type `sampler`, name `reference`
* `sampler`: type `sampler`
* `upscale_model_opt`: type `upscale_model`
* `vae_opt`: type `vae`

If not specified, then name is blank.

#### Inputs

* `value`: Input value, the type varies based on the `input_type` parameter (see below).
* `params_opt`: Optional: You can connect the output from another DiffuseHighParam node here to specify multiple parameters.
* `yaml_parameters`: Optional: Allows specifying custom parameters via YAML. This input can be converted into a multiline text widget. Note: When specifying paramaters this way, there is very little error checking.

#### Parameters

* `input_type`: Specify the input type of the connected `value`.
* `name`: Allows specifying a name for an input.


***

## Tips/Recommendations

I tried to set the node defaults to align with the official implementation. These are my personal recommendations mainly based on usage with SD15:

* If you're using SD15 (possibly SDXL also), MSW-MSA attention from my [jankhidiffusion](https://github.com/blepping/comfyui_jankhidiffusion) is a significant speed increase. I feel like it really is a performance free lunch.
* Using `latent` guidance mode is about twice as fast as `image`. You may need to reduce the guidance factor a bit and/or enable fadeout.
* Using a fast upscale model like RealESRGAN_x2 may increase quality without much of a performance cost.
* It's very important that the initial reference is as close to flawless as possible. Unlike the normal highres fix approach which can sometimes fix issues when you set the denoise relatively high, DiffuseHigh guidance keeps the model from diverging from the reference too much. This can be a double edged sword in some cases.
* The sampler has a workaround for a [long standing bug in ComfyUI](https://github.com/comfyanonymous/ComfyUI/issues/2833) where generations aren't deterministic when `add_noise` is disabled in the sampler. However, this may change seeds. You can disable the workaround via the advanced YAML options - see `seed_rng` and `seed_rng_offset`.
* For `taesd` VAE mode, you will need the TAESD encoder models available at https://github.com/madebyollin/taesd - put them in `models/vae_approx`.
* You can use DiffuseHigh as an enhanced highres-fix by passing a pre-upscaled reference image, setting the iteration count to one and using a scale factor of 1.0.
* Setting `sigma_dishonesty_factor` and/or `sigma_dishonesty_factor_guidance` to a low negative value can be used to increase detail even for non-ancestral samplers (similar effect to increasing `s_noise`). See the YAML parameters section of this README.
* Using an upscale model or `image` guidance seems to make the most difference when you're going from low to mid-resolution (i.e. 512x512 to 1024x1024) so it may make sense to use the relatively slow `image` guidance and an upscale model for the first iteration and then switch to `latent` guidance and set `use_upscale_model: false` for subsequent iterations.
* It's possible to switch VAEs, upscale models and samplers between iterations using named `DiffuseHigh Param` inputs and a YAML parameter like `vae_name: whatever`.
***

## Integration

Some additional features will be available if you have other node packs installed:

* [ComfyUI-TiledDiffusion](https://github.com/shiimizu/ComfyUI-TiledDiffusion) - provides better tiled VAE that JankDiffuseHigh can take advantage of.
* You can use `OCS_CUSTOM_NOISE` or `SONAR_CUSTOM_NOISE` in the DiffuseHigh Param node if you have the respective node packs installed: [Overly Complicated Sampling](https://github.com/blepping/comfyui_overly_complicated_sampling), [ComfyUI-sonar](https://github.com/blepping/ComfyUI-sonar) installed.

***

## Credits

* Initial version heavily referenced from the official implementation: [DiffuseHigh](https://github.com/yhyun225/DiffuseHigh/)
* Contrast-adaptive sharpening sources: [1](https://github.com/GPUOpen-Effects/FidelityFX-CAS/blob/master/ffx-cas/ffx_cas.h), [2](https://github.com/Jamy-L/Pytorch-Contrast-Adaptive-Sharpening/), [3](https://github.com/Clybius)
* `sigma_dishonesty_factor` concept from A1111's [Detail Daemon](https://github.com/muerrilla/sd-webui-detail-daemon) extension. (There's also a [ComfyUI version](https://github.com/Jonseed/ComfyUI-Detail-Daemon) now.)

Thanks!
