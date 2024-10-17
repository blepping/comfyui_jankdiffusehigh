# ComfyUI jank DiffuseHigh
Janky implementation of [DiffuseHigh](https://github.com/yhyun225/DiffuseHigh/) for ComfyUI.

Facilitates generating directly to resolutions higher than the model was trained for, similar to Kohya Deep Shrink, HiDiffusion, etc.

This is a best-effort attempt at implementation. If you experience poor results, please don't let it reflect on the official version. There's a good chance it's something I did wrong.

## Current Status

Alpha - early implementation. Many rough edges but the core functionality is there. Mainly targetted at advanced users who can deal with some weird stuff and frequent workflow-breaking changes.

**Known issues/caveats**

* There will be frequent workflow-breaking changes for a while yet.
* Progress and previews are pretty wonky (you can look at the log for some progress information).
* Using VAE or upscale models may result in the main model getting repeatedly unloaded/reloaded. Try using `latent` as the `guidance_mode`. If you actually have enough VRAM, maybe disabling smart memory (via ComfyUI commandline parameter) would help.
* Currently only tested on SD15 and SDXL, may not work with models like Flux. (Not much testing in general as of yet.)

## Description

The DiffuseHigh approach is similar to an iterative upscale/run some more steps at low denoise approach with a twist: it mixes in guidance from a reference image for a number of steps at the beginning of each sampling iteration. The guidance is derived from the low frequency parts of the reference and also gets sharpened first to increase detail.

My approach implements it as a sampler which means it's _mostly_ model-agnostic and avoids some common issues with alternative approaches like Deep Shrink and HiDiffusion that require model patches. It's also possible to generate a low or mid-resolution image to see if you like the results and then increase the number of iterations to get a similar result where with Deep Shrink/HiDiffusion type effects enabling/disabling the patch will effectively change the seed.

The main disadvantage compared to the alternatives I mentioned is that it is relatively slow and VRAM hungry since it requires multiple iterations at high res while Deep Shrink/HiDiffusion actually speed up generation while the scaling effect is active.

## Nodes

### `DiffuseHighSampler`

#### Inputs

* `highres_sigmas`: Sigmas used for everything other than the initial reference image. **Note**: Should be around 0.3-0.5 denoise. You won't get good results connecting something like `KarrasScheduler` here without splitting the sigmas.
* `sampler`: Default sampler used for steps. If not specified the sampler will default to non-ancestral Euler.
* `reference_image_opt`: Optional: Image used for the initial pass. If not connected, a low-res initial reference will be generated using the schedule from the normal sigmas (i.e. the sigmas attached to `SamplerCustom` or whatever actual sampler node you're using).
* `guidance_sampler_opt`: Optional: Sampler used for guidance steps. If not specified, will fallback to the base sampler. Note: The sampler is called on individual steps, samplers that keep history will not work well here.
* `reference_sampler_opt`: Optional: Sampler used to generate the initial low-resolution reference. Only used if reference_image_opt is not connected.
* `vae_opt`: Optional when vae_mode is set to `taesd`, otherwise this is the VAE that will be used for encoding/decoding images. If using TAESD, you will require the corresponding encoder (which I believe ComfyUI does not install by default). TAESD models available here: https://github.com/madebyollin/taesd
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
* `vae_mode`: Mode used for encoding/decoding images. TAESD is fast/low VRAM but may reduce quality (you will also need the TAESD encoders installed). Normal will just use the normal VAE node, tiled with use the tiled VAE node. Alternatively, if you have [ComfyUI-TiledDiffusion](https://github.com/shiimizu/ComfyUI-TiledDiffusion) installed you can use `tiled_diffusion` here.

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
# "image" can only be used when guidance_mode is also "image" - will fall back to "wavelets" in that case.
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

# Mode used for sharpening. Can be one of: gaussian, contrast_adaptive
# If using contrast_adaptive, I'd recommend setting sharpen_strength a bit lower.
sharpen_mode: "gaussian"

# Allows disabling sharpening. Setting it to false is effectively the same as sharpening_strength: 0
sharpen_reference: true

sharpen_gaussian_kernel_size: 3
sharpen_gaussian_sigma: [0.1, 2.0]
sharpen_strength: 1.0

# Disables the callback function (basically disables previews).
skip_callback: false

# Allows specifying an offset into highres_sigmas.
sigma_offset: 0

# Allows passing extra arguments to the VAE encoder/decoder. Must be null or an object.
# Mainly useful with tiled_diffusion where you could do something like:
#   vae_decode_kwargs: { fast: false }
vae_decode_kwargs: null
vae_encode_kwargs: null

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


</details>

***

## Credits

Heavily referenced from the official implementation: [DiffuseHigh](https://github.com/yhyun225/DiffuseHigh/)
