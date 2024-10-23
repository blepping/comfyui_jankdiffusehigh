# Changes

Note, only relatively significant changes to user-visible functionality will be included here. Most recent changes at the top.

## 20241023

* Initial support for rectified flow models (Flux, SD3, SD3.5). Might slightly change seeds for other models.
* Improve contrast adaptive sharpening (hopefully). Will change seeds for workflows using `sharpen_mode: contrast_adaptive`, you can use `sharpen_mode: contrast_adaptive_raw` for the old behavior.

## 20241019

* Added workaround for https://github.com/comfyanonymous/ComfyUI/issues/2833 - may change seeds. You can disable it with `seed_rng: false` in YAML parameters.
* Added `enable_gc` and `enable_cache_clearing` options to YAML parameters.
* Added `use_upscale_model` option to YAML parameters - allows disabling use of an upscale model even when connected.
* Made `highres_sigmas` input optional, now defaults to the same schedule as the official implementation.
* Added `schedule_override` option to YAML parameters, allows internally generating a schedule. See README for details.
