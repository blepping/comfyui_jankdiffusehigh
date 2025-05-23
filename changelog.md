# Changes

Note, only relatively significant changes to user-visible functionality will be included here. Most recent changes at the top.

## 20250506

ComfyUI in its infinite wisdom decided to make it so you can no longer have parameters that default to inputs but can be converted to widgets and widgets take up the full space even if you're using an input now. Because of this, the YAML parameters in the node will take up a lot more space and can't be hidden. I'd make it so the parameter was just always an input but then it would be impossible to acces the parameters in any workflows that had previously converted the input to a widget.

* Work around breakage caused by recent ComfyUI frontend versions.

## 20250109

* Fixed an issue where sharpen didn't respect the mask if it existed.
* Added `ensure_model_mode` YAML parameter. Can be used to control the strategy jankdiffushigh uses to make sure the model doesn't get unloaded.
* You can now pass noise sampler parameters via `custom_noise_params`.

## 20241224

Reworked approach to integrating with external node packs. This _shouldn't_ cause any visible changes from a user perspective but please create an issue if you notice anything weird.

## 20241116

* Added the ability to mask both guidance and global changes. See the README section on masks.

## 20241031

* Added the `DiffuseHighParam` node and the ability to connect multiple VAEs, upscale models, noise generators and samplers as well as switch between them per iteration.

## 20241027

* `sigma_offset` YAML parameter removed - you can use schedule overrides to accomplish the same effect (see README).
* Chunked sampling mode added, should make samplers that care about state (i.e. momentum or history like `dpmpp_2m`) work better for guidance steps. May change seeds, you can disable with `chunked_sampling: false` in YAML parameters.
* Added `sigma_dishonesty_factor` and `sigma_dishonesty_factor_guidance` YAML parameters - can be used to increase detail. See README.

## 20241023

* Initial support for rectified flow models (Flux, SD3, SD3.5). Might slightly change seeds for other models.
* Improve contrast adaptive sharpening (hopefully). Will change seeds for workflows using `sharpen_mode: contrast_adaptive`, you can use `sharpen_mode: contrast_adaptive_raw` for the old behavior.

## 20241019

* Added workaround for https://github.com/comfyanonymous/ComfyUI/issues/2833 - may change seeds. You can disable it with `seed_rng: false` in YAML parameters.
* Added `enable_gc` and `enable_cache_clearing` options to YAML parameters.
* Added `use_upscale_model` option to YAML parameters - allows disabling use of an upscale model even when connected.
* Made `highres_sigmas` input optional, now defaults to the same schedule as the official implementation.
* Added `schedule_override` option to YAML parameters, allows internally generating a schedule. See README for details.
