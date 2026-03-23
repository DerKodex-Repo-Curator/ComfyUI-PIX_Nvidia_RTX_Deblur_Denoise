# Nvidia_RTX_Deblur_Denoise_ComfyUI

Two GPU-accelerated post-processing nodes for ComfyUI powered by NVIDIA RTX technology via the `nvvfx` SDK. Only works on NVIDIA RTX GPUs.

This is a ComfyUI port of the Deblur and Denoise features from [sd-forge-nvidia-vfx](https://github.com/Haoming02/sd-forge-nvidia-vfx/tree/main) by [Haoming02](https://github.com/Haoming02). Only these two features have been ported as a custom node pack for ComfyUI.

## Nodes

### RTX Deblur (`image/postprocessing`)
Corrects blur and softness in images using NVIDIA's AI deblur model. Output is the same resolution as input.

### RTX Denoise (`image/postprocessing`)
Removes noise and grain from images using NVIDIA's AI denoise model. Output is the same resolution as input.

## Quality Levels (both nodes)

| Level | Description |
|-------|-------------|
| Light | Minimal correction, preserves most original texture |
| Moderate | Balanced (default) |
| Aggressive | Stronger correction |
| Maximum | Strongest correction, may over-process |

## Requirements

- NVIDIA RTX GPU
- `nvidia-vfx` Python package

## Installation

1. Place this folder in your ComfyUI `custom_nodes/` directory
2. Restart ComfyUI
3. Search "RTX Deblur" or "RTX Denoise" in the node browser under `image/postprocessing`

## Example Workflows

**Basic deblur:**
`LoadImage → RTX Deblur (Moderate) → SaveImage`

**Basic denoise:**
`LoadImage → RTX Denoise (Light) → SaveImage`

**Combined:**
`LoadImage → RTX Denoise (Moderate) → RTX Deblur (Moderate) → SaveImage`

## Notes

- Both nodes use zero-copy GPU memory sharing via DLPack — tensors stay on the GPU throughout
- Batch processing is supported; batches are automatically split to fit within GPU memory limits
- These are distinct from the `RTXVideoSuperResolution` node — they do not change image dimensions

## Credits

- Original implementation by [Haoming02](https://github.com/Haoming02) — [sd-forge-nvidia-vfx](https://github.com/Haoming02/sd-forge-nvidia-vfx/tree/main)
- This project ports the Deblur and Denoise features from that extension into a standalone ComfyUI custom node pack
