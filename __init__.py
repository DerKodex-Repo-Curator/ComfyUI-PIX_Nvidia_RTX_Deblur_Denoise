import torch
import nvvfx
from typing_extensions import override

from comfy_api.latest import ComfyExtension, io


class RTXDeblur(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="RTXDeblur",
            display_name="RTX Deblur",
            category="image/postprocessing",
            search_aliases=["rtx", "nvidia", "deblur", "sharpen", "blur removal"],
            inputs=[
                io.Image.Input("images"),
                io.Combo.Input(
                    "quality",
                    options=["Light", "Moderate", "Aggressive", "Maximum"],
                    default="Moderate",
                    tooltip="Deblur strength. Light=minimal correction, Maximum=strongest.",
                ),
            ],
            outputs=[
                io.Image.Output("images"),
            ],
        )

    @classmethod
    def execute(cls, images: torch.Tensor, quality: str) -> io.NodeOutput:
        b, h, w, c = images.shape
        MAX_PIXELS = 1024 * 1024 * 16
        batch_size = max(1, MAX_PIXELS // (w * h))

        quality_mapping = {
            "Light":      nvvfx.effects.QualityLevel.DEBLUR_LOW,
            "Moderate":   nvvfx.effects.QualityLevel.DEBLUR_MEDIUM,
            "Aggressive": nvvfx.effects.QualityLevel.DEBLUR_HIGH,
            "Maximum":    nvvfx.effects.QualityLevel.DEBLUR_ULTRA,
        }
        selected_quality = quality_mapping.get(quality, nvvfx.effects.QualityLevel.DEBLUR_MEDIUM)

        with nvvfx.VideoSuperRes(selected_quality) as sr:
            sr.output_width = w
            sr.output_height = h
            sr.load()

            out_tensor = torch.empty((b, h, w, c), device=images.device, dtype=images.dtype)
            for i in range(0, b, batch_size):
                batch_cuda = images[i:i + batch_size].cuda().permute(0, 3, 1, 2).float().contiguous()
                for j in range(batch_cuda.shape[0]):
                    dlpack_out = sr.run(batch_cuda[j]).image
                    out_tensor[i + j: i + j + 1] = torch.from_dlpack(dlpack_out).movedim(0, -1).unsqueeze(0)

        return io.NodeOutput(out_tensor)


class RTXDenoise(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="RTXDenoise",
            display_name="RTX Denoise",
            category="image/postprocessing",
            search_aliases=["rtx", "nvidia", "denoise", "noise removal", "clean"],
            inputs=[
                io.Image.Input("images"),
                io.Combo.Input(
                    "quality",
                    options=["Light", "Moderate", "Aggressive", "Maximum"],
                    default="Moderate",
                    tooltip="Denoise strength. Light=minimal correction, Maximum=strongest.",
                ),
            ],
            outputs=[
                io.Image.Output("images"),
            ],
        )

    @classmethod
    def execute(cls, images: torch.Tensor, quality: str) -> io.NodeOutput:
        b, h, w, c = images.shape
        MAX_PIXELS = 1024 * 1024 * 16
        batch_size = max(1, MAX_PIXELS // (w * h))

        quality_mapping = {
            "Light":      nvvfx.effects.QualityLevel.DENOISE_LOW,
            "Moderate":   nvvfx.effects.QualityLevel.DENOISE_MEDIUM,
            "Aggressive": nvvfx.effects.QualityLevel.DENOISE_HIGH,
            "Maximum":    nvvfx.effects.QualityLevel.DENOISE_ULTRA,
        }
        selected_quality = quality_mapping.get(quality, nvvfx.effects.QualityLevel.DENOISE_MEDIUM)

        with nvvfx.VideoSuperRes(selected_quality) as sr:
            sr.output_width = w
            sr.output_height = h
            sr.load()

            out_tensor = torch.empty((b, h, w, c), device=images.device, dtype=images.dtype)
            for i in range(0, b, batch_size):
                batch_cuda = images[i:i + batch_size].cuda().permute(0, 3, 1, 2).float().contiguous()
                for j in range(batch_cuda.shape[0]):
                    dlpack_out = sr.run(batch_cuda[j]).image
                    out_tensor[i + j: i + j + 1] = torch.from_dlpack(dlpack_out).movedim(0, -1).unsqueeze(0)

        return io.NodeOutput(out_tensor)


class NVVFXDeblurDenoiseExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [RTXDeblur, RTXDenoise]


async def comfy_entrypoint() -> NVVFXDeblurDenoiseExtension:
    return NVVFXDeblurDenoiseExtension()


# hack so registry picks up the node names
if False:
    NODE_CLASS_MAPPINGS = {
        "RTXDeblur": RTXDeblur,
        "RTXDenoise": RTXDenoise,
    }
