import os
import json
import torch
import folder_paths
import safetensors.torch
import comfy.utils
import safetensors
from collections import OrderedDict

try:
    import comfy_kitchen as ck
    from comfy_kitchen.tensor import TensorCoreNVFP4Layout
except ImportError:
    print("⚠️ [Convert-to-NVFP4] comfy-kitchen introuvable.")

class ConvertToNVFP4:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("diffusion_models"),),
                "output_filename": ("STRING", {"default": "model-nvfp4"}),
                "model_type": ([
                    "Z-Image-Turbo", 
                    "Z-Image-Base", 
                    "Flux.1-dev", 
                    "Flux.1-Fill", 
                    "Flux.2-dev", 
                    "Qwen-Image-Edit-2511", 
                    "Qwen-Image-2512", 
                    "Wan2.2-i2v-high-low",
                    "LTX-2-19b-dev-or-distilled"
                ], {"default": "Z-Image-Turbo"}),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "convert"
    CATEGORY = "Kitchen"
    OUTPUT_NODE = True

    def convert(self, model_name, output_filename, model_type, device):
        input_path = folder_paths.get_full_path("diffusion_models", model_name)
        output_path = os.path.join(os.path.dirname(input_path), f"{output_filename}.safetensors")
        
        # --- CONFIGURATION DES PROFILS ---
        if model_type == "Qwen-Image-Edit-2511":
            BLACKLIST = ["img_in", "txt_in", "time_text_embed", "norm_out", "proj_out"]
            FP8_LAYERS = []
        elif model_type == "Qwen-Image-2512":
            BLACKLIST = ["img_in", "txt_in", "time_text_embed", "norm_out", "proj_out", "img_mod.1"]
            FP8_LAYERS = ["txt_mlp", "txt_mod"]
        elif model_type == "Wan2.2-i2v-high-low":
            BLACKLIST = ["text_embedding", "time_embedding", "time_projection", "head"]
            FP8_LAYERS = []
        elif model_type in ["Flux.1-dev", "Flux.1-Fill", "Flux.2-dev"]:
            BLACKLIST = ["img_in", "txt_in", "time_in", "vector_in", "guidance_in", "final_layer", "class_embedding", "single_stream_modulation", "double_stream_modulation_img", "double_stream_modulation_txt"]
            FP8_LAYERS = []
        elif model_type == "Z-Image-Base":
            BLACKLIST = ["adaLN_modulation", "norm", "final_layer", "cap_embedder", "x_embedder", "noise_refiner", "context_refiner", "t_embedder"]
            FP8_LAYERS = ["attention.out", "layers.0.", "layers.29."]
        elif model_type == "Z-Image-Turbo":
            BLACKLIST = ["cap_embedder", "x_embedder", "noise_refiner", "context_refiner", "t_embedder", "final_layer"]
            FP8_LAYERS = []
        elif model_type == "LTX-2-19b-dev-or-distilled":
            BLACKLIST = [
                "vae.", "vocoder.", "connector", "proj_out",
                "norm", "bias", "scale", "embedder", "patchify", "table",
                "transformer_blocks.0.",
                "transformer_blocks.43.", "transformer_blocks.44.", 
                "transformer_blocks.45.", "transformer_blocks.46.", 
                "transformer_blocks.47.", "projection", "adaln_single"
            ]
            FP8_LAYERS = []
        else:
            BLACKLIST = ["cap_embedder", "x_embedder", "noise_refiner", "context_refiner", "t_embedder", "final_layer"]
            FP8_LAYERS = []

        print(f"🚀 Mode {model_type} activé")
        
        temp_diffusers_meta = {}
        if model_type == "LTX-2-19b-dev-or-distilled":
            with safetensors.safe_open(input_path, framework="pt") as f:
                orig_meta = f.metadata()
                if orig_meta:
                    for key in ["config", "license", "encrypted_wandb_properties"]:
                        if key in orig_meta:
                            temp_diffusers_meta[key] = orig_meta[key]

        sd = safetensors.torch.load_file(input_path)
        quant_map = {"format_version": "1.0", "layers": {}}
        new_sd = {}
        
        pbar = comfy.utils.ProgressBar(len(sd))
        print(f"⚙️ Conversion lancée sur : {device}")

        for i, (k, v) in enumerate(sd.items()):
            pbar.update_absolute(i + 1)

            if any(name in k for name in BLACKLIST):
                new_sd[k] = v.to(dtype=torch.bfloat16)
                continue

            if v.ndim == 2 and ".weight" in k:
                base_k_file = k.replace(".weight", "")
                
                if model_type == "LTX-2-19b-dev-or-distilled":
                    base_k_meta = k.replace(".weight", "") 
                else:
                    if "model.diffusion_model." in base_k_file:
                        base_k_meta = base_k_file.split("model.diffusion_model.")[-1]
                    else:
                        base_k_meta = base_k_file
      
                v_tensor = v.to(device=device, dtype=torch.bfloat16)

                if FP8_LAYERS and any(name in k for name in FP8_LAYERS):
                    print(f"🌸 FP8 Cuisine : {k}")
                    weight_scale = (v_tensor.abs().max() / 448.0).clamp(min=1e-12).float()
                    weight_quantized = ck.quantize_per_tensor_fp8(v_tensor, weight_scale)
                    new_sd[k] = weight_quantized.cpu()
                    new_sd[f"{base_k_file}.weight_scale"] = weight_scale.to(torch.bfloat16).cpu()
                    quant_map["layers"][base_k_meta] = {"format": "float8_e4m3fn"}
                    if device == "cuda": del v_tensor
                    continue

                print(f"💎 NVFP4 : {k}")
                try:
                    qdata, params = TensorCoreNVFP4Layout.quantize(v_tensor)
                    tensors = TensorCoreNVFP4Layout.state_dict_tensors(qdata, params)
                    for suffix, tensor in tensors.items():
                        new_sd[f"{base_k_file}.weight{suffix}"] = tensor.cpu()
                    quant_map["layers"][base_k_meta] = {"format": "nvfp4"}
                except Exception:
                    new_sd[k] = v.to(dtype=torch.bfloat16)
                
                if device == "cuda": del v_tensor
            else:
                new_sd[k] = v.to(dtype=torch.bfloat16)

        final_metadata = OrderedDict()
        final_metadata["_quantization_metadata"] = json.dumps(quant_map)
        
        if model_type == "LTX-2-19b-dev-or-distilled":
            for k, v in temp_diffusers_meta.items():
                final_metadata[k] = v

        print(f"💾 Saving file | Type: {model_type} | Path: {output_path}")
        safetensors.torch.save_file(new_sd, output_path, metadata=final_metadata)
        
        total_bytes = os.path.getsize(output_path)
        print(f"✅ Terminé. Taille finale : {round(total_bytes / (1024**3), 2)} Go")
        
        return (f"Succès ({model_type}) : {output_filename}.safetensors",)

NODE_CLASS_MAPPINGS = {"ConvertToNVFP4": ConvertToNVFP4}
NODE_DISPLAY_NAME_MAPPINGS = {"ConvertToNVFP4": "🍳 Kitchen NVFP4 Converter"}