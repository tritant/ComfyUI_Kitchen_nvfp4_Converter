import os
import json
import torch
import folder_paths
import safetensors.torch
import comfy.utils

# Importation de la cuisine
try:
    import comfy_kitchen as ck
    from comfy_kitchen.tensor import TensorCoreNVFP4Layout
except ImportError:
    print("⚠️ [Convert-to-NVFP4] comfy-kitchen introuvable. Installation requise : pip install comfy-kitchen")

class ConvertToNVFP4:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("diffusion_models"),),
                "output_filename": ("STRING", {"default": "model-nvfp4"}),
                "model_type": (["Z-Image", "Flux.1"], {"default": "Z-Image"}),
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
        
        # --- CHOIX DE LA BLACKLIST ---
        if model_type == "Flux.1":
            BLACKLIST = ["img_in", "txt_in", "time_in", "vector_in", "guidance_in", "final_layer", "class_embedding"]
            print(f"🚀 Mode Flux.1 activé")
        else:
            BLACKLIST = ["cap_embedder", "x_embedder", "noise_refiner", "context_refiner", "t_embedder", "final_layer"]
            print(f"🚀 Mode Z-Image activé")

        print(f"📦 Chargement du modèle : {model_name}")
        sd = safetensors.torch.load_file(input_path)
        
        quant_map = {"format_version": "1.0", "layers": {}}
        new_sd = {}
        
        total_steps = len(sd)
        pbar = comfy.utils.ProgressBar(total_steps)

        print(f"⚙️ Conversion NVFP4 lancée sur : {device}")
        
        for i, (k, v) in enumerate(sd.items()):
            pbar.update_absolute(i + 1)

            # 1. Vérification de la Blacklist
            if any(name in k for name in BLACKLIST):
                new_sd[k] = v.to(dtype=torch.bfloat16)
                continue

            # 2. Quantification des poids linéaires (2D)
            if v.ndim == 2 and ".weight" in k:
                base_k_file = k.replace(".weight", "")
                
                # Nettoyage du préfixe pour les métadonnées (Standard ComfyUI)
                base_k_meta = base_k_file
                if base_k_meta.startswith("model.diffusion_model."):
                    base_k_meta = base_k_meta.replace("model.diffusion_model.", "")
                
                # Passage sur le device choisi
                v_tensor = v.to(device=device, dtype=torch.bfloat16)
                
                try:
                    # Quantification
                    qdata, params = TensorCoreNVFP4Layout.quantize(v_tensor)
                    tensors = TensorCoreNVFP4Layout.state_dict_tensors(qdata, params)
                    
                    for suffix, tensor in tensors.items():
                        new_sd[f"{base_k_file}.weight{suffix}"] = tensor.cpu()
                    
                    quant_map["layers"][base_k_meta] = {"format": "nvfp4"}

                except Exception as e:
                    print(f"⚠️ Erreur sur {k}, fallback BF16: {e}")
                    new_sd[k] = v.to(dtype=torch.bfloat16)
                
                if device == "cuda":
                    del v_tensor
            else:
                new_sd[k] = v.to(dtype=torch.bfloat16)

        # Injection des métadonnées
        metadata = {"_quantization_metadata": json.dumps(quant_map)}
        
        print(f"💾 Sauvegarde du fichier final...")
        safetensors.torch.save_file(new_sd, output_path, metadata=metadata)
        
        if device == "cuda":
            torch.cuda.empty_cache()
            
        print(f"✅ Terminé : {output_path}")
        return (f"Succès ({model_type}) : {output_filename}.safetensors",)

NODE_CLASS_MAPPINGS = {"ConvertToNVFP4": ConvertToNVFP4}
NODE_DISPLAY_NAME_MAPPINGS = {"ConvertToNVFP4": "🍳 Kitchen NVFP4 Converter"}