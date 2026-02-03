# ComfyUI Kitchen NVFP4 Converter
Please, only use the original files (bf16/fp16, not fp8, fine-tuned/merged or not) from comfyui. https://huggingface.co/Comfy-Org
# Mise à jours:
- Base support Z-Image-Turbo
- Ajout du support pour Flux.1-dev (Philippe Joye)
- Ajout du support pour Flux.1-Fill
- Ajout du support pour Qwen-image-edit 2511 (Merci Philippe)
- Ajout du support pour Qwen-image 2512
- Ajout du support pour Flux.2-dev
- Ajout du support pour Wan2.2-i2v-high-low
- Ajout du support pour Z-Image-Base
- Ajout du support pour Ltx-2-19b  use the dev or distilled version (not fp8) https://huggingface.co/Lightricks/LTX-2/tree/main
- Ajout du support pour Flux.2-klein-9b
---
Un nœud ComfyUI haute performance pour convertir vos modèles en NVFP4. Basculez entre les architectures Z-Image, Flux.1, Flux.1-Fill, Qwen-image-edit 2511, Qwen-image 2512 et flux.2-dev et plus en un clic et profitez de la puissance des Tensor Cores.

Ce format permet de diviser la taille des modèles par 3.5 tout en conservant une qualité quasi identique au BF16, tout en profitant des **Tensor Cores** des cartes NVIDIA récentes.

<img width="1139" height="709" alt="image" src="https://github.com/user-attachments/assets/5edd8897-5ad3-4c44-b6b2-9b8fb2b8f63e" />



## 🛠️ Installation

1. **Prérequis** :
Assurez-vous d'avoir installé la bibliothèque `comfy-kitchen` dans l'environnement Python de votre ComfyUI :
```bash
pip install comfy-kitchen

```


2. **Installation du nœud** :
Allez dans votre dossier `custom_nodes` et clonez ce dépôt (ou via manager) :
```bash
cd custom_nodes
git clone https://github.com/tritant/ComfyUI_ZimageTurbo_nvfp4_Converter.git
```


3. **Redémarrez ComfyUI**.

## 📖 Utilisation

1. Cherchez le nœud **🍳 Kitchen NVFP4 Converter** dans la catégorie `Kitchen`.
2. Sélectionnez votre modèle source dans la liste `model_name`.
3. Choisissez un nom pour le fichier de sortie (ex: `mon_modele_nvfp4`).
4. Réglez le `device` sur **cuda** pour une vitesse maximale.
5. Appuyez sur **Queue Prompt**.
---
