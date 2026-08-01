# ComfyUI Kitchen Quant Converter

Please, only use the original files (bf16/fp16, not fp8, fine-tuned/merged or not) from comfyui. https://huggingface.co/Comfy-Org

# Mise à jour :

## v2.0.0 — Multi-formats
- **Nouveaux formats de quantification** : le nœud ne fait plus uniquement du NVFP4. Un menu `quant_format` permet désormais de choisir entre **NVFP4**, **MXFP8**, **INT8_CONVROT** et **INT4_CONVROT**.
- **Nom de sortie automatique** : le suffixe du format est ajouté au fichier (`mon_modele_int8_convrot.safetensors`). Champ laissé vide = le nom du modèle source est repris. Plus besoin de renommer entre deux conversions.
- **Nouveaux profils** : ACE-Step, Anima, Boogu-Image, Chroma, ERNIE-Image, Ideogram-4, SeedVR.
- Message d'erreur explicite si un format n'est pas disponible dans la version installée de comfy-kitchen.

## Historique
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
- Ajout du support pour Krea2 turbo

---

Un nœud ComfyUI haute performance pour quantifier vos modèles de diffusion. Basculez entre les architectures Z-Image, Flux.1, Flux.2, Qwen-Image, Wan, LTX-2, Krea 2, Chroma, Ideogram 4 et plus en un clic, et choisissez le format adapté à votre carte.

Les couches sensibles de chaque architecture (embeddings, normalisations, projections d'entrée/sortie) sont conservées en BF16 selon un profil dédié, ce qui préserve la qualité là où la quantification ferait le plus de dégâts.

<img width="1139" height="709" alt="image" src="https://github.com/user-attachments/assets/5edd8897-5ad3-4c44-b6b2-9b8fb2b8f63e" />

## 📦 Quel format choisir ?

| Format | Taille | Support matériel |
|---|---|---|
| **NVFP4** | ÷ 3,5 | Natif sur Blackwell (RTX 50). Émulé ailleurs. |
| **MXFP8** | ÷ 2 | Natif sur Blackwell (RTX 50). Émulé ailleurs. |
| **INT8_CONVROT** | ÷ 2 | Natif depuis Turing (RTX 20 et plus). |
| **INT4_CONVROT** | ÷ 3,5 | Natif depuis Turing (les poids 4 bits sont déballés vers les unités INT8). |

Sur une carte antérieure aux RTX 50, **INT8_CONVROT** et **INT4_CONVROT** sont les seuls formats accélérés matériellement. NVFP4 et MXFP8 réduisent bien la taille du fichier mais passent par des noyaux émulés, donc plus lents.

Les formats **ConvRot** appliquent une rotation de Hadamard par groupes avant quantification : les valeurs aberrantes sont réparties sur 256 canaux au lieu d'écraser la plage utile, ce qui améliore nettement la qualité, en particulier en INT4.

Au démarrage, ComfyUI indique les formats accélérés sur votre machine :

```
Native ops: convrot_w4a4, int8_tensorwise, float8_e4m3fn ... emulated ops: mxfp8, nvfp4
```

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
git clone https://github.com/tritant/ComfyUI_Kitchen_nvfp4_Converter.git
```

3. **Redémarrez ComfyUI**.

## 📖 Utilisation

1. Cherchez le nœud **🍳 Kitchen Quant Converter** dans la catégorie `Kitchen`.
2. Sélectionnez votre modèle source dans la liste `model_name`.
3. Sélectionnez l'architecture correspondante dans `model_type` (le profil détermine les couches à préserver).
4. Choisissez le format dans `quant_format`.
5. `output_filename` : laissez vide pour reprendre le nom du modèle source, le suffixe du format est ajouté automatiquement.
6. Réglez le `device` sur **cuda** pour une vitesse maximale.
7. Appuyez sur **Queue Prompt**.

Le fichier converti est écrit dans le même dossier que le modèle source.

---
