# ComfyUI Kitchen NVFP4 Converter

# Mise à jours:
- Base support Z-Image-Turbo
- Ajout du support pour Flux.1 (Philippe Joye)
- Ajout du support pour Flux.1 Fill
- Ajout du support pour Qwen-image-edit 2511 (Merci Philippe)
- Ajout du support pour Qwen-image 2512
- Ajout du support pour Flux.2-dev
---
Un nœud ComfyUI haute performance pour convertir vos modèles en NVFP4. Basculez entre les architectures Z-Image, Flux.1, Flux.1-Fill, Qwen-image-edit 2511 en un clic et profitez de la puissance des Tensor Cores.

Ce format permet de diviser la taille des modèles par 3.5 tout en conservant une qualité quasi identique au BF16, tout en profitant des **Tensor Cores** des cartes NVIDIA récentes.

<img width="831" height="553" alt="Capture d&#39;écran 2026-01-23 102008" src="https://github.com/user-attachments/assets/fb4948b6-d4c3-4612-9f21-ee27635d63c0" />



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
