---
title: Plant Leaf Disease Detector
emoji: 🌿
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 5.50.0
app_file: app.py
pinned: false
license: mit
---

# Plant Leaf Disease Detector

EfficientNet-B0 fine-tuned on the PlantVillage dataset (15 classes: tomato, potato and bell pepper leaves).
Upload a leaf photo to get the predicted disease, a Grad-CAM heatmap and treatment guidance.

## Model card

- **Data:** PlantVillage (emmarex/plantdisease on Kaggle), 20,638 images; 14 exact duplicate files removed.
- **Split:** stratified 70 / 15 / 15 (train / val / test), seed 42.
- **Training:** ImageNet-pretrained EfficientNet-B0, AdamW, one-cycle LR, class-weighted loss with label smoothing, 8 epochs.
- **Test result:** accuracy 99.64%, macro-F1 0.9968 on 3,094 held-out images.
- **Caveat:** PlantVillage has many near-identical shots of the same leaf, so held-out scores are optimistic. Expect lower accuracy on field photos with busy backgrounds.

## Treatment guidance

`treatments.json` holds short summaries paraphrased from university extension publications, with source links per disease.
It is general information, not a prescription. Always follow product labels and local agricultural advice.

## Files

`app.py`, `treatments.json`, `requirements.txt`, plus the trained files `plant_disease_effnetb0.pth`, `labels.json` and `model_meta.json`.
