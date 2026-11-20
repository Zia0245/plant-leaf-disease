# Plant Leaf Disease Detection

Classifies **tomato, potato and bell pepper** leaves into 15 classes (healthy + disease), shows a Grad-CAM heatmap of where the model looked, and gives treatment guidance. It works only on those three crops.

**Live:** model Space [`zia2/plant-leaf-disease`](https://huggingface.co/spaces/zia2/plant-leaf-disease) · front end [`zia2/Leaf-doctor`](https://huggingface.co/spaces/zia2/Leaf-doctor)

## Results

EfficientNet-B0 (ImageNet-pretrained), PlantVillage, stratified 70/15/15 split, seed 42.
Test accuracy **99.64%**, macro-F1 **0.9968** on 3,094 held-out images.

> PlantVillage has many near-identical photos of the same leaf and plain backgrounds, so these scores are optimistic. Expect lower accuracy on field photos.

## Repo layout

| Path | What |
|---|---|
| `model_space/` | Gradio app (Hugging Face Space): inference, Grad-CAM (full + transparent overlay layer), `treatments.json` |
| `frontend/` | Static dashboard (single `index.html`): heatmap overlay, compare view, light/dark/system theme, PDF report with the leaf image |
| `notebooks/` | `train_full.ipynb` (end-to-end Colab training), `explore.ipynb`, `train_quick.ipynb` |

## Pipeline

- **Data:** PlantVillage (`emmarex/plantdisease`, Kaggle), 20,638 images; 14 exact duplicates removed.
- **Training:** AdamW, one-cycle LR, class-weighted loss with label smoothing, 8 epochs.
- **Explainability:** Grad-CAM, rendered as a transparent layer so the heatmap does not wash out the leaf.
- **Treatments:** short summaries paraphrased from university extension publications, with source links. General information, not a prescription.

## Run locally

```bash
cd model_space
pip install -r requirements.txt gradio
python app.py
```

The weights file is not in this repo; train with `notebooks/train_full.ipynb` or download it from the Space.
Open `frontend/index.html?demo=ok` for an offline preview that does not call the model.

## Roadmap

User accounts and scan history (Firebase), more crops, an in-browser ONNX version to avoid ZeroGPU quota limits.

## License

MIT
