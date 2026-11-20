import base64
import io
import json
import threading
from pathlib import Path

import gradio as gr
import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

matplotlib.use("Agg")

# ZeroGPU Spaces require at least one @spaces.GPU function; elsewhere this is a no-op.
try:
    import spaces

    gpu = spaces.GPU(duration=8)
except ImportError:
    def gpu(fn):
        return fn

BASE = Path(__file__).parent
LABELS = json.loads((BASE / "labels.json").read_text())
META = json.loads((BASE / "model_meta.json").read_text())
TREAT = json.loads((BASE / "treatments.json").read_text())
IMG = META["img_size"]
CONF_MIN = 0.70  # below this we do not show a diagnosis

missing = [l for l in LABELS if l not in TREAT]
if missing:
    raise RuntimeError(f"treatments.json is missing entries for: {missing}")

torch.set_num_threads(2)
model = models.efficientnet_b0(weights=None)
model.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.classifier[1].in_features, META["num_classes"]))
model.load_state_dict(torch.load(BASE / "plant_disease_effnetb0.pth", map_location="cpu"))
model.eval()

TF = transforms.Compose([
    transforms.Resize((IMG, IMG)),
    transforms.ToTensor(),
    transforms.Normalize(META["mean"], META["std"]),
])
LOCK = threading.Lock()  # Grad-CAM hooks touch shared model state, so serialize requests

DISCLAIMER = (
    "*This tool is a student project trained on the PlantVillage dataset (single leaves, plain background). "
    "It can be wrong on field photos and only knows the 15 classes listed in the model card. Treatment notes are "
    "general extension-service guidance, not a prescription: check product labels and local agricultural advice "
    "before applying any chemical.*"
)


def display_name(label):
    t = TREAT[label]
    return f"{t['crop']}: {t['name']}"


def forward_with_cam(x):
    feats, grads = {}, {}
    layer = model.features[-1]

    def fwd(module, inp, out):
        feats["a"] = out
        out.register_hook(lambda g: grads.__setitem__("g", g))

    handle = layer.register_forward_hook(fwd)
    try:
        model.zero_grad()
        with torch.enable_grad():
            logits = model(x)
            probs = F.softmax(logits, dim=1)[0]
            top = int(probs.argmax())
            logits[0, top].backward()
    finally:
        handle.remove()
    w = grads["g"].mean((2, 3), keepdim=True)
    cam = F.relu((w * feats["a"]).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return probs.detach(), top, cam.detach().numpy()


def overlay(img, cam, alpha=0.45):
    base = np.asarray(img.resize((IMG, IMG)), dtype=np.float32) / 255.0
    heat = matplotlib.colormaps["jet"](cam)[..., :3]
    out = (1 - alpha) * base + alpha * heat
    return Image.fromarray((out * 255).astype(np.uint8))


def heat_layer(cam, floor=0.4):
    """Transparent RGBA Grad-CAM layer: low-attention pixels are fully see-through,
    so the front end can lay it over the photo without colouring the whole image."""
    rgba = matplotlib.colormaps["jet"](cam)
    rgba[..., 3] = np.clip((cam - floor) / (1 - floor), 0, 1) ** 0.8 * 0.9
    return Image.fromarray((rgba * 255).astype(np.uint8), "RGBA")


def bullets(items):
    return [f"- {s}" for s in items]


def build_report(label, conf):
    t = TREAT[label]
    parts = [f"## {t['crop']}: {t['name']}", f"**Confidence:** {conf:.1%} &nbsp;|&nbsp; **Type:** {t['kind']}", "", t["summary"]]
    if t["symptoms"]:
        parts += ["", "**What to look for**"] + bullets(t["symptoms"])
    parts += ["", f"**Is there a cure?** {t['cure_note']}"]
    parts += ["", "**What to do now**"] + bullets(t["treatment"])
    parts += ["", "**Prevention**"] + bullets(t["prevention"])
    if t["sources"]:
        parts += ["", "**Sources**"] + [f"- [{s['title']}]({s['url']})" for s in t["sources"]]
    return "\n".join(parts)


@gpu
def predict(img):
    if img is None:
        return None, None, "Upload a leaf photo first."
    img = img.convert("RGB")
    x = TF(img).unsqueeze(0)
    with LOCK:
        probs, top, cam = forward_with_cam(x)
    k = torch.topk(probs, 3)
    scores = {display_name(LABELS[int(i)]): float(p) for p, i in zip(k.values, k.indices)}
    conf = float(probs[top])
    heat = overlay(img, cam)
    if conf < CONF_MIN:
        msg = (
            f"### Not sure ({conf:.0%} confidence)\n"
            f"Best guess: **{display_name(LABELS[top])}**, but that is too uncertain to act on.\n\n"
            "Retake the photo: one leaf, close up, in daylight, in focus, plain background. "
            "If it still looks wrong, ask a local agriculture extension office."
        )
        return scores, heat, msg
    return scores, heat, build_report(LABELS[top], conf)


def to_data_url(pil, fmt="JPEG"):
    buf = io.BytesIO()
    if fmt == "PNG":
        pil.save(buf, format="PNG")
    else:
        pil.save(buf, format="JPEG", quality=85)
    return f"data:image/{fmt.lower()};base64," + base64.b64encode(buf.getvalue()).decode()


@gpu
def analyze(img):
    """Structured result for the custom front end (served at /analyze)."""
    if img is None:
        return {"error": "No image received."}
    img = img.convert("RGB")
    x = TF(img).unsqueeze(0)
    with LOCK:
        probs, top, cam = forward_with_cam(x)
    k = torch.topk(probs, 3)
    conf = float(probs[top])
    confident = conf >= CONF_MIN
    return {
        "confidence": conf,
        "confident": confident,
        "threshold": CONF_MIN,
        "top": [
            {"label": LABELS[int(i)], "name": display_name(LABELS[int(i)]), "p": float(p)}
            for p, i in zip(k.values, k.indices)
        ],
        "heatmap": to_data_url(overlay(img, cam)),
        "heat_layer": to_data_url(heat_layer(cam), "PNG"),
        "info": TREAT[LABELS[top]] if confident else None,
    }


with gr.Blocks(title="Plant Leaf Disease Detector") as demo:
    gr.Markdown(
        "# Plant Leaf Disease Detector\n"
        "Upload a clear photo of a single **tomato, potato or bell pepper** leaf. "
        "You get the likely disease, a Grad-CAM heatmap of where the model looked, and guidance on what to do."
    )
    with gr.Row():
        with gr.Column():
            inp = gr.Image(type="pil", label="Leaf photo", sources=["upload", "webcam"])
            btn = gr.Button("Analyze", variant="primary")
        with gr.Column():
            lab = gr.Label(num_top_classes=3, label="Top predictions")
            cam_out = gr.Image(label="Where the model looked (Grad-CAM)")
    report = gr.Markdown()
    btn.click(predict, inputs=inp, outputs=[lab, cam_out, report])
    gr.Markdown(DISCLAIMER)

    # hidden JSON endpoint used by the custom front end (api_name="analyze")
    api_in = gr.Image(type="pil", visible=False)
    api_out = gr.JSON(visible=False)
    api_btn = gr.Button(visible=False)
    api_btn.click(analyze, inputs=api_in, outputs=api_out, api_name="analyze")

demo.queue(max_size=8)

if __name__ == "__main__":
    demo.launch(ssr_mode=False)
