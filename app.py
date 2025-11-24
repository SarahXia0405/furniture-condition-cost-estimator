import os
from pathlib import Path
from io import BytesIO

import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
import pandas as pd
from PIL import Image

import streamlit as st
from ultralytics import YOLO

# -------------------------------
# 0. Paths & global config
# -------------------------------
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model weight paths (adjust names if yours differ)
PLANT_TYPE_MODEL_PATH   = MODELS_DIR / "plant_type_efficientnet_b0.pth"
PLANT_HEALTH_MODEL_PATH = MODELS_DIR / "plant_health_efficientnet_b3.pth"

TV_COND_MODEL_PATH      = MODELS_DIR / "tv_condition_efficientnet_b0.pth"
COUCH_COND_MODEL_PATH   = MODELS_DIR / "couch_condition_efficientnet_b0.pth"
CHAIR_COND_MODEL_PATH   = MODELS_DIR / "chair_condition_efficientnet_v2s.pth"

TV_ACTION_MODEL_PATH    = MODELS_DIR / "tv_action_efficientnet_b0_weighted.pth"

MATERIAL_MODEL_PATH     = MODELS_DIR / "material_vgg16.pth"

# YOLOv8 COCO model (pretrained)
YOLO_MODEL_NAME = "yolov8n.pt"   # small & fast; you can change to yolov8s.pt etc.

# COCO class ids for our furniture types
COCO_FURNITURE_CLASSES = {
    56: "chair",        # 'chair'
    57: "couch",        # 'couch'
    58: "potted plant", # 'potted plant'
    62: "tv",           # 'tvmonitor'
    65: "bed",          # 'bed'
}

# -------------------------------
# 1. Transforms
# -------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def make_efficientnet_transform(input_size=224):
    return transforms.Compose([
        transforms.Resize(int(input_size * 1.15)),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

effnet_tf = make_efficientnet_transform(224)

material_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# -------------------------------
# 2. Model loading helpers
# -------------------------------
def load_efficientnet_b0(num_classes, weights_path: Path):
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model

def load_efficientnet_b3(num_classes, weights_path: Path):
    model = models.efficientnet_b3(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model

def load_efficientnet_v2s(num_classes, weights_path: Path):
    model = models.efficientnet_v2_s(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model

def load_vgg16_material(num_classes, weights_path: Path):
    base = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_FEATURES)
    for p in base.parameters():
        p.requires_grad = False
    base.classifier = nn.Sequential(
        nn.Linear(512 * 7 * 7, 4096),
        nn.ReLU(True),
        nn.Dropout(0.5),
        nn.Linear(4096, 4096),
        nn.ReLU(True),
        nn.Dropout(0.5),
        nn.Linear(4096, num_classes),
    )
    state = torch.load(weights_path, map_location=device)
    base.load_state_dict(state)
    base.to(device)
    base.eval()
    return base

def predict_single_image(model, img: Image.Image, transform, class_names):
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
    idx = int(torch.argmax(probs).item())
    return class_names[idx], float(probs[idx].item())

# -------------------------------
# 3. Cost heuristics
# -------------------------------
PLANT_TYPE_CLASSES   = ["flowering_plant", "large_foliage", "ornamental_grass", "small_foliage"]
PLANT_HEALTH_CLASSES = ["keep", "replace"]

COND_CLASSES         = ["damaged", "good"]
TV_ACTION_CLASSES    = ["fix", "replace"]
MATERIAL_CLASSES     = ["fabric", "leather", "metal", "wood"]

PLANT_COST_TABLE = {
    "flowering_plant":   25.0,
    "small_foliage":     20.0,
    "large_foliage":     80.0,
    "ornamental_grass":  30.0,
    "other":             30.0,
}

def plant_estimated_cost(plant_type, health_label):
    if health_label != "replace":
        return 0.0
    return float(PLANT_COST_TABLE.get(plant_type, 30.0))

def estimate_tv_size_bucket(box_width_px, image_width_px):
    # crude heuristic
    rel = box_width_px / float(image_width_px)
    # small relative width -> small TV, bigger -> large TV
    if rel < 0.25:
        return '32"'
    else:
        return '43"'

def estimate_tv_cost(action, size_bucket):
    if action == "fix":
        return 120.0
    if size_bucket == '32"':
        return 150.0
    else:
        return 250.0

def estimate_furniture_cost(furniture_type, material, decision):
    if decision == "keep":
        return 0.0

    if furniture_type == "couch":
        if material == "leather":
            return 1200.0 if decision == "replace" else 250.0
        elif material == "fabric":
            return 500.0 if decision == "replace" else 150.0
        elif material == "wood":
            return 300.0 if decision == "replace" else 100.0
        elif material == "metal":
            return 150.0 if decision == "replace" else 80.0
        else:
            return 500.0 if decision == "replace" else 150.0

    if furniture_type == "chair":
        if material == "leather":
            return 400.0 if decision == "replace" else 200.0
        elif material == "fabric":
            return 250.0 if decision == "replace" else 100.0
        elif material == "wood":
            return 200.0 if decision == "replace" else 80.0
        elif material == "metal":
            return 150.0 if decision == "replace" else 80.0
        else:
            return 250.0 if decision == "replace" else 100.0

    return 0.0

def estimate_damage_ratio_from_box(box_area, img_area):
    """
    Simple proxy: proportion of image area.
    > 0.25 of the image => big object => treat as "heavy damage".
    """
    if img_area <= 0:
        return 0.5
    ratio = box_area / float(img_area)
    return float(np.clip(ratio / 0.5, 0.0, 1.0))  # scaled so ~0.5 area ~ ratio 1.0

# -------------------------------
# 4. Load all models once
# -------------------------------
@st.cache_resource
def load_all_models():
    # detection model
    yolo = YOLO(YOLO_MODEL_NAME)

    plant_type_model   = load_efficientnet_b0(len(PLANT_TYPE_CLASSES),   PLANT_TYPE_MODEL_PATH)
    plant_health_model = load_efficientnet_b3(len(PLANT_HEALTH_CLASSES), PLANT_HEALTH_MODEL_PATH)

    tv_cond_model      = load_efficientnet_b0(len(COND_CLASSES), TV_COND_MODEL_PATH)
    couch_cond_model   = load_efficientnet_b0(len(COND_CLASSES), COUCH_COND_MODEL_PATH)
    chair_cond_model   = load_efficientnet_v2s(len(COND_CLASSES), CHAIR_COND_MODEL_PATH)

    tv_action_model    = load_efficientnet_b0(len(TV_ACTION_CLASSES), TV_ACTION_MODEL_PATH)

    material_model     = load_vgg16_material(len(MATERIAL_CLASSES), MATERIAL_MODEL_PATH)

    return {
        "yolo": yolo,
        "plant_type": plant_type_model,
        "plant_health": plant_health_model,
        "tv_cond": tv_cond_model,
        "couch_cond": couch_cond_model,
        "chair_cond": chair_cond_model,
        "tv_action": tv_action_model,
        "material": material_model,
    }

# -------------------------------
# 5. Core: analyze one uploaded image
# -------------------------------
def analyze_uploaded_image(img_pil: Image.Image, models_dict):
    """
    Runs YOLO to detect furniture, then classification+cost per crop.
    Returns a pandas DataFrame with row per detected furniture item.
    """
    yolo = models_dict["yolo"]
    plant_type_model   = models_dict["plant_type"]
    plant_health_model = models_dict["plant_health"]
    tv_cond_model      = models_dict["tv_cond"]
    couch_cond_model   = models_dict["couch_cond"]
    chair_cond_model   = models_dict["chair_cond"]
    tv_action_model    = models_dict["tv_action"]
    material_model     = models_dict["material"]

    img_w, img_h = img_pil.size
    img_area = img_w * img_h

    # YOLO expects numpy / path; easiest is to convert PIL->np
    results = yolo(img_pil, verbose=False)

    records = []
    for r in results:
        boxes = r.boxes
        if boxes is None:
            continue

        for box in boxes:
            cls_id = int(box.cls.item())
            if cls_id not in COCO_FURNITURE_CLASSES:
                continue

            class_name = COCO_FURNITURE_CLASSES[cls_id]
            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, xyxy)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_w, x2), min(img_h, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            crop = img_pil.crop((x1, y1, x2, y2))
            width = x2 - x1
            height = y2 - y1
            area = width * height

            subtype = None
            condition = None
            action = None
            material = None
            size_bucket = None
            estimated_cost = 0.0
            damage_ratio = None

            type_conf = None
            health_conf = None
            cond_conf = None
            action_conf = None
            material_conf = None

            # ---------- POTTED PLANT ----------
            if class_name == "potted plant":
                # type
                plant_type, type_conf = predict_single_image(
                    plant_type_model, crop, effnet_tf, PLANT_TYPE_CLASSES
                )
                if type_conf <= 0.45:
                    plant_type = "other"

                # health
                health_label, health_conf = predict_single_image(
                    plant_health_model, crop, effnet_tf, PLANT_HEALTH_CLASSES
                )
                if health_conf < 0.70:
                    health_label = "TBD"

                estimated_cost = plant_estimated_cost(plant_type, health_label)

                subtype = plant_type
                condition = health_label
                action = health_label

            # ---------- TV ----------
            elif class_name == "tv":
                cond_label, cond_conf = predict_single_image(
                    tv_cond_model, crop, effnet_tf, COND_CLASSES
                )
                if cond_label == "good":
                    condition = "good"
                    action = "keep"
                    estimated_cost = 0.0
                else:
                    condition = "damaged"
                    action_label, action_conf = predict_single_image(
                        tv_action_model, crop, effnet_tf, TV_ACTION_CLASSES
                    )
                    action = action_label
                    size_bucket = estimate_tv_size_bucket(width, img_w)
                    estimated_cost = estimate_tv_cost(action, size_bucket)

            # ---------- COUCH / CHAIR ----------
            elif class_name in ["couch", "chair"]:
                if class_name == "couch":
                    cond_model = couch_cond_model
                else:
                    cond_model = chair_cond_model

                cond_label, cond_conf = predict_single_image(
                    cond_model, crop, effnet_tf, COND_CLASSES
                )
                condition = cond_label

                if cond_label == "good":
                    action = "keep"
                    estimated_cost = 0.0
                else:
                    material_label, material_conf = predict_single_image(
                        material_model, crop, material_tf, MATERIAL_CLASSES
                    )
                    material = material_label
                    damage_ratio = estimate_damage_ratio_from_box(area, img_area)
                    action = "replace" if damage_ratio > 0.5 else "fix"
                    estimated_cost = estimate_furniture_cost(class_name, material, action)

            # ---------- BED ----------
            elif class_name == "bed":
                condition = "TBD"
                action = "TBD"
                estimated_cost = 0.0

            records.append({
                "class_name": class_name,
                "subtype_or_plant_type": subtype,
                "condition": condition,
                "action": action,
                "material": material,
                "size_bucket": size_bucket,
                "damage_ratio_proxy": damage_ratio,
                "estimated_cost": estimated_cost,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "width": width,
                "height": height,
                "area": area,
            })

    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)

# -------------------------------
# 6. Streamlit UI
# -------------------------------
def main():
    st.set_page_config(page_title="Furniture Condition & Cost Estimator", layout="wide")
    st.title("Furniture Condition & Cost Estimator")

    st.markdown(
        """
        Upload a room photo.  
        The app will:
        - Detect **TV, couch, chair, bed, potted plants** (YOLOv8)
        - Classify condition (good/damaged), plant type, TV fix/replace, material
        - Estimate **repair vs replacement costs**
        """
    )

    models_dict = load_all_models()
    st.success("Models loaded on device: {}".format(device))

    uploaded_file = st.file_uploader("Upload a room image", type=["jpg", "jpeg", "png"])
    if uploaded_file is None:
        st.info("Please upload an image to start.")
        return

    # Save file to uploads folder
    img_bytes = uploaded_file.read()
    img_pil = Image.open(BytesIO(img_bytes)).convert("RGB")

    st.subheader("Uploaded image")
    st.image(img_pil, use_column_width=True)

    with st.spinner("Analyzing furniture & costs..."):
        df = analyze_uploaded_image(img_pil, models_dict)

    if df.empty:
        st.warning("No target furniture detected in this image.")
        return

    # Summary metrics
    total_cost = df["estimated_cost"].sum()
    avg_cost = df["estimated_cost"].mean()
    item_count = len(df)

    st.subheader("Cost summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Detected items", f"{item_count}")
    c2.metric("Total estimated cost", f"${total_cost:,.0f}")
    c3.metric(
        "Average cost per item",
        f"${avg_cost:,.0f}" if not np.isnan(avg_cost) else "-"
    )

    # Detailed table
    st.subheader("Item-level details")
    cols_to_show = [
        "class_name",
        "subtype_or_plant_type",
        "condition",
        "action",
        "material",
        "size_bucket",
        "damage_ratio_proxy",
        "estimated_cost",
        "x1", "y1", "x2", "y2",
    ]
    cols_to_show = [c for c in cols_to_show if c in df.columns]
    st.dataframe(
        df[cols_to_show].sort_values("estimated_cost", ascending=False).reset_index(drop=True)
    )

    # Optional: only items needing replacement
    st.subheader("Items needing replacement")
    rep = df[df["action"] == "replace"].copy()
    if rep.empty:
        st.write("No items marked as 'replace'.")
    else:
        st.dataframe(
            rep[cols_to_show].sort_values("estimated_cost", ascending=False).reset_index(drop=True)
        )

if __name__ == "__main__":
    main()
