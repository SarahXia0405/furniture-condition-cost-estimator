# Furniture Condition & Cost Estimator

Author: Qinghua (Sarah) Xia

GitHub: @SarahXia0405

This project is an end-to-end computer vision pipeline that estimates the **repair / replacement cost** of furniture from a single **room photograph**.

Given an input image (e.g., a living room), the system will:

1. Detect furniture objects (TV, couch, chair, bed, plants, etc.).
2. Classify their condition (good vs damaged).
3. Infer material type for upholstered furniture (fabric / leather / wood / metal).
4. Decide whether each item should be **kept, fixed, or replaced**.
5. Estimate the **cost** of the recommended action using simple domain-informed rules.
6. Produce a structured table summarizing all items in the room.

The repository also includes a Streamlit app so that non-technical users can upload room images and immediately see a cost breakdown.

---

## 1. Project Overview

This project was built to support **real-estate and senior-housing CapEx planning**:

- Asset managers can upload unit photos.
- The model turns unstructured images into a **line-item cost estimate**.
- This allows faster portfolio-level budgeting and scenario analysis.

The architecture is intentionally **modular** so each component (detection, condition, material, cost model) can be upgraded independently.

---

## 2. Core Features

- **Object detection** with YOLOv8  
  Detects TVs, couches, chairs, beds, potted plants, and other large furniture.

- **Condition assessment**
  - TV / couch / chair: good vs damaged (EfficientNet variants).
  - Plants: type + health (e.g., replace vs keep).

- **Damage severity & action**
  - TVs: classify **fix** vs **replace**.
  - Couches & chairs: approximate damage severity via bounding-box area; if damage ratio > threshold → replace, else → fix.

- **Material classification**
  - VGG16-based classifier trained on custom patches of **fabric, leather, wood, metal**.
  - Used to estimate more realistic repair / replacement costs.

- **Cost estimation**
  - Rule-based heuristics derived from real retrofitting experience.
  - Different price rules for:
    - TVs (by size & action).
    - Couches / chairs (by material & action).
    - Plants (by type & health).

- **Streamlit web UI**
  - Upload a room image.
  - See detections overlaid on the image.
  - Download a CSV / view a table of all items and estimated costs.

---

## 3. Repository Structure

```text
.
├── app.py                 # Streamlit app entrypoint
├── requirements.txt       # Python dependencies
├── yolov8n.pt             # YOLOv8 detection weights (example small model)
├── models/                # Local directory for additional model weights (ignored by git)
│   ├── tv_condition_efficientnet_b0.pth
│   ├── couch_condition_efficientnet_b0.pth
│   ├── chair_condition_efficientnet_v2s.pth
│   ├── plant_type_efficientnet_b0.pth
│   ├── plant_health_efficientnet_b3.pth
│   ├── tv_action_efficientnet_b0_weighted.pth
└── └── material_vgg16.pth

```

---

## 4. System Design & Architecture

The system is architected as a modular, multi-stage computer-vision pipeline.
Each stage is independent, so you can upgrade or replace components (YOLO → DETR, EfficientNet → ConvNeXt, VGG → ViT, etc.) without changing the rest of the system.


```text
                        ┌──────────────────────────┐
                        │     Input Room Image     │
                        └──────────────┬───────────┘
                                       │
                                       ▼
                    ┌──────────────────────────────────┐
                    │   Stage 1 — Object Detection     │
                    │     YOLOv8 (TV, Couch, Chair,    │
                    │        Bed, Plant, etc.)         │
                    └──────────────┬───────────────────┘
                                   │   Detected crops
              ┌────────────────────┼────────────────────┐
              ▼                    ▼                    ▼
┌────────────────────┐  ┌──────────────────────┐  ┌─────────────────────┐
│       TV Crop      │  │ Couch / Chair Crop   │  │     Plant Crop      │
└─────────┬──────────┘  └───────────┬─────────-┘  └─────────┬───────────┘
          │                         │                       │
          ▼                         ▼                       ▼
┌──────────────────-─┐      ┌────────────────--──┐     ┌─────────────────────┐
│ TV Condition       │      │ Furniture Condition│     │ Plant Type Model    │
│ EfficientNet-B0    │      │ EffNet-B0/V2-S     │     │ EfficientNet-B0     │
└─────────┬──────────┘      └──────────┬─────────┘     └──────────-┬─────────┘
          │ damaged                    │ damaged                   │
          ▼                            ▼                            ▼
┌──────────────────-─┐       ┌──────────────────────────┐  ┌────────────────────-┐
│ TV Action Model    │       │ Material Classifier      │  │ Plant Health Model  │
│ fix / replace      │       │ VGG16 (fab/leath/wood...)│  │ EfficientNet-B3     │
└─────────┬──────────┘       └──────────────┬───────────┘  └──────────┬────────-─┘
          │                                 │                         │
          ▼                                 ▼                         ▼
┌─────────────────-─┐               ┌──────────────────-─┐     ┌──────────────────┐
│ TV Cost Model     │               │ Furniture Cost     │     │ Plant Cost Model │
│ (size-based)      │               │ (material-based)   │     │ (type-based)     │
└─────────┬─────────┘               └─────────┬────────-─┘     └─────────┬────────┘
          │                                   │                          │
          └──────────────────┬────────────────┴──────────────┬─────----──┘
                             ▼                               ▼
                  ┌──────────────────────────┐     ┌──────────────────────────┐
                  │ Aggregation & Formatting │     │  Streamlit Visualization │
                  │  (CSV, per-room totals)  │     │ (final tables, overlays) │
                  └──────────────┬───────────┘     └──────────────────────────┘
                                 ▼
                     ┌──────────────────────────┐
                     │ Final Cost Report Output │
                     │ (Itemized + room totals) │
                     └──────────────────────────┘

```

### 4.2 Data Flow

```text
Room → YOLOv8 → Crops → Condition Model → Material Model → Action Decision → Cost Estimate → Aggregated Report
```
---

