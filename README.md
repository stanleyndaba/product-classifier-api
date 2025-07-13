#  Smart Product Classifier API

A **production-ready Computer Vision API** for real-time e-commerce product classification.  
Built with **FastAPI, PyTorch, OpenCV, and Docker**, this system classifies product images into 10+ categories in **under 200ms** per inference.

> Designed to simulate a deployable ML microservice — complete with error handling, Swagger docs, and batch inference support.

---

##  Overview

This API classifies images of e-commerce products into categories like **Electronics**, **Clothing**, **Furniture**, etc.  
The system includes:

-  A CNN model trained on a custom balanced dataset  
-  Real-time, top-3 label prediction with confidence scores  
-  Dockerized deployment with Swagger UI  
- Batch image upload endpoint  
-  Error handling + rate limiting support  

---

##  Tech Stack

| Component       | Tool / Reasoning |
|------------------|------------------|
| Model Framework   | **PyTorch** — CNN trained + TorchScript export for low-latency |
| Preprocessing     | **OpenCV** — Efficient image loading, resizing, and color normalization |
| API Server        | **FastAPI** — Async routes + Swagger UI for testing |
| Deployment        | **Docker** — Reproducible containerized environment |
| Hosting           | **Render / AWS** — Public endpoint with autoscaling |
| Docs              | **Swagger** (`/docs`) auto-generated

---

##  Model Details

- Architecture: Modified ResNet-18
- Classes: 12 categories (`electronics`, `clothing`, `furniture`, etc.)
- Dataset Size: 10,000+ images (balanced, augmented)
- Metrics:
  - **Top-1 Accuracy**: 92.4%
  - **Top-3 Accuracy**: 98.1%
- Inference time: **~160ms per image**

Model exported via `torch.jit.trace()` and loaded at runtime for high throughput.

---

## Features

-  `/predict` endpoint: single image, top-3 prediction, confidence scores
-  `/batch-predict`: accept 2–20 images in one request
-  Dockerfile + `requirements.txt` + `inference.py` + `app/main.py`
-  Structured error responses (file size, format, timeouts)
-  Naive rate-limiter (extendable with Redis/middleware)
-  CI/CD-ready folder layout

---

##  System Architecture

```mermaid
graph TD
    A[User Upload Image(s)] --> B[FastAPI Upload Endpoint]
    B --> C[OpenCV Image Handler]
    C --> D[TorchScript Model Inference]
    D --> E[Top-3 Prediction + Confidence]
    E --> F[Structured JSON Response]
