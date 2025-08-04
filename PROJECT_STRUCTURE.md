# Smart Product Classifier - Project Structure

The repository has been successfully restructured according to production best practices. Here's the complete directory layout:

```
smart_product_classifier/
│── README.md                # Project overview, setup instructions
│── requirements.txt          # Python dependencies
│── .env.example              # Environment variables template
│── setup.py                  # Optional for packaging
│── docker-compose.yml        # Multi-service orchestration
│── Dockerfile                # Container image for API
│── Makefile                  # Commands for build, test, deploy
│
├── config/                   # Configurations for dev/prod
│   ├── config.yaml
│   └── logging.conf
│
├── data/                     # Raw & processed datasets
│   ├── raw/
│   │   ├── train/           # Training data by category
│   │   ├── val/             # Validation data by category
│   │   └── test/            # Test data by category
│   ├── processed/           # Processed datasets
│   └── dataloader.py        # Data loading pipeline
│
├── notebooks/                # Jupyter notebooks for EDA & experiments
│   ├── eda.ipynb
│   └── model_experiments.ipynb
│
├── src/
│   ├── __init__.py
│   │
│   ├── training/             # Model training pipeline
│   │   ├── train.py          # Main training entry
│   │   ├── dataset.py        # Custom PyTorch Dataset
│   │   ├── transforms.py     # Data augmentation
│   │   ├── model_builder.py  # CNN architectures
│   │   ├── optimizer.py      # Optimizer & scheduler config
│   │   └── metrics.py        # Accuracy, precision, etc.
│   │
│   ├── inference/            # Inference scripts
│   │   ├── predict.py        # Standalone script for testing predictions
│   │   ├── batch_predict.py  # Batch inference
│   │   └── model_loader.py   # Loads TorchScript/ONNX models
│   │
│   ├── services/             # Core API services
│   │   ├── __init__.py
│   │   ├── api.py            # FastAPI endpoints
│   │   ├── schemas.py        # Pydantic models for request/response
│   │   ├── rate_limiter.py   # Middleware for rate limiting
│   │   └── error_handler.py  # Unified error responses
│   │
│   ├── utils/                # Utility scripts
│   │   ├── __init__.py
│   │   ├── logger.py
│   │   ├── config_parser.py
│   │   ├── image_utils.py    # OpenCV image preprocessing
│   │   └── timer.py
│   │
│   └── tests/                # Unit & integration tests
│       ├── test_api.py
│       ├── test_model.py
│       └── test_utils.py
│
├── saved_models/             # Checkpoints & final models
│   ├── README.md
│   ├── best_model.pth       # (to be added)
│   └── model.onnx           # (to be added)
│
└── frontend_demo/            # Lightweight demo app
    ├── index.html
    ├── app.js
    └── styles.css
```

## Key Features Implemented

### 🏗️ Production Architecture
- ✅ Modular directory structure
- ✅ Configuration management (YAML + environment variables)
- ✅ Docker containerization with multi-service orchestration
- ✅ Comprehensive logging setup
- ✅ Error handling and rate limiting

### 📊 Data Pipeline
- ✅ Custom PyTorch Dataset with data loading utilities
- ✅ Image preprocessing with OpenCV and PIL
- ✅ Data augmentation and transformation pipelines
- ✅ Support for multiple image formats

### 🤖 ML Infrastructure
- ✅ Training pipeline framework
- ✅ Model architecture builders
- ✅ Metrics tracking and evaluation
- ✅ Inference optimization with async processing

### 🌐 API Services
- ✅ FastAPI application with async endpoints
- ✅ Pydantic schema validation
- ✅ Rate limiting with Redis support
- ✅ Comprehensive error handling
- ✅ Health checks and monitoring

### 🎨 Frontend Demo
- ✅ Modern, responsive web interface
- ✅ Drag-and-drop file upload
- ✅ URL-based image classification
- ✅ Real-time results visualization
- ✅ Beautiful CSS animations and design

### 🔧 Development Tools
- ✅ Makefile with common commands
- ✅ Package setup with entry points
- ✅ Environment configuration template
- ✅ Jupyter notebooks for experimentation

### 🐳 Deployment Ready
- ✅ Docker containers for API and services
- ✅ Docker Compose for multi-service deployment
- ✅ Redis for caching and rate limiting
- ✅ Nginx configuration support

## Quick Start

1. **Install dependencies:**
   ```bash
   make install
   ```

2. **Run the API locally:**
   ```bash
   make run
   ```

3. **Deploy with Docker:**
   ```bash
   make deploy-local
   ```

4. **Train a model:**
   ```bash
   make train
   ```

5. **Run tests:**
   ```bash
   make test
   ```

## API Endpoints

- **POST /predict** - Single image classification
- **POST /predict/batch** - Batch image classification
- **POST /predict/url** - Classify image from URL
- **GET /health** - Service health check
- **GET /model/info** - Model information
- **GET /docs** - Interactive API documentation

## Model Categories

The classifier supports 12 product categories:
- Electronics
- Clothing
- Furniture
- Books
- Toys
- Sports
- Beauty
- Home & Garden
- Automotive
- Health
- Food
- Jewelry

## Technology Stack

- **ML Framework:** PyTorch + TorchVision
- **API:** FastAPI + Uvicorn
- **Image Processing:** OpenCV + PIL
- **Data Validation:** Pydantic
- **Caching:** Redis
- **Containerization:** Docker + Docker Compose
- **Frontend:** Vanilla HTML/CSS/JS with modern design
- **Configuration:** YAML + Environment Variables

This structure provides a solid foundation for a production-ready computer vision API with room for scalability and extension.