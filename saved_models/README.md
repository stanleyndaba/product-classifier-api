# Saved Models Directory

This directory contains the trained model files for the Smart Product Classifier.

## Files
- `best_model.pth` - PyTorch model checkpoint (best validation accuracy)
- `model.onnx` - ONNX format for optimized inference
- `model_config.json` - Model configuration and metadata
- `training_history.json` - Training metrics and history

## Usage
Place your trained model files here and update the `MODEL_PATH` in your configuration.

For development, you can download a pre-trained model or train your own using:
```bash
make train
```
