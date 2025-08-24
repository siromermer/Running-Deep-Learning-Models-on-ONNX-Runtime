# ONNX Runtime Deep Learning Examples

This repository contains comprehensive examples of running deep learning models using ONNX Runtime with GPU acceleration. The examples cover image classification, object detection, and image segmentation tasks.

## 📖 Overview

This project demonstrates how to:
- Export PyTorch models to ONNX format
- Run inference using ONNX Runtime with GPU acceleration
- Visualize results for classification, detection, and segmentation tasks

## 📁 Project Structure

```
onnx-runtime/
├── README.md
├── requirements.txt
|
│
├── notebooks/
│   ├── classification_resnet.ipynb      # Image classification with ResNet-50
│   ├── detection_faster_rcnn.ipynb     # Object detection with Faster R-CNN
│   ├── detection_yolo.ipynb            # Object detection with YOLO
│   └── segmentation_mask_rcnn.ipynb    # Instance segmentation with Mask R-CNN
│
├── models/
│   ├── resnet50.onnx                   # Pre-trained ResNet-50 ONNX model
│   ├── fasterrcnn.onnx                 # Pre-trained Faster R-CNN ONNX model
│   ├── maskrcnn.onnx                   # Pre-trained Mask R-CNN ONNX model
│   ├── yolov8n.onnx                    # Pre-trained YOLOv8 ONNX model
│   └── yolov8n.pt                      # YOLOv8 PyTorch model
│
├── resources/
│   ├── coco_labels_rcnn.txt            # COCO class labels for R-CNN models
│   ├── coco_labels_yolo.txt            # COCO class labels for YOLO
│   ├── imagenet_labels.json            # ImageNet class labels
│   ├── test_image_cat.jpg              # Test image: cat
│   ├── test_image_shark.jpg            # Test image: shark
│   ├── test_video_plane.mp4            # Test video: plane
│   └── test_video_street.mp4           # Test video: street scene
│
└── video_results/
    ├── mask_rcnn_onnx_runtime.mp4      # Mask R-CNN segmentation demo
    ├── faster_rcnn_onnx_runtime.mp4    # Faster R-CNN detection demo
    └── yolo_onnx_runtime.mp4           # YOLO detection demo
```

## 📚 Notebooks Description

### 1. `classification_resnet.ipynb`
**Image Classification with ResNet-50**
- Exports a pre-trained ResNet-50 model to ONNX format
- Performs image classification on test images
- Demonstrates GPU-accelerated inference with ONNX Runtime
- Visualizes top-k predictions with confidence scores

### 2. `detection_faster_rcnn.ipynb`
**Object Detection with Faster R-CNN**
- Exports Faster R-CNN model to ONNX format
- Performs object detection on images and videos
- Draws bounding boxes with class labels and confidence scores
- Processes video files with real-time detection

### 3. `detection_yolo.ipynb`
**Object Detection with YOLO**
- Converts YOLOv8 model to ONNX format
- Implements efficient object detection pipeline
- Processes videos with optimized inference
- Shows FPS counter and detection results

### 4. `segmentation_mask_rcnn.ipynb`
**Instance Segmentation with Mask R-CNN**
- Exports Mask R-CNN model to ONNX format
- Performs instance segmentation on images and videos
- Generates object masks with different colors
- Combines detection and segmentation in video processing

## 🚀 Quick Start

### Prerequisites
- Python 3.10
- NVIDIA GPU with CUDA support
- CUDA 12.8 (or compatible version)
- Conda or pip package manager

### Environment Setup

1. **Create a new conda environment:**
   ```bash
   conda create -n onnxruntime-env python=3.10 -y
   conda activate onnxruntime-env
   ```

2. **Install ONNX Runtime GPU:**
   ```bash
   pip install onnxruntime-gpu>=1.19
   ```

3. **Install all dependencies:**
   ```bash
   pip install -r requirements.txt
   ```


## 🎥 Example Output Videos

The repository includes three demonstration videos in the `video_results/` folder showing the models in action:

### 1. `video_results/yolo_onnx_runtime.mp4`
- **Model:** YOLOv8 ONNX
- **Task:** Real-time object detection
- **Features:** High-speed detection with FPS counter
- **Use Case:** Fast detection for real-time applications

### 2. `video_results/faster_rcnn_onnx_runtime.mp4`
- **Model:** Faster R-CNN ONNX
- **Task:** Accurate object detection
- **Features:** Precise bounding boxes with confidence scores
- **Use Case:** High-accuracy detection applications

### 3. `video_results/mask_rcnn_onnx_runtime.mp4`
- **Model:** Mask R-CNN ONNX
- **Task:** Instance segmentation
- **Features:** Object detection + pixel-level segmentation masks
- **Use Case:** Detailed scene understanding and object segmentation

## 🔧 CUDA Compatibility

This project is tested with:
- **CUDA Version:** 12.8
- **Driver Version:** 570.169
- **GPU:** NVIDIA GeForce GTX 1660 (and compatible)
- **ONNX Runtime:** GPU execution provider

For CUDA compatibility, check the [official ONNX Runtime documentation](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html).

## 🔍 Key Features

- **GPU Acceleration:** All models run with CUDA acceleration through ONNX Runtime
- **Model Export:** Complete examples of PyTorch to ONNX conversion
- **Video Processing:** Real-time inference on video streams
- **Visualization:** Rich visualization of results with bounding boxes, masks, and labels
- **Performance Metrics:** FPS counters and timing information
- **Multiple Tasks:** Classification, detection, and segmentation examples

## 📋 System Requirements

- **OS:** Linux (tested on Ubuntu)
- **Python:** 3.10
- **Memory:** 8GB+ RAM recommended
- **GPU Memory:** 6GB+ VRAM recommended
- **Storage:** 2GB+ for models and dependencies

## 🤝 Usage Notes

1. Ensure your GPU drivers and CUDA installation are compatible
2. Models are automatically downloaded on first run
3. Adjust confidence thresholds and other parameters in the notebooks as needed
4. For CPU-only inference, replace `onnxruntime-gpu` with `onnxruntime` in requirements



