# Selective Search and RCNN Implementation

This repository contains two Python notebooks implementing **Selective Search** and **RCNN (Region-based Convolutional Neural Networks)** for object detection tasks. Below are brief overviews of each implementation.

## 1. Selective Search

### Overview
Selective Search is a popular region proposal algorithm used to identify object-like regions in an image. It generates candidate regions (bounding boxes) that may contain objects, which can later be used by object detection models like RCNN.

### Key Steps
- **Image Preprocessing**: Load and convert the input image from BGR to RGB format.
- **Selective Search**: Perform selective search on the image using different scales and minimum size parameters.
- **Region Proposal Extraction**: Extract bounding boxes (regions) based on the image area, filtering out irrelevant ones.

## 2. RCNN (Region-based Convolutional Neural Networks)

### Overview
RCNN is a popular object detection model that uses region proposals from Selective Search as input and classifies the objects within each proposed region. The model also performs bounding box regression to refine the predicted locations.

### Key Steps
- **Dataset Preparation**: A custom dataset loader is implemented to handle the Open Images Dataset, loading image data and associated bounding boxes.
- **Model Architecture**: The model is based on a pre-trained ResNet50 backbone, with a classification head for object detection and a regression head for bounding box localization.
- **Training and Evaluation**: The model is trained using SGD, with loss functions for both classification and bounding box regression.
- **Prediction and Post-Processing**: Non-Maximum Suppression (NMS) is applied after prediction to filter out overlapping boxes.

## How to Run the Code

### Prerequisites
- Python 3.x
- OpenCV
- SelectiveSearch library
- PyTorch
- TorchVision

### Steps
1. Clone the repository:
    ```bash
    git clone https://github.com/your-repo/rcnn-selective-search.git
    ```
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Run the Selective Search code.
   
4. Run the RCNN code.
   
## Results
- The Selective Search generates region proposals for object detection.
- RCNN model processes these proposals, classifies objects, and localizes them using bounding box regression.


