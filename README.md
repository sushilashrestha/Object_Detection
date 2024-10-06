# Selective Search and RCNN Implementation

This repository contains two Python notebooks implementing **Selective Search** and **RCNN (Region-based Convolutional Neural Networks)** for object detection tasks. Below are brief overviews of each implementation.

## 1. Selective Search

### Overview
Selective Search is a popular region proposal algorithm used to identify object-like regions in an image. It generates candidate regions (bounding boxes) that may contain objects, which can later be used by object detection models like RCNN.

### Key Steps
- **Image Preprocessing**: Load and convert the input image from BGR to RGB format.
- **Selective Search**: Perform selective search on the image using different scales and minimum size parameters.
- **Region Proposal Extraction**: Extract bounding boxes (regions) based on the image area, filtering out irrelevant ones.

### Code Snippets
- **Image loading and visualization**:
    ```python
    img = cv2.imread('test.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    ```

- **Selective Search for region proposals**:
    ```python
    imgs, regions = selectivesearch.selective_search(img, scale=100, min_size=50)
    ```

- **Candidate extraction**:
    ```python
    def extract_candidates(img):
        _, regions = selectivesearch.selective_search(img, scale=100, min_size=50)
        candidates = []
        img_area = np.prod(img.shape[:2])
        for region in regions:
            if region['rect'] in candidates:
                continue
            if region['size'] < 0.05*img_area:
                continue
            candidates.append(region['rect'])
        return candidates
    ```

## 2. RCNN (Region-based Convolutional Neural Networks)

### Overview
RCNN is a popular object detection model that uses region proposals from Selective Search as input and classifies the objects within each proposed region. The model also performs bounding box regression to refine the predicted locations.

### Key Steps
- **Dataset Preparation**: A custom dataset loader is implemented to handle the Open Images Dataset, loading image data and associated bounding boxes.
- **Model Architecture**: The model is based on a pre-trained ResNet50 backbone, with a classification head for object detection and a regression head for bounding box localization.
- **Training and Evaluation**: The model is trained using SGD, with loss functions for both classification and bounding box regression.
- **Prediction and Post-Processing**: Non-Maximum Suppression (NMS) is applied after prediction to filter out overlapping boxes.

### Code Snippets
- **RCNN Model**:
    ```python
    class RCNN(nn.Module):
        def __init__(self, backbone, n_classes):
            super().__init__()
            self.backbone = backbone
            self.classification_head = nn.Linear(2048, n_classes)
            self.bbox_regression_head = nn.Sequential(
                nn.Linear(2048, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 4),
                nn.Tanh()
            )
    ```

- **Training Loop**:
    ```python
    for epoch in range(1, n_epochs+1):
        for inputs, labels, deltas in train_dataloader:
            _labels, _deltas, total_loss, classification_loss, localization_loss, acc = train_batch(rcnn, optimizer, inputs, labels, deltas)
    ```

- **Prediction and NMS**:
    ```python
    def predict(inputs):
        with torch.no_grad():
            rcnn.eval()
            labels, deltas = rcnn(inputs)
            probs = torch.nn.functional.softmax(labels, -1)
            conf, clss = probs.max(-1)
        return conf, clss, probs, deltas
    ```

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

### Example Output
- Color Temperature that is used to combine the similar feature as one in selective search. Below are original and temperature images respectively
    ![image](https://github.com/user-attachments/assets/81ab5dfa-8bfc-4ead-a265-8e4350424a74)

- Original image with bounding boxes drawn after RCNN prediction:
    ![2900bfdc-9cbf-44db-9457-0fd2b0e48948](https://github.com/user-attachments/assets/9f2508ec-3a4e-4604-ad0a-48370074f1bb)


