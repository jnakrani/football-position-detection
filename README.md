# Custom Object Detection Project

## Description
This project uses YOLOv11 for custom object detection, such as identifying specific positions in sports videos (e.g., "QB" for Quarterback and "C" for Center). It includes steps for training, testing, and running a YOLOv11 model.

## Setup

### 1. Create a Virtual Environment
To keep dependencies isolated, create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### 2. Install Requirements
Install all required dependencies:
```bash
pip install -r requirements.txt
```

---

## Train the Model
1. **Prepare Data**:
   - Use the `train_model.ipynb` notebook for training the model.
   - Upload raw video/image data to [Roboflow](https://roboflow.com/), annotate the objects (e.g., "QB", "C"), and export the annotated dataset.

2. **Add Configuration**:
   - Create a `.env` file in the root directory.
   - Add the following environment variables:
     ```env
     ROBOFLOW_API_KEY=<YOUR_API_KEY>
     WORKSPACE=<YOUR_WORKSPACE_NAME>
     PROJECT_NAME=<YOUR_PROJECT_NAME>
     ```

3. **Train the Model**:
   - Run the `train_model.ipynb` notebook to download data and train the model.

---

## Test the Model
1. Place test videos in the `test_data` folder or use your own videos.
2. Set up the following paths in your script:
   - **Model Path**: Path to the trained YOLOv11 model (e.g., `position_detection.pt`).
   - **Video Path**: Path to the input test video.
   - **Output Path**: Path to save the processed video with predictions.

Example setup:
```python
model_path = "path/to/position_detection.pt"
video_path = "path/to/test_data/video.mp4"
output_path = "path/to/output_video.mp4"
```

3. Run the testing script to process and annotate the video.

---

## Results
The resulting video will display detected objects with custom labels (e.g., "QB", "C") centered in circular markers.

---
