

# Bone Fracture Classification and Localization

A machine-learning project that classifies bone X-ray images and aims to localize suspected fracture regions using object detection.

## Features

- Binary X-ray classification using HOG image features and an SVM classifier.
- Random Forest model for comparison.
- YOLO-based fracture localization using annotated bounding boxes.
- Streamlit interface for uploading and viewing X-ray images.
- Evaluation using accuracy, precision, recall, F1-score, and confusion matrices.

## Technologies

Python, OpenCV, scikit-image, scikit-learn, NumPy, Joblib, Ultralytics YOLO, Streamlit.

## Project Structure

```text
bonefactureml/
├── app.py
├── prepare_fracatlas.py
├── requirements.txt
├── readme.md
├── .gitignore
├── models/
│   ├── svm.joblib
│   └── random_forest.joblib
├── src/
│   ├── check_dataset.py
│   ├── extract_features.py
│   ├── train_models.py
│   ├── evaluate_models.py
│   └── predict_image.py
├── data/
│   └── (datasets excluded from Git)
└── runs/
    └── (training outputs excluded from Git)
```

## Setup

1. Install Python 3.11 or a compatible version.
2. Clone this repository.
3. Create and activate a virtual environment.
4. Install dependencies:

```bash
pip install -r requirements.txt
```

5. Obtain the datasets separately and place them in the directories expected by the scripts.
6. Ensure the trained model files are available in the expected model paths.

## Run the Application

```bash
streamlit run app.py
```

Open the local URL shown in the terminal.

## Fracture Localization

The localization model is trained using FracAtlas bounding-box annotations. The dataset must be downloaded separately. After training, place the final model weights at the path configured in the application.

A bounding box indicates a model-predicted region, not a confirmed fracture.

## Limitations

- Classification and localization may produce false positives or false negatives.
- Dataset annotation quality affects performance.
- Evaluation on an independent dataset is needed to assess generalization.
- The system does not determine fracture severity or replace clinical assessment.

## Disclaimer

This project is intended for educational and research purposes only. It is not a medical diagnostic tool.



Local URL: http://localhost:8501
  Network URL: http://10.75.179.189:8501
