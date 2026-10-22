
from pathlib import Path

import cv2
import joblib
import numpy as np
import streamlit as st
from PIL import Image, ImageEnhance, ImageOps

try:
    from skimage.feature import hog
except ImportError:
    hog = None

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


# --------------------------------------------------
# Configuration
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
SVM_PATH = BASE_DIR / "models" / "svm.joblib"

YOLO_CANDIDATES = [
    BASE_DIR / "runs" / "detect" / "runs" / "fracatlas" /
    "fracture_detector" / "weights" / "best.pt",
    BASE_DIR / "runs" / "fracatlas" /
    "fracture_detector" / "weights" / "best.pt",
    BASE_DIR / "runs" / "detect" / "fracture_detector" /
    "weights" / "best.pt",
]

st.set_page_config(
    page_title="Bone Fracture Analysis",
    page_icon="🩻",
    layout="wide",
)


# --------------------------------------------------
# Load saved models
# --------------------------------------------------
@st.cache_resource
def load_svm():
    if not SVM_PATH.exists():
        return None
    return joblib.load(SVM_PATH)


@st.cache_resource
def load_detector(model_path):
    if YOLO is None or not model_path:
        return None
    return YOLO(model_path)


def find_detector_path():
    for path in YOLO_CANDIDATES:
        if path.exists():
            return path
    return None


# --------------------------------------------------
# Existing HOG + SVM classification
# --------------------------------------------------
def extract_hog_features(image):
    gray = ImageOps.grayscale(image)
    gray = gray.resize((224, 224))
    array = np.asarray(gray)

    features = hog(
        array,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )

    return features.reshape(1, -1)


def classify_image(model, image):
    features = extract_hog_features(image)
    prediction = int(model.predict(features)[0])

    # Show probability only if this estimator supports it.
    probability = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
        probability = float(np.max(probabilities))

    return prediction, probability


# --------------------------------------------------
# Streamlit interface
# --------------------------------------------------
st.title("🩻 Bone Fracture Classification & Localization")
st.write(
    "Upload a bone X-ray to run the existing machine-learning "
    "classifier and, when available, the fracture-localization model."
)

st.warning(
    "Educational prototype only. This tool is not a medical diagnosis "
    "and must not be used to make treatment decisions."
)

uploaded_file = st.file_uploader(
    "Upload a bone X-ray image",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    left, right = st.columns(2)

    with left:
        st.subheader("Original X-ray")
        st.image(image, use_container_width=True)

    with right:
        st.subheader("Contrast-enhanced preview")
        gray = ImageOps.grayscale(image)
        enhanced = ImageEnhance.Contrast(gray).enhance(1.5)
        st.image(enhanced, use_container_width=True)

    st.divider()
    st.subheader("1. Image classification")

    svm_model = load_svm()

    if svm_model is None:
        st.error("SVM model not found at models/svm.joblib.")
    else:
        try:
            prediction, probability = classify_image(svm_model, image)
            class_name = str(prediction)

            # The training script may use string labels or numeric labels.
            if hasattr(svm_model, "classes_"):
                class_name = str(svm_model.classes_[prediction])

            st.write(f"Model output: **{class_name}**")

            if probability is not None:
                st.caption(
                    f"Maximum model class probability: "
                    f"{probability:.1%}. This is not clinical certainty."
                )
        except Exception as exc:
            st.error(f"Classification failed: {exc}")

    st.divider()
    st.subheader("2. Fracture localization")

    detector_path = find_detector_path()

    if detector_path is None:
        st.info(
            "The YOLO model is not available yet. Once training finishes, "
            "the app will look for best.pt in the configured training paths."
        )
    elif YOLO is None:
        st.error(
            "Ultralytics is not installed in this Python environment. "
            "Install it using: python -m pip install ultralytics"
        )
    else:
        try:
            detector = load_detector(str(detector_path))
            results = detector.predict(
                source=np.asarray(image),
                conf=0.25,
                verbose=False,
            )

            result = results[0]
            plotted = result.plot()
            plotted_rgb = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)

            if result.boxes is not None and len(result.boxes) > 0:
                st.image(
                    plotted_rgb,
                    caption="Predicted regions — not a confirmed diagnosis",
                    use_container_width=True,
                )

                for index, box in enumerate(result.boxes, start=1):
                    confidence = float(box.conf[0])
                    st.write(
                        f"Region {index}: detection confidence "
                        f"{confidence:.1%}"
                    )
            else:
                st.image(
                    np.asarray(image),
                    caption="No region detected by the model.",
                    use_container_width=True,
                )
                st.info(
                    "No box was produced. This does not rule out a fracture."
                )

            st.caption(f"Detector weights: {detector_path}")
        except Exception as exc:
            st.error(f"Localization failed: {exc}")

st.divider()
st.caption(
    "Research prototype • Classification and localization can make errors. "
    "Consult a qualified healthcare professional for medical assessment."
)
