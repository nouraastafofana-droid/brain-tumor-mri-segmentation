from huggingface_hub import hf_hub_download

import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd

from pathlib import Path
from PIL import Image

st.set_page_config(
    page_title="Brain MRI Segmentation",
    page_icon="🧠",
    layout="wide"
)

APP_DIR = Path(__file__).resolve().parent
MODEL_REPO = "read855/brain-mri-segmentation-resnet50-unet"
SAMPLES_DIR = APP_DIR / "samples"

st.title("🧠 Brain MRI Segmentation")
st.caption("Interactive research demonstration | ResNet50 U-Net")

st.warning(
    "Research demonstration only. This model has not been "
    "validated for medical diagnosis or clinical decision-making. "
    "Do not upload personal medical records or identifiable images."
)

@st.cache_resource
def load_segmentation_model():
    model_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename="resnet50_unet.keras",
        repo_type="model"
    )

    return tf.keras.models.load_model(
        model_path,
        custom_objects={
            "preprocess_input":
                tf.keras.applications.resnet50.preprocess_input
        },
        compile=False,
        safe_mode=True
    )

def segment_image(image):
    image = image.convert("L").resize((256, 256))
    image_array = np.asarray(image, dtype=np.float32) / 255.0
    model_input = image_array[None, ..., None]

    model = load_segmentation_model()
    prediction = model.predict(model_input, verbose=0)[0, :, :, 0]

    mask = prediction >= 0.5

    overlay = np.stack([image_array] * 3, axis=-1)
    overlay[mask] = (
        0.45 * overlay[mask]
        + 0.55 * np.array([1.0, 0.2, 0.3])
    )

    return image_array, mask, overlay

def display_segmentation(image):
    with st.spinner("Generating segmentation..."):
        original, mask, overlay = segment_image(image)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(original, caption="Original MRI", width="stretch", clamp=True)

    with col2:
        st.image(
            mask.astype(np.uint8) * 255,
            caption="Predicted mask",
            width="stretch",
            clamp=True
        )

    with col3:
        st.image(
            overlay,
            caption="Segmentation overlay",
            width="stretch",
            clamp=True
        )

tab1, tab2, tab3 = st.tabs([
    "📊 Model Performance",
    "🧠 Explore MRI Samples",
    "📤 Upload Research MRI"
])

with tab1:
    st.subheader("Model Comparison")

    results = pd.DataFrame([
        ["Baseline U-Net", 0.8353, 0.6830, 22],
        ["Augmented U-Net", 0.8417, 0.7004, 17],
        ["U-Net + Tversky", 0.8510, 0.7352, 13],
        ["ResNet50 U-Net", 0.8570, 0.7595, 10],
    ], columns=[
        "Model",
        "Mean Dice",
        "Glioma Dice",
        "Severe glioma failures"
    ])

    st.dataframe(results, hide_index=True, width="stretch")

    st.subheader("Validation Dice Comparison")
    st.bar_chart(
        results.set_index("Model")[["Mean Dice", "Glioma Dice"]],
        stack=False,
        y_label="Dice score",
        horizontal=False
    )

    st.info(
        "Metrics were calculated on a single validation partition "
        "of 588 images. Patient-level independence could not be "
        "verified. These results are exploratory."
    )







with tab2:
    st.subheader("Explore MRI Samples")

    st.write(
        "Choose an MRI from the validation dataset and compare "
        "the reference annotation with the model prediction."
    )

    manifest_path = SAMPLES_DIR / "samples_manifest.csv"
    masks_dir = APP_DIR / "ground_truth"

    if not manifest_path.exists():
        st.info("Demo samples are not available.")
    else:
        manifest = pd.read_csv(manifest_path)

        st.markdown("#### 1. Choose a tumor category")

        tumor = st.radio(
            "Tumor category",
            ["glioma", "meningioma", "pituitary"],
            horizontal=True,
            format_func=lambda x: x.capitalize(),
            label_visibility="collapsed"
        )

        filtered = manifest[
            manifest["tumor_label"] == tumor
        ].reset_index(drop=True)

        sample_names = filtered["sample"].tolist()

        st.markdown("#### 2. Explore an MRI")

        # Keep a separate selection for each tumor category
        selection_key = f"selected_sample_{tumor}"

        if (
            selection_key not in st.session_state
            or st.session_state[selection_key] not in sample_names
        ):
            st.session_state[selection_key] = sample_names[0]

        def sample_label(filename):
            index = sample_names.index(filename) + 1
            return f"Example {index}"

        # Button comes before the widget to avoid session state errors
        if st.button("🎲 Try a random MRI", key=f"random_{tumor}"):
            import random

            choices = [
                name for name in sample_names
                if name != st.session_state[selection_key]
            ]

            st.session_state[selection_key] = random.choice(
                choices or sample_names
            )

        selection = st.selectbox(
            "Select an MRI example",
            sample_names,
            key=selection_key,
            format_func=sample_label
        )

        selected_row = filtered[
            filtered["sample"] == selection
        ].iloc[0]

        image_path = SAMPLES_DIR / selection
        mask_path = masks_dir / (
            Path(selection).stem + ".png"
        )

        st.markdown("#### 3. Understand the prediction")

        with st.spinner("Generating segmentation..."):
            image = Image.open(image_path)
            original, predicted_mask, overlay = segment_image(image)

        reference_mask = Image.open(mask_path).convert("L")
        reference_mask = reference_mask.resize(
            (256, 256),
            Image.Resampling.NEAREST
        )

        reference_array = np.asarray(reference_mask)
        reference_binary = (reference_array >= 128).astype(np.uint8)

        col1, col2 = st.columns(2)

        with col1:
            st.image(
                original,
                caption="Original MRI",
                width="stretch",
                clamp=True
            )

        with col2:
            st.image(
                reference_binary * 255,
                caption="Ground truth - reference annotation",
                width="stretch",
                clamp=True
            )

        col3, col4 = st.columns(2)

        with col3:
            st.image(
                predicted_mask.astype(np.uint8) * 255,
                caption="Predicted segmentation mask",
                width="stretch",
                clamp=True
            )

        with col4:
            st.image(
                overlay,
                caption="Prediction overlay",
                width="stretch",
                clamp=True
            )

        st.metric(
            "Dice score - selected MRI",
            f"{selected_row['validation_dice']:.4f}"
        )

        st.caption(
            "Dice measures agreement between the predicted "
            "mask and the reference annotation for this specific "
            "image."
        )

        st.info(
            "These 12 examples were selected from the validation "
            "dataset to illustrate different performance levels. "
            "They are not an independent evaluation."
        )

with tab3:
    st.subheader("Upload an Anonymized Research MRI")

    st.write(
        "Upload a PNG or JPG research image to explore how "
        "the experimental model generates a segmentation mask."
    )

    st.warning(
        "Do not upload personal medical scans or images containing "
        "patient-identifying information. The model is not designed "
        "to confirm or rule out a brain tumor."
    )

    uploaded_file = st.file_uploader(
        "Choose a research MRI image",
        type=["png", "jpg", "jpeg"],
        max_upload_size=10
    )

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            display_segmentation(image)
        except Exception:
            st.error(
                "Unable to process this image. "
                "Please use a valid PNG or JPG research image."
            )

st.divider()

st.caption(
    "Experimental machine learning portfolio project. "
    "Not intended for medical diagnosis, treatment, or clinical use."
)


st.caption(
    "Dataset: BRISC2025 — "
    "[Kaggle source](https://www.kaggle.com/datasets/briscdataset/brisc2025) "
    "| [CC BY 4.0 License](https://creativecommons.org/licenses/by/4.0/). "
    "Demo samples were selected and converted for visualization. "
    "Dataset attribution does not imply endorsement of this model."
)
