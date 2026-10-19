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


# Published experiment results, calculated offline from the saved checkpoints.
# These tables are not recomputed when visitors open the app.
validation_results = pd.DataFrame([
    ["Baseline U-Net", 0.8353, 0.6830, 22],
    ["Augmented U-Net", 0.8417, 0.7004, 17],
    ["Tversky U-Net", 0.8510, 0.7352, 13],
    ["ResNet50 U-Net", 0.8570, 0.7595, 10],
], columns=["Model", "Mean Dice", "Glioma Dice", "Severe glioma failures"])

test_results = pd.DataFrame([
    ["Baseline U-Net", 0.8209, 0.7412, 0.8580, 0.8452],
    ["Augmented U-Net", 0.8124, 0.7309, 0.8374, 0.8439],
    ["Tversky U-Net", 0.8269, 0.7422, 0.8052, 0.8982],
    ["ResNet50 U-Net", 0.8373, 0.7556, 0.8273, 0.9042],
], columns=["Model", "Mean Dice", "Mean IoU", "Mean Precision", "Mean Recall"])

test_category_results = pd.DataFrame([
    ["Baseline U-Net", 0.6698, 0.9256, 0.8432],
    ["Augmented U-Net", 0.6693, 0.9154, 0.8295],
    ["Tversky U-Net", 0.7117, 0.9191, 0.8312],
    ["ResNet50 U-Net", 0.7364, 0.9233, 0.8356],
], columns=["Model", "Glioma", "Meningioma", "Pituitary"])

test_failures = pd.DataFrame([
    ["Baseline U-Net", 30],
    ["Augmented U-Net", 31],
    ["Tversky U-Net", 17],
    ["ResNet50 U-Net", 20],
], columns=["Model", "Severe failures"])


tab1, tab2, tab3 = st.tabs([
    "📊 Model Performance",
    "🧠 Explore MRI Samples",
    "📤 Upload Research MRI"
])

with tab1:
    st.subheader("Model Comparison")
    st.write(
        "Four saved architectures were compared on 588 validation images "
        "and 853 test images. The final ResNet50 U-Net was selected."
    )

    st.markdown("#### Validation and Test - Mean Dice")
    dice_comparison = validation_results[["Model", "Mean Dice"]].rename(
        columns={"Mean Dice": "Validation (n=588)"}
    ).merge(
        test_results[["Model", "Mean Dice"]].rename(
            columns={"Mean Dice": "Test (n=853)"}
        ),
        on="Model"
    )
    st.dataframe(dice_comparison, hide_index=True, width="stretch")
    st.bar_chart(
        dice_comparison.set_index("Model"),
        stack=False,
        y_label="Mean Dice"
    )

    st.markdown("#### Test Set - Overall Metrics")
    st.dataframe(test_results, hide_index=True, width="stretch")
    st.caption(
        "Mean metrics are calculated per image and then averaged across "
        "the 853 test images. Dice and IoU measure mask overlap; "
        "precision and recall provide complementary information about "
        "false-positive and missed tumor pixels."
    )

    st.markdown("#### Test Set - Dice by Tumor Category")
    st.dataframe(test_category_results, hide_index=True, width="stretch")
    st.bar_chart(
        test_category_results.set_index("Model"),
        stack=False,
        y_label="Mean Dice"
    )
    st.caption(
        "ResNet50 U-Net achieved the highest test glioma Dice (0.7364), "
        "while the baseline achieved the highest test Dice for "
        "meningioma (0.9256) and pituitary tumors (0.8432)."
    )

    st.markdown("#### Severe Glioma Failures - Validation and Test")
    failures_comparison = validation_results[["Model", "Severe glioma failures"]].rename(
        columns={"Severe glioma failures": "Validation (out of 160)"}
    ).merge(
        test_failures.rename(columns={"Severe failures": "Test (out of 254)"}),
        on="Model"
    )
    st.dataframe(failures_comparison, hide_index=True, width="stretch")
    st.caption(
        "A severe failure is defined as an image-level Dice below 0.20. "
        "The denominators differ between splits, so raw failure counts "
        "should not be compared across validation and test as rates. "
        "Tversky U-Net had the fewest severe glioma failures on test (17/254)."
    )

    st.info(
        "Evaluation methodology: Models were selected using a validation "
        "set of 588 images and subsequently evaluated on a test set of "
        "853 images."
    )

with tab2:
    st.subheader("Explore MRI Samples — Validation Examples")
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

        if not sample_names:
            st.info("No example is available for this tumor category.")
        else:
            selection_key = f"selected_sample_{tumor}"

            if (
                selection_key not in st.session_state
                or st.session_state[selection_key] not in sample_names
            ):
                st.session_state[selection_key] = sample_names[0]

            def sample_label(filename):
                index = sample_names.index(filename) + 1
                return f"Example {index}"

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
            mask_path = masks_dir / (Path(selection).stem + ".png")

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

            # Four equally sized images, with modest width even on wide screens.
            col1, col2, col3, col4 = st.columns(4, gap="small")

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
                    caption="Ground truth",
                    width="stretch",
                    clamp=True
                )

            with col3:
                st.image(
                    predicted_mask.astype(np.uint8) * 255,
                    caption="Predicted mask",
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
                "Validation Dice - selected MRI",
                f"{selected_row['validation_dice']:.4f}"
            )
            st.caption(
                "This score is recorded in the curated validation sample "
                "manifest and describes this example, not overall test "
                "performance. Dice measures overlap between the predicted "
                "mask and the reference annotation."
            )
            st.info(
                "These 12 curated examples come from the validation set "
                "and illustrate individual segmentation outcomes."
            )

with tab3:
    st.subheader("Upload an Anonymized Research MRI")

    st.write(
        "Upload a PNG research image to explore how "
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
                "Please use a valid PNG research image."
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
