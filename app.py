import io
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# The best.pt file must be in the same GitHub folder as app.py.
MODEL_PATH = Path(__file__).parent / "best.pt"

CLASS_ORDER = [
    "Clean",
    "Dusty",
    "Bird-drop",
    "Physical-Damage",
    "Snow-Covered",
]

st.set_page_config(
    page_title="Solar Panel Condition Analyzer",
    page_icon="☀️",
    layout="wide",
)

st.title("Solar Panel Condition Analyzer")
st.write(
    "Upload a solar-panel image. For a photo containing several panels, "
    "draw one rectangle around each panel and analyze them separately."
)

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "best.pt was not found. Upload best.pt to the same GitHub "
            "folder as app.py."
        )
    return YOLO(str(MODEL_PATH))


def predict_crop(model, crop: Image.Image):
    result = model.predict(crop, device="cpu", verbose=False)[0]
    probabilities = result.probs.data.cpu().numpy()

    # Use the class names stored inside the trained model.
    names = result.names
    labels = [names[i] for i in range(len(probabilities))]
    scores = {
        str(labels[i]): float(probabilities[i]) * 100
        for i in range(len(probabilities))
    }
    predicted = max(scores, key=scores.get)
    return predicted, scores


def make_pie(scores, title):
    labels = list(scores.keys())
    values = list(scores.values())
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.pie(values, labels=labels, autopct="%1.1f%%", startangle=90)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def make_bar(scores, title):
    labels = list(scores.keys())
    values = list(scores.values())
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, values)
    ax.set_title(title)
    ax.set_ylabel("Model confidence (%)")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    return fig


uploaded = st.file_uploader(
    "Upload Solar Panel Image",
    type=["jpg", "jpeg", "png", "webp"],
)

if uploaded:
    original = Image.open(uploaded).convert("RGB")
    st.image(original, caption="Uploaded image", use_container_width=True)

    analysis_mode = st.radio(
        "How many panels are in this image?",
        ["One panel (analyze full image)", "Multiple panels (draw rectangles)"],
        horizontal=True,
    )

    panel_crops = []

    if analysis_mode == "One panel (analyze full image)":
        panel_crops = [("Panel 1", original)]

    else:
        try:
            from streamlit_drawable_canvas import st_canvas

            max_width = 850
            scale = min(1.0, max_width / original.width)
            canvas_width = int(original.width * scale)
            canvas_height = int(original.height * scale)
            preview = original.resize((canvas_width, canvas_height))

            st.info(
                "Select the rectangle tool and draw one rectangle around "
                "each solar panel. Draw separate rectangles for separate panels."
            )
            canvas = st_canvas(
                fill_color="rgba(255, 0, 0, 0.08)",
                stroke_width=3,
                stroke_color="#ff3333",
                background_image=preview,
                update_streamlit=True,
                height=canvas_height,
                width=canvas_width,
                drawing_mode="rect",
                key="solar_panel_canvas",
            )

            if canvas.json_data and canvas.json_data.get("objects"):
                sx = original.width / canvas_width
                sy = original.height / canvas_height

                for index, obj in enumerate(canvas.json_data["objects"], start=1):
                    left = max(0, int(obj.get("left", 0) * sx))
                    top = max(0, int(obj.get("top", 0) * sy))
                    width = int(obj.get("width", 0) * obj.get("scaleX", 1) * sx)
                    height = int(obj.get("height", 0) * obj.get("scaleY", 1) * sy)

                    right = min(original.width, left + width)
                    bottom = min(original.height, top + height)

                    if right > left and bottom > top:
                        panel_crops.append(
                            (f"Panel {index}", original.crop((left, top, right, bottom)))
                        )
        except ImportError:
            st.error(
                "The drawing tool is unavailable. Add "
                "'streamlit-drawable-canvas' to requirements.txt and redeploy."
            )

    if st.button("Analyze Solar Panel(s)", type="primary"):
        if not panel_crops:
            st.warning("Please draw at least one rectangle around a panel.")
        else:
            try:
                model = load_model()
                rows = []
                st.subheader("Individual panel results")

                for panel_name, crop in panel_crops:
                    predicted, scores = predict_crop(model, crop)

                    row = {
                        "Panel": panel_name,
                        "Predicted condition": predicted,
                    }
                    row.update(
                        {f"{label} confidence (%)": round(value, 2)
                         for label, value in scores.items()}
                    )
                    rows.append(row)

                    st.markdown(f"### {panel_name}: {predicted}")
                    col1, col2, col3 = st.columns([1, 1, 1.4])
                    with col1:
                        st.image(crop, caption=panel_name, use_container_width=True)
                    with col2:
                        st.pyplot(make_pie(scores, f"{panel_name} confidence"))
                    with col3:
                        st.pyplot(make_bar(scores, f"{panel_name} confidence"))

                results_df = pd.DataFrame(rows)
                st.subheader("Panel-wise prediction table")
                st.dataframe(results_df, use_container_width=True)

                csv_bytes = results_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download results table (CSV)",
                    data=csv_bytes,
                    file_name="solar_panel_results.csv",
                    mime="text/csv",
                )

                st.subheader("Overall condition summary")
                condition_counts = results_df["Predicted condition"].value_counts()
                summary_col1, summary_col2 = st.columns(2)

                with summary_col1:
                    fig, ax = plt.subplots(figsize=(6, 5))
                    ax.pie(
                        condition_counts.values,
                        labels=condition_counts.index,
                        autopct="%1.1f%%",
                        startangle=90,
                    )
                    ax.set_title("Overall predicted conditions")
                    st.pyplot(fig)
                    plt.close(fig)

                with summary_col2:
                    fig, ax = plt.subplots(figsize=(7, 5))
                    ax.bar(condition_counts.index, condition_counts.values)
                    ax.set_title("Number of panels by predicted condition")
                    ax.set_ylabel("Number of panels")
                    ax.tick_params(axis="x", rotation=25)
                    fig.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

                st.caption(
                    "Percentages are model confidence scores for each class. "
                    "They are NOT measurements of the physical area covered by "
                    "dust, snow, bird droppings, or damage."
                )

            except Exception as exc:
                st.error(f"Analysis failed: {exc}")

st.divider()
st.caption(
    "Model inference runs on CPU. For multiple panels, draw a separate "
    "rectangle around every panel before clicking Analyze."
)
