"""
Solar panel image-classification training script for Kaggle.
Downloads the public dataset, prepares five classes, trains YOLOv8-cls
on CPU, saves best.pt, and exports panel-wise test tables and charts.

Run in Kaggle with Internet enabled and CPU accelerator selected.
"""

import shutil
from pathlib import Path

import kagglehub
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from ultralytics import YOLO

WORK_DIR = Path("/kaggle/working")
DATA_DIR = WORK_DIR / "solar_dataset"
CLASSES = ["Clean", "Dusty", "Bird-drop", "Physical-Damage", "Snow-Covered"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

ALIASES = {
    "clean": "Clean",
    "dust": "Dusty",
    "dusty": "Dusty",
    "birddrop": "Bird-drop",
    "birddropping": "Bird-drop",
    "physicaldamage": "Physical-Damage",
    "snowcovered": "Snow-Covered",
}

def normalize(value):
    return "".join(char.lower() for char in value if char.isalnum())


print("Downloading Solar Panel Images dataset...")
dataset_path = Path(
    kagglehub.dataset_download("pythonafroz/solar-panel-images")
)
print("Dataset downloaded to:", dataset_path)

image_records = []
for image_path in dataset_path.rglob("*"):
    if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
        continue

    found_class = None
    for parent in image_path.parents:
        if parent == dataset_path.parent:
            break
        normalized_folder = normalize(parent.name)
        if normalized_folder in ALIASES:
            found_class = ALIASES[normalized_folder]
            break

    if found_class:
        image_records.append({"path": image_path, "class": found_class})

df = pd.DataFrame(image_records)
if df.empty:
    raise RuntimeError(
        "No images were matched to the five expected class folders. "
        "Inspect the downloaded dataset folders and update ALIASES."
    )

print("\nImages found per class:")
print(df["class"].value_counts().reindex(CLASSES, fill_value=0))

# Remove old prepared folders so reruns do not duplicate files.
if DATA_DIR.exists():
    shutil.rmtree(DATA_DIR)

for split in ["train", "val", "test"]:
    for class_name in CLASSES:
        (DATA_DIR / split / class_name).mkdir(parents=True, exist_ok=True)

for class_name in CLASSES:
    images = df.loc[df["class"] == class_name, "path"].tolist()
    if len(images) < 10:
        raise RuntimeError(
            f"Class {class_name} has only {len(images)} matched images; "
            "at least 10 are required for this split."
        )

    train_images, temp_images = train_test_split(
        images, test_size=0.30, random_state=42
    )
    val_images, test_images = train_test_split(
        temp_images, test_size=1/3, random_state=42
    )

    for split_name, split_images in [
        ("train", train_images),
        ("val", val_images),
        ("test", test_images),
    ]:
        for index, source in enumerate(split_images):
            destination = (
                DATA_DIR / split_name / class_name /
                f"{class_name}_{index}_{source.name}"
            )
            shutil.copy2(source, destination)

print("\nPrepared dataset:")
for split_name in ["train", "val", "test"]:
    print(split_name, {
        cls: len(list((DATA_DIR / split_name / cls).glob("*")))
        for cls in CLASSES
    })

print("\nStarting CPU training...")
model = YOLO("yolov8n-cls.pt")
training_results = model.train(
    data=str(DATA_DIR),
    epochs=20,
    imgsz=160,
    batch=8,
    device="cpu",
    workers=0,
    patience=5,
    project=str(WORK_DIR / "solar_training"),
    name="solar_panel_model",
    plots=True,
)

trained_weights = Path(training_results.save_dir) / "weights" / "best.pt"
if not trained_weights.exists():
    raise FileNotFoundError(f"Training checkpoint not found: {trained_weights}")

best_pt = WORK_DIR / "best.pt"
shutil.copy2(trained_weights, best_pt)
print("\nBEST MODEL SAVED:", best_pt)

# Test-set predictions and per-panel confidence table.
best_model = YOLO(str(best_pt))
test_images = [
    path
    for cls in CLASSES
    for path in (DATA_DIR / "test" / cls).iterdir()
    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
]

rows = []
pie_dir = WORK_DIR / "pie_charts"
bar_dir = WORK_DIR / "bar_graphs"
pie_dir.mkdir(exist_ok=True)
bar_dir.mkdir(exist_ok=True)

for image_path in test_images:
    prediction = best_model.predict(
        source=str(image_path), device="cpu", verbose=False
    )[0]
    probabilities = prediction.probs.data.cpu().numpy()
    model_names = prediction.names
    labels = [model_names[i] for i in range(len(probabilities))]
    predicted_class = labels[int(probabilities.argmax())]

    row = {
        "Panel": image_path.name,
        "Actual Condition": image_path.parent.name,
        "Predicted Condition": predicted_class,
    }
    scores = {}
    for index, label in enumerate(labels):
        score = float(probabilities[index]) * 100
        row[f"{label} Confidence (%)"] = round(score, 2)
        scores[label] = score
    rows.append(row)

    safe_stem = image_path.stem.replace(" ", "_")

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie(list(scores.values()), labels=list(scores.keys()),
           autopct="%1.1f%%", startangle=90)
    ax.set_title(f"Panel: {image_path.name}")
    fig.tight_layout()
    fig.savefig(pie_dir / f"{safe_stem}_pie.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(list(scores.keys()), list(scores.values()))
    ax.set_title(f"Panel: {image_path.name}")
    ax.set_xlabel("Solar panel condition")
    ax.set_ylabel("Model confidence (%)")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(bar_dir / f"{safe_stem}_bar.png")
    plt.close(fig)

results_df = pd.DataFrame(rows)
results_csv = WORK_DIR / "solar_panel_results.csv"
results_df.to_csv(results_csv, index=False)

if not results_df.empty:
    counts = results_df["Predicted Condition"].value_counts()

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%", startangle=90)
    ax.set_title("Overall Solar Panel Conditions")
    fig.tight_layout()
    fig.savefig(WORK_DIR / "overall_pie_chart.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(counts.index, counts.values)
    ax.set_title("Total Solar Panels by Condition")
    ax.set_xlabel("Condition")
    ax.set_ylabel("Number of panels")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(WORK_DIR / "overall_bar_graph.png")
    plt.close(fig)

print("\nTRAINING AND EXPORT COMPLETE")
print("Model:", best_pt)
print("Results table:", results_csv)
print("Individual pie charts:", pie_dir)
print("Individual bar graphs:", bar_dir)
print(
    "\nConfidence percentages are model scores, not measured physical "
    "defect-area percentages."
)
