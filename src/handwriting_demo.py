from pathlib import Path
import base64
import csv
import json
import sys

import cv2
import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = (
    ROOT / "models"
    / "full_13065_v2_full_epoch_003.keras"
)
MAPPING_PATH = (
    ROOT / "results" / "stage9"
    / "character_mapping_for_demo.csv"
)
ERROR_PATH = (
    ROOT / "results"
    / "full_13065_v2_test_errors_sample.csv"
)
OUT = ROOT / "results" / "stage9"
OUT.mkdir(parents=True, exist_ok=True)

N_CLASSES = 13065
IMAGE_SIZE = 96

MODES = [
    "area_inverted",
    "area_normal",
    "linear_inverted",
    "linear_normal",
    "binary_inverted",
    "binary_normal",
]


def read_csv(path):
    with path.open(
        "r", encoding="utf-8-sig", newline=""
    ) as file:
        return list(csv.DictReader(file))


def resolve_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_image(path):
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(str(path))

    data = np.fromfile(str(path), dtype=np.uint8)

    image = cv2.imdecode(
        data, cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        raise ValueError(
            "Cannot decode image: " + str(path)
        )

    return image


def preprocess(image, mode):
    if mode.startswith("linear_"):
        interpolation = cv2.INTER_LINEAR
    else:
        interpolation = cv2.INTER_AREA

    resized = cv2.resize(
        image,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=interpolation,
    )

    values = resized.astype(np.float32) / 255.0

    if mode.startswith("binary_"):
        values = (
            values >= (127.0 / 255.0)
        ).astype(np.float32)

    if mode.endswith("_inverted"):
        values = 1.0 - values

    return values[..., np.newaxis]


def load_mapping():
    rows = read_csv(MAPPING_PATH)

    mapping = {
        int(row["class_id"]): row
        for row in rows
    }

    expected = set(range(1, N_CLASSES + 1))

    if (
        len(rows) != N_CLASSES
        or set(mapping.keys()) != expected
    ):
        raise ValueError(
            "Character mapping is incomplete."
        )

    if len({
        row["character"] for row in rows
    }) != N_CLASSES:
        raise ValueError(
            "Duplicate mapped characters detected."
        )

    return mapping


def load_reference_examples():
    examples = []
    used_paths = set()

    for row in read_csv(ERROR_PATH):
        if len(examples) >= 32:
            break

        path = resolve_path(row["image_path"])

        if str(path) in used_paths:
            continue

        try:
            image = read_image(path)
            expected = int(
                row["predicted_class_id"]
            )
        except (OSError, ValueError, KeyError):
            continue

        if not 1 <= expected <= N_CLASSES:
            continue

        examples.append((image, expected))
        used_paths.add(str(path))

    if len(examples) < 12:
        raise RuntimeError(
            "Too few reference images for "
            "preprocessing compatibility check."
        )

    return examples


def calibrate_preprocessing(model):
    print("")
    print("Checking inference compatibility...")
    print("Using saved test predictions as references.")

    references = load_reference_examples()

    expected = np.array(
        [item[1] for item in references],
        dtype=np.int64,
    )

    scores = []

    for mode in MODES:
        batch = np.stack([
            preprocess(image, mode)
            for image, _ in references
        ])

        predictions = model.predict(
            batch, batch_size=16, verbose=0
        )

        if isinstance(predictions, (list, tuple)):
            raise RuntimeError(
                "Unexpected multiple model outputs."
            )

        indices = np.argmax(
            predictions, axis=1
        )

        for offset in (0, 1):
            matches = int(np.sum(
                indices + offset == expected
            ))

            scores.append({
                "mode": mode,
                "class_id_offset": offset,
                "matched": matches,
                "total": len(expected),
                "agreement": (
                    matches / len(expected)
                ),
            })

    scores.sort(
        key=lambda item: item["matched"],
        reverse=True,
    )

    best = scores[0]

    report = {
        "model": MODEL_PATH.name,
        "reference_samples": len(references),
        "best_candidate": best,
        "all_candidates": scores,
        "important_note": (
            "This checks agreement with cached "
            "test predictions. It is not a new "
            "accuracy evaluation or model training."
        ),
    }

    report_path = OUT / "demo_preflight.json"

    report_path.write_text(
        json.dumps(
            report, indent=2, ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print("")
    print("Top preprocessing candidates:")

    for item in scores[:6]:
        print(
            f"  {item['mode']:<20} "
            f"offset={item['class_id_offset']} "
            f"matches={item['matched']}/"
            f"{item['total']}"
        )

    print("Compatibility report:", report_path)

    if best["agreement"] < 0.90:
        raise RuntimeError(
            "Preprocessing agreement is too low. "
            "Demo blocked to avoid misleading "
            "Chinese-character predictions. "
            "Please inspect demo_preflight.json."
        )

    if best["class_id_offset"] != 1:
        raise RuntimeError(
            "Class-ID offset does not match the "
            "expected 1-based mapping. "
            "Demo blocked for manual verification."
        )

    print(
        "Preprocessing compatibility PASSED!",
        f"Agreement: {best['agreement'] * 100:.1f}%"
    )

    return best["mode"]


def probabilities_from_output(values):
    values = np.asarray(
        values, dtype=np.float64
    ).reshape(-1)

    if not np.all(np.isfinite(values)):
        raise RuntimeError(
            "Model returned non-finite values."
        )

    if (
        np.min(values) >= 0
        and np.max(values) <= 1
        and abs(np.sum(values) - 1) < 0.001
    ):
        return values / np.sum(values)

    shifted = values - np.max(values)
    exp_values = np.exp(shifted)

    return exp_values / np.sum(exp_values)


def create_ui(model, mapping, mode):
    import tkinter as tk
    from tkinter import filedialog, messagebox

    BG = "#101827"
    PANEL = "#1b2739"
    WHITE = "#f1f5f9"
    MUTED = "#a9b7cb"
    ACCENT = "#38bdf8"

    root = tk.Tk()
    root.title(
        "Traditional Chinese Handwriting Recognition"
    )
    root.geometry("960x690")
    root.minsize(820, 600)
    root.configure(bg=BG)

    tk.Label(
        root,
        text="繁體中文手寫識別系統",
        font=("Microsoft YaHei UI", 24, "bold"),
        fg=WHITE,
        bg=BG,
    ).pack(pady=(24, 6))

    tk.Label(
        root,
        text="DIT5411  |  CNN V2  |  13,065 Character Classes",
        font=("Segoe UI", 11),
        fg=MUTED,
        bg=BG,
    ).pack(pady=(0, 22))

    content = tk.Frame(root, bg=BG)
    content.pack(
        fill="both", expand=True,
        padx=30, pady=8
    )

    left = tk.Frame(content, bg=PANEL)
    left.pack(
        side="left", fill="both",
        expand=True, padx=(0, 10)
    )

    right = tk.Frame(content, bg=PANEL)
    right.pack(
        side="left", fill="both",
        expand=True, padx=(10, 0)
    )

    tk.Label(
        left, text="輸入手寫圖片",
        font=("Microsoft YaHei UI", 15, "bold"),
        fg=WHITE, bg=PANEL
    ).pack(pady=(20, 8))

    preview_box = tk.Frame(
        left,
        bg="white",
        width=310,
        height=300
    )
    preview_box.pack(
        padx=12,
        pady=10
    )
    preview_box.pack_propagate(False)

    image_label = tk.Label(
        preview_box,
        text="請選擇圖片",
        bg="white",
        fg="#64748b",
        font=("Microsoft YaHei UI", 12)
    )
    image_label.pack(expand=True)

    file_label = tk.Label(
        left,
        text="尚未選擇圖片",
        font=("Microsoft YaHei UI", 10),
        fg=MUTED,
        bg=PANEL,
        wraplength=350
    )
    file_label.pack(pady=8)

    tk.Label(
        right, text="模型識別結果",
        font=("Microsoft YaHei UI", 15, "bold"),
        fg=WHITE, bg=PANEL
    ).pack(pady=(20, 16))

    main_result = tk.Label(
        right,
        text="等待圖片",
        font=("Microsoft YaHei UI", 23, "bold"),
        fg=ACCENT,
        bg=PANEL,
        wraplength=340
    )
    main_result.pack(pady=(0, 14))

    result_frame = tk.Frame(right, bg=PANEL)
    result_frame.pack(
        padx=20, pady=(0, 12),
        fill="both", expand=True
    )

    scrollbar = tk.Scrollbar(
        result_frame, orient="vertical"
    )
    scrollbar.pack(side="right", fill="y")

    details = tk.Text(
        result_frame,
        bg=PANEL,
        fg=WHITE,
        insertbackground=WHITE,
        font=("Microsoft YaHei UI", 11),
        relief="flat",
        width=29,
        height=14,
        spacing2=5,
        wrap="word",
        yscrollcommand=scrollbar.set,
    )
    details.pack(
        side="left", fill="both", expand=True
    )
    scrollbar.configure(command=details.yview)

    details.insert(
        "1.0",
        "請選擇一張單字手寫圖片。\n\n"
        "系統將顯示 Top-5 預測候選。"
    )
    details.config(state="disabled")

    def predict_file():
        path = filedialog.askopenfilename(
            title="選擇手寫漢字圖片",
            filetypes=[
                (
                    "Image files",
                    "*.png *.jpg *.jpeg *.bmp *.webp"
                ),
                ("All files", "*.*"),
            ],
        )

        if not path:
            return

        try:
            image = read_image(path)

            sample = preprocess(
                image, mode
            )[np.newaxis, ...]

            output = model.predict(
                sample, verbose=0
            )[0]

            probabilities = (
                probabilities_from_output(output)
            )

            indices = np.argsort(
                probabilities
            )[-5:][::-1]

            results = []

            for rank, index in enumerate(indices, 1):
                class_id = int(index) + 1
                item = mapping[class_id]

                results.append({
                    "rank": rank,
                    "class_id": class_id,
                    "character": item["character"],
                    "status": item["status"],
                    "confidence": (
                        float(probabilities[index])
                    ),
                })

            preview = cv2.resize(
                image,
                (285, 285),
                interpolation=cv2.INTER_AREA
            )

            ok, encoded = cv2.imencode(
                ".png", preview
            )

            if not ok:
                raise ValueError(
                    "Unable to prepare preview."
                )

            photo = tk.PhotoImage(
                data=base64.b64encode(
                    encoded.tobytes()
                ).decode("ascii")
            )

            image_label.configure(
                image=photo,
                text=""
            )
            image_label.image = photo

            file_label.config(
                text=Path(path).name
            )

            winner = results[0]

            main_result.config(
                text=(
                    f"{winner['character']}\n"
                    f"Class {winner['class_id']}"
                )
            )

            lines = [
                "Top-5 Recognition Results",
                "------------------------------",
                "",
            ]

            for item in results:
                lines.extend([
                    (
                        f"{item['rank']}. "
                        f"{item['character']}  "
                        f"(Class {item['class_id']})"
                    ),
                    (
                        f"   Score: "
                        f"{item['confidence'] * 100:.2f}%"
                    ),
                    (
                        f"   Mapping: {item['status']}"
                    ),
                    "",
                ])

            details.config(state="normal")
            details.delete("1.0", tk.END)
            details.insert(
                "1.0", "\n".join(lines)
            )
            details.config(state="disabled")

        except Exception as exc:
            messagebox.showerror(
                "Prediction failed",
                str(exc)
            )

    tk.Button(
        left,
        text="選擇圖片並開始識別",
        command=predict_file,
        font=("Microsoft YaHei UI", 13, "bold"),
        bg=ACCENT,
        fg="#082f49",
        activebackground="#7dd3fc",
        relief="flat",
        padx=16,
        pady=12,
        cursor="hand2",
    ).pack(pady=(16, 20))

    tk.Label(
        root,
        text=(
            "Top-5 scores are model estimates, not "
            "guarantees. Use one character per image. "
            "Some character mappings remain inferred."
        ),
        bg=BG,
        fg=MUTED,
        font=("Segoe UI", 9),
        wraplength=840
    ).pack(pady=(10, 20))

    print("Stage 9B GUI launched successfully!")
    root.mainloop()


def main():
    print("=" * 60)
    print("Stage 9B - Chinese Handwriting Demo")
    print("=" * 60)

    print("Loading character mapping...")
    mapping = load_mapping()

    print("Loading CNN V2 checkpoint...")
    model = tf.keras.models.load_model(
        str(MODEL_PATH),
        compile=False
    )

    input_shape = model.input_shape
    output_shape = model.output_shape

    print("Model input shape:", input_shape)
    print("Model output shape:", output_shape)

    if (
        not isinstance(input_shape, tuple)
        or tuple(input_shape[1:]) != (96, 96, 1)
    ):
        raise RuntimeError(
            "Unexpected model input shape."
        )

    if (
        not isinstance(output_shape, tuple)
        or output_shape[-1] != N_CLASSES
    ):
        raise RuntimeError(
            "Unexpected model output classes."
        )

    # The public repository does not contain
    # the original test images or cached test CSV.
    #
    # On the development machine, keep the full
    # compatibility check. For a fresh GitHub
    # checkout, use the previously validated
    # preprocessing configuration.
    if ERROR_PATH.is_file():
        mode = calibrate_preprocessing(model)
    else:
        mode = "area_inverted"
        print("")
        print(
            "Original test references are unavailable."
        )
        print(
            "Using validated preprocessing: "
            "area_inverted"
        )
        print(
            "Cached prediction preflight skipped. "
            "This is not a new accuracy evaluation."
        )

    print("Selected preprocessing:", mode)
    print("Character mapping count:", len(mapping))
    print("Stage 9B Preflight PASSED!")

    create_ui(model, mapping, mode)


if __name__ == "__main__":
    main()