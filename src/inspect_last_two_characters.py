from pathlib import Path
import csv
import json
import numpy as np
import cv2

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "extracted" / "cleaned_data"
OUT = ROOT / "results" / "stage8" / "mapping_v3"
OUT.mkdir(parents=True, exist_ok=True)

CASES = [
    (2781, "濈u", "濈"),
    (12332, "p嶒", "嶒"),
]

font_path = None
for name in ("msjh.ttc", "msyh.ttc", "mingliu.ttc"):
    p = Path(r"C:\Windows\Fonts") / name
    if p.exists():
        font_path = p
        break

font = (
    FontProperties(fname=str(font_path))
    if font_path
    else FontProperties()
)

fig, axes = plt.subplots(2, 7, figsize=(20, 7))
report = []

for row_index, (class_id, original_text, candidate) in enumerate(CASES):
    folder = DATA / str(class_id)
    images = sorted(folder.glob("*.png")) if folder.exists() else []

    info = {
        "class_id": class_id,
        "decoded_filename_label": original_text,
        "single_character_candidate": candidate,
        "candidate_verified": False,
        "available_images": len(images),
    }
    report.append(info)

    left = axes[row_index, 0]
    left.axis("off")
    left.text(
        0.5, 0.65,
        f"Class {class_id}\n"
        f"Decoded: {original_text}\n"
        f"Candidate: {candidate}\n"
        f"UNVERIFIED",
        ha="center", va="center",
        transform=left.transAxes,
        fontsize=12,
        fontproperties=font,
    )

    if images:
        indices = np.linspace(
            0, len(images) - 1,
            min(6, len(images)),
            dtype=int
        )
        chosen = [images[i] for i in indices]
    else:
        chosen = []

    for col_index in range(1, 7):
        ax = axes[row_index, col_index]
        ax.axis("off")

        if col_index > len(chosen):
            ax.text(
                0.5, 0.5, "No image",
                ha="center", va="center",
                transform=ax.transAxes
            )
            continue

        path = chosen[col_index - 1]
        data = np.fromfile(str(path), dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)

        if image is None:
            ax.text(
                0.5, 0.5, "Read failed",
                ha="center", va="center",
                transform=ax.transAxes
            )
            continue

        ax.imshow(image, cmap="gray", vmin=0, vmax=255)
        ax.set_title(path.name, fontsize=9)

fig.suptitle(
    "Remaining Two Classes - Visual Label Audit\n"
    "Candidates are NOT verified ground truth",
    fontsize=16
)
fig.tight_layout(rect=[0, 0, 1, 0.90])

picture = OUT / "last_two_classes_visual_audit.png"
fig.savefig(picture, dpi=180, bbox_inches="tight")
plt.close(fig)

report_path = OUT / "last_two_classes_visual_audit.json"
report_path.write_text(
    json.dumps(report, ensure_ascii=False, indent=2),
    encoding="utf-8"
)

print("=" * 55)
print("Stage 8C-6 - Visual Label Audit")
for item in report:
    print(
        f"Class {item['class_id']}: "
        f"{item['available_images']} images | "
        f"Candidate: {item['single_character_candidate']} "
        f"(UNVERIFIED)"
    )
print("Image:", picture)
print("Report:", report_path)
print("Stage 8C-6 PASSED!")