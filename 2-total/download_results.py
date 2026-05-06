"""
download_results.py
────────────────────────────────────────────────────────────
Batch process all NIfTI volumes through TotalSegmentator V2
Reads from : test_data/nifti_volumes/
Saves to   : test_data/results/
────────────────────────────────────────────────────────────
"""

import requests
import json
import os
import time
import numpy as np
import nibabel as nib
from PIL import Image
from pathlib import Path
from datetime import datetime

URL = "https://tnsqai-2--totalsegmentator-brain-v2-fastapi-app.modal.run"

BASE_DIR   = Path(__file__).parent
INPUT_DIR  = BASE_DIR / "test_data" / "nifti_volumes"
OUTPUT_DIR = BASE_DIR / "test_data" / "results" / "totalsegmentator"
SEG_DIR    = OUTPUT_DIR / "segmentations"
INFO_DIR   = OUTPUT_DIR / "info"
SLICES_DIR = OUTPUT_DIR / "slices"

for d in [SEG_DIR, INFO_DIR, SLICES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── 12-class color map — all distinct, no duplicates ─────────────────────────
COLORS = {
    0:  [0,   0,   0  ],   # Background          — black
    1:  [255, 0,   0  ],   # Frontal_Lobe        — pure red
    2:  [0,   0,   255],   # Parietal_Lobe       — pure blue
    3:  [255, 165, 0  ],   # Temporal_Lobe       — orange
    4:  [0,   255, 0  ],   # Occipital_Lobe      — pure green
    5:  [255, 0,   255],   # Cerebellum          — magenta
    6:  [255, 255, 0  ],   # Ventricle_Left      — pure yellow
    7:  [0,   255, 255],   # Ventricle_Right     — pure cyan
    8:  [128, 0,   128],   # Basal_Ganglia_Left  — purple
    9:  [255, 105, 180],   # Basal_Ganglia_Right — hot pink
    10: [255, 255, 255],   # Brainstem           — pure white
    11: [255, 140, 0  ],   # Thalamus_Left       — dark orange (distinct from all)
    12: [139, 69,  19 ],   # Thalamus_Right      — brown
}

CLASS_NAMES = {
    1:  "Frontal_Lobe",
    2:  "Parietal_Lobe",
    3:  "Temporal_Lobe",
    4:  "Occipital_Lobe",
    5:  "Cerebellum",
    6:  "Lateral_Ventricle_Left",
    7:  "Lateral_Ventricle_Right",
    8:  "Basal_Ganglia_Left",
    9:  "Basal_Ganglia_Right",
    10: "Brainstem",
    11: "Thalamus_Left",
    12: "Thalamus_Right",
}

# ── Orientation — confirmed correct for your scans ───────────────────────────
# Verified: frontal at top, R on left, L on right (radiological view)
ROTATION_K = 3      # 90° clockwise
FLIP_LR    = True   # R on left, L on right
FLIP_UD    = True   # frontal at top
# ─────────────────────────────────────────────────────────────────────────────


def apply_orientation(arr: np.ndarray) -> np.ndarray:
    arr = np.rot90(arr, k=ROTATION_K)
    if FLIP_LR:
        arr = np.fliplr(arr)
    if FLIP_UD:
        arr = np.flipud(arr)
    return arr


def colorize_slice(slice_data: np.ndarray) -> np.ndarray:
    color_slice = np.zeros((*slice_data.shape, 3), dtype=np.uint8)
    for label, color in COLORS.items():
        color_slice[slice_data == label] = color
    return color_slice


def download_with_retry(url: str, max_retries: int = 5, wait: int = 8) -> requests.Response:
    for attempt in range(1, max_retries + 1):
        resp = requests.get(url, timeout=120)
        if resp.status_code == 200:
            return resp
        print(f"    ⏳ Attempt {attempt}/{max_retries} → {resp.status_code}, retrying in {wait}s...")
        time.sleep(wait)
    raise RuntimeError(f"Download failed after {max_retries} attempts: {resp.status_code}")


def extract_slices(seg_path: Path, patient_slices_dir: Path):
    patient_slices_dir.mkdir(parents=True, exist_ok=True)
    coronal_dir = patient_slices_dir / "coronal"
    coronal_dir.mkdir(exist_ok=True)

    seg_img  = nib.load(str(seg_path))
    seg_data = seg_img.get_fdata().astype(np.uint8)

    # Axial slices (z-axis)
    num_axial = seg_data.shape[2]
    for i in range(num_axial):
        color_slice = colorize_slice(seg_data[:, :, i])
        Image.fromarray(apply_orientation(color_slice), mode="RGB").save(
            patient_slices_dir / f"axial_{i:03d}.png"
        )

    # Coronal slices (y-axis)
    num_coronal = seg_data.shape[1]
    for i in range(num_coronal):
        color_slice = colorize_slice(seg_data[:, i, :])
        Image.fromarray(apply_orientation(color_slice), mode="RGB").save(
            coronal_dir / f"coronal_{i:03d}.png"
        )

    return num_axial, num_coronal


def main():
    nifti_files = sorted(INPUT_DIR.glob("*.nii.gz"))

    print(f"{'='*62}")
    print(f"  TotalSegmentator V2 — Batch Processing & Download")
    print(f"{'='*62}")
    print(f"  Input       : {INPUT_DIR}")
    print(f"  Output      : {OUTPUT_DIR}")
    print(f"  URL         : {URL}")
    print(f"  Orientation : rot90 k={ROTATION_K}, flip_lr={FLIP_LR}, flip_ud={FLIP_UD}")
    print(f"  Classes     : {len(CLASS_NAMES)} (incl. Thalamus L/R)")
    print(f"  Patients    : {len(nifti_files)}")
    print(f"  Started     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*62}\n")

    if len(nifti_files) == 0:
        print(f"No .nii.gz files found in {INPUT_DIR}")
        print(f"Run preprocessing.py first to generate NIfTI volumes.")
        return

    summary = []
    failed  = []

    for idx, nifti_file in enumerate(nifti_files, 1):
        patient_id = nifti_file.stem.replace(".nii", "")
        file_size  = nifti_file.stat().st_size / (1024 * 1024)

        print(f"[{idx}/{len(nifti_files)}] {patient_id}  ({file_size:.1f} MB)")
        print(f"  {'─'*55}")

        try:
            print(f"  ↑  Uploading to TotalSegmentator...")
            with open(nifti_file, "rb") as f:
                resp = requests.post(
                    f"{URL}/segment",
                    files={"file": (nifti_file.name, f, "application/gzip")},
                    timeout=600
                )

            if resp.status_code != 200:
                raise RuntimeError(f"Segment API {resp.status_code}: {resp.text[:300]}")

            result = resp.json()
            if result.get("status") != "success":
                raise RuntimeError(f"Segmentation failed: {result}")

            job_id         = result["job_id"]
            inference_time = result["inference_time_seconds"]
            total_vol      = result["total_volume_ml"]
            structures     = result["structures"]

            print(f"  ✓  Done in {inference_time}s | {len(structures)} structures | {total_vol} mL total")

            print(f"  ↓  Downloading segmentation mask...")
            seg_resp = download_with_retry(f"{URL}/download/{job_id}")
            seg_path = SEG_DIR / f"{patient_id}_seg.nii.gz"
            with open(seg_path, "wb") as f:
                f.write(seg_resp.content)
            print(f"  ✓  Mask saved → segmentations/{patient_id}_seg.nii.gz ({seg_path.stat().st_size/1e6:.1f} MB)")

            print(f"  ↓  Downloading structure info...")
            info_resp = download_with_retry(f"{URL}/info/{job_id}")
            info_data = info_resp.json()

            patient_info = {
                "patient_id":             patient_id,
                "job_id":                 job_id,
                "input_file":             nifti_file.name,
                "input_size_mb":          round(file_size, 2),
                "inference_time_seconds": inference_time,
                "model":                  "TotalSegmentator V2 (brain_structures)",
                "image_shape":            info_data.get("image_shape"),
                "voxel_spacing_mm":       info_data.get("voxel_spacing_mm"),
                "total_volume_ml":        info_data.get("total_volume_ml"),
                "total_structures":       info_data.get("total_structures"),
                "structures":             info_data.get("structures", []),
                "processed_at":           datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            with open(INFO_DIR / f"{patient_id}_info.json", "w") as f:
                json.dump(patient_info, f, indent=2)
            print(f"  ✓  Info saved  → info/{patient_id}_info.json")

            print(f"\n  {'Structure':<30} {'Volume(mL)':>10} {'Voxels':>10}")
            print(f"  {'─'*53}")
            for s in sorted(info_data.get("structures", []), key=lambda x: x["class_id"]):
                print(f"  {s['name']:<30} {s['volume_ml']:>10.1f} {s['voxels']:>10,}")
            print(f"  {'─'*53}")
            print(f"  {'TOTAL':<30} {info_data.get('total_volume_ml', 0):>10.1f}")

            print(f"\n  Extracting slices to PNG...")
            num_axial, num_coronal = extract_slices(seg_path, SLICES_DIR / patient_id)
            print(f"  ✓  Slices saved → slices/{patient_id}/ ({num_axial} axial + {num_coronal} coronal)")

            summary.append({
                "patient_id":             patient_id,
                "job_id":                 job_id,
                "status":                 "success",
                "inference_time_seconds": inference_time,
                "total_volume_ml":        total_vol,
                "num_structures":         len(structures),
                "seg_file":               f"segmentations/{patient_id}_seg.nii.gz",
                "info_file":              f"info/{patient_id}_info.json",
                "slices_dir":             f"slices/{patient_id}/",
            })

            print(f"\n  ✅ {patient_id} complete!\n")

        except Exception as e:
            print(f"\n  ❌ FAILED: {str(e)}\n")
            failed.append({"patient_id": patient_id, "error": str(e), "file": str(nifti_file)})

    summary_data = {
        "run_info": {
            "model":          "TotalSegmentator V2 (brain_structures)",
            "url":            URL,
            "total_patients": len(nifti_files),
            "successful":     len(summary),
            "failed":         len(failed),
            "orientation":    f"rot90 k={ROTATION_K}, flip_lr={FLIP_LR}, flip_ud={FLIP_UD}",
            "completed_at":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "output_dir":     str(OUTPUT_DIR)
        },
        "class_schema": CLASS_NAMES,
        "color_map":    {str(k): v for k, v in COLORS.items()},
        "results":      summary,
    }

    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(summary_data, f, indent=2)

    if failed:
        with open(OUTPUT_DIR / "failed.json", "w") as f:
            json.dump(failed, f, indent=2)

    print(f"\n{'='*62}")
    print(f"  BATCH COMPLETE")
    print(f"{'='*62}")
    print(f"  Total    : {len(nifti_files)} patients")
    print(f"  ✅ Done  : {len(summary)}")
    print(f"  ❌ Failed: {len(failed)}")
    if failed:
        print(f"\n  Run: python retry_failed.py")
    print(f"{'='*62}\n")


if __name__ == "__main__":
    main()