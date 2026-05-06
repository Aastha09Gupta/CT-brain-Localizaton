import json
import numpy as np
import pydicom
from PIL import Image
from pathlib import Path
from datetime import datetime

# ── CONFIG ────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
INPUT_DIR  = BASE_DIR / "test_data" / "s3_downloads"
OUTPUT_DIR = BASE_DIR / "test_data" / "dicom_pngs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CT_PLAIN_PATTERNS = [
    "CT 5mm Plain",
    "CT 55mm Plain",
    "CT Plain 55mm-2",
    "CT Plain 55mm",
    "CT Plain",
    "CT*Plain*",
]

WINDOW_PRESETS = {
    "brain":    {"center": 40,  "width": 80  },
    "subdural": {"center": 75,  "width": 215 },
    "stroke":   {"center": 32,  "width": 8   },
    "bone":     {"center": 600, "width": 2800},
}

WINDOW = WINDOW_PRESETS["brain"]


def apply_window(pixel_array: np.ndarray, center: int, width: int) -> np.ndarray:
    min_hu   = center - (width / 2)
    max_hu   = center + (width / 2)
    windowed = np.clip(pixel_array, min_hu, max_hu)
    windowed = ((windowed - min_hu) / (max_hu - min_hu) * 255).astype(np.uint8)
    return windowed


def load_dicom(dcm_path: Path):
    dcm       = pydicom.dcmread(str(dcm_path))
    pixels    = dcm.pixel_array.astype(np.float32)
    slope     = float(getattr(dcm, "RescaleSlope",     1))
    intercept = float(getattr(dcm, "RescaleIntercept", 0))
    pixels    = pixels * slope + intercept
    slice_loc = float(getattr(dcm, "SliceLocation",
                    getattr(dcm, "InstanceNumber", 0)))
    return pixels, slice_loc


def find_ct_plain_folder(patient_folder: Path):
    scmc_folders = list(patient_folder.glob("SCMC*"))
    search_root  = scmc_folders[0] if scmc_folders else patient_folder

    if scmc_folders:
        print(f"  SCMC folder : {search_root.name}")
    else:
        print(f"  No SCMC folder, searching patient root")

    for pattern in CT_PLAIN_PATTERNS:
        matches = list(search_root.glob(pattern))
        if matches:
            print(f"  CT folder   : '{matches[0].name}'")
            return matches[0]

    plain_folders = [f for f in search_root.iterdir()
                     if f.is_dir() and "plain" in f.name.lower()]
    if plain_folders:
        print(f"  CT folder   : '{plain_folders[0].name}' (fallback match)")
        return plain_folders[0]

    available = [f.name for f in search_root.iterdir() if f.is_dir()]
    print(f"  ⚠️  No CT Plain folder found.")
    print(f"  Available folders: {', '.join(available[:8])}")
    return None


def convert_patient(ct_folder: Path, output_dir: Path, window: dict, patient_id: str) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    dcm_files = list(ct_folder.glob("*.dcm"))
    if not dcm_files:
        dcm_files = [f for f in ct_folder.glob("*")
                     if f.is_file() and f.suffix not in
                     [".json", ".txt", ".xml", ".png", ".jpg"]]

    if not dcm_files:
        raise ValueError(f"No DICOM files found in {ct_folder}")

    print(f"  Loading     : {len(dcm_files)} DICOM files")

    slices = []
    errors = 0
    for dcm_path in dcm_files:
        try:
            pixels, slice_loc = load_dicom(dcm_path)
            slices.append((slice_loc, pixels))
        except Exception:
            errors += 1

    if errors > 0:
        print(f"  ⚠️  Skipped {errors} unreadable files")

    if not slices:
        raise ValueError("No valid DICOM slices loaded")

    slices.sort(key=lambda x: x[0])
    print(f"  Slices      : {len(slices)} | "
          f"range {slices[0][0]:.1f} → {slices[-1][0]:.1f} mm")

    for idx, (_, pixels) in enumerate(slices):
        windowed = apply_window(pixels, window["center"], window["width"])

        # CHANGED: filename is now patientid_slicenumber.png
        filename = f"{patient_id}_{idx}.png"
        Image.fromarray(windowed, mode="L").save(output_dir / filename)

    print(f"  Saved as    : {patient_id}_0.png → {patient_id}_{len(slices)-1}.png")

    return {
        "num_slices":    len(slices),
        "image_shape":   list(slices[0][1].shape),
        "window_center": window["center"],
        "window_width":  window["width"],
        "slice_loc_min": round(slices[0][0], 2),
        "slice_loc_max": round(slices[-1][0], 2),
        "skipped_files": errors,
        "filename_format": f"{patient_id}_{{slice_number}}.png"
    }


def main():
    if not INPUT_DIR.exists():
        print(f"Input folder not found: {INPUT_DIR}")
        return

    patient_folders = sorted([f for f in INPUT_DIR.iterdir() if f.is_dir()])

    print(f"{'='*62}")
    print(f"  DICOM to PNG Converter — Brain CT")
    print(f"{'='*62}")
    print(f"  Input       : {INPUT_DIR}")
    print(f"  Output      : {OUTPUT_DIR}")
    print(f"  Window      : center={WINDOW['center']} / width={WINDOW['width']} HU")
    print(f"  Filename    : {{patient_id}}_{{slice_number}}.png")
    print(f"  Patients    : {len(patient_folders)}")
    print(f"  Started     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*62}\n")

    if not patient_folders:
        print(f"No patient folders found in {INPUT_DIR}")
        return

    summary = []
    failed  = []
    skipped = []

    for idx, patient_folder in enumerate(patient_folders, 1):
        patient_id = patient_folder.name
        print(f"[{idx}/{len(patient_folders)}] Patient: {patient_id}")
        print(f"  {'─'*50}")

        try:
            ct_folder = find_ct_plain_folder(patient_folder)

            if ct_folder is None:
                print(f"  ⏭️  Skipped\n")
                skipped.append({
                    "patient_id": patient_id,
                    "reason":     "No CT Plain folder found"
                })
                continue

            # CHANGED: pass patient_id into convert_patient for filename
            stats = convert_patient(ct_folder, OUTPUT_DIR / patient_id, WINDOW, patient_id)
            print(f"  ✅ Saved {stats['num_slices']} PNGs → "
                  f"test_data/dicom_pngs/{patient_id}/\n")

            summary.append({
                "patient_id": patient_id,
                "status":     "success",
                "ct_folder":  ct_folder.name,
                "output_dir": str(OUTPUT_DIR / patient_id),
                **stats
            })

        except Exception as e:
            print(f"  ❌ ERROR: {e}\n")
            failed.append({"patient_id": patient_id, "error": str(e)})

    with open(OUTPUT_DIR / "conversion_summary.json", "w") as f:
        json.dump({
            "run_info": {
                "script_location":  str(BASE_DIR),
                "input_dir":        str(INPUT_DIR),
                "output_dir":       str(OUTPUT_DIR),
                "total_patients":   len(patient_folders),
                "successful":       len(summary),
                "skipped":          len(skipped),
                "failed":           len(failed),
                "window":           WINDOW,
                "filename_format":  "{patient_id}_{slice_number}.png",
                "completed_at":     datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            },
            "ct_plain_patterns_tried": CT_PLAIN_PATTERNS,
            "results":  summary,
            "skipped":  skipped,
            **({"failed": failed} if failed else {})
        }, f, indent=2)

    print(f"{'='*62}")
    print(f"  COMPLETE")
    print(f"{'='*62}")
    print(f"  ✅ Converted : {len(summary)} patients")
    print(f"  ⏭️  Skipped  : {len(skipped)} (no CT Plain folder)")
    print(f"  ❌ Failed    : {len(failed)}")
    print(f"\n  Output: {OUTPUT_DIR}")
    print(f"  Example filenames:")
    if summary:
        pid = summary[0]['patient_id']
        n   = summary[0]['num_slices'] - 1
        print(f"    {pid}_0.png, {pid}_1.png ... {pid}_{n}.png")
    print(f"{'='*62}\n")


if __name__ == "__main__":
    main()