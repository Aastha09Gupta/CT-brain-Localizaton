import nibabel as nib
import numpy as np
from PIL import Image
from pathlib import Path

# ── Config ────────────────────────────────────────────────
seg_file   = "44e36187_seg.nii.gz"   
output_dir = Path("slices_output")
output_dir.mkdir(exist_ok=True)

# ── 12-class color map ────────────────────────────────────
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
    11: [255, 165, 0  ],   # Thalamus_Left       — orange
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

# ── Load ──────────────────────────────────────────────────
print(f"Loading: {seg_file}")
seg_img  = nib.load(seg_file)
seg_data = seg_img.get_fdata().astype(np.uint8)

print(f"Shape : {seg_data.shape}")
print(f"Voxel spacing: {seg_img.header.get_zooms()} mm")

# ── Label stats ───────────────────────────────────────────
print("\nLabel distribution:")
voxel_vol_ml = float(np.prod(seg_img.header.get_zooms())) / 1000
for label in np.unique(seg_data):
    if label > 0:
        voxels = int(np.sum(seg_data == label))
        vol_ml = round(voxels * voxel_vol_ml, 2)
        name   = CLASS_NAMES.get(label, f"unknown_{label}")
        print(f"  Class {label:2d} ({name:30s}): {voxels:8d} voxels  ({vol_ml:.1f} mL)")

# ── Extract axial slices (z-axis) ─────────────────────────
num_slices = seg_data.shape[2]
print(f"\nExtracting {num_slices} axial slices...")

for i in range(num_slices):
    slice_data  = seg_data[:, :, i]
    color_slice = np.zeros((*slice_data.shape, 3), dtype=np.uint8)

    for label, color in COLORS.items():
        color_slice[slice_data == label] = color

    color_slice = np.rot90(color_slice)
    img = Image.fromarray(color_slice, mode='RGB')
    img.save(output_dir / f"axial_{i:03d}.png")

    if (i + 1) % 10 == 0 or (i + 1) == num_slices:
        print(f"  {i+1}/{num_slices} done")

# ── Also extract coronal slices (y-axis) ──────────────────
print(f"\nExtracting {seg_data.shape[1]} coronal slices...")
coronal_dir = output_dir / "coronal"
coronal_dir.mkdir(exist_ok=True)

for i in range(seg_data.shape[1]):
    slice_data  = seg_data[:, i, :]
    color_slice = np.zeros((*slice_data.shape, 3), dtype=np.uint8)

    for label, color in COLORS.items():
        color_slice[slice_data == label] = color

    color_slice = np.rot90(color_slice)
    img = Image.fromarray(color_slice, mode='RGB')
    img.save(coronal_dir / f"coronal_{i:03d}.png")

print(f"\n✅ Done!")
print(f"\nOutput folders:")
print(f"  {output_dir}/axial_*.png     — {num_slices} axial slices")
print(f"  {output_dir}/coronal/        — {seg_data.shape[1]} coronal slices")
print(f"\nColor Legend:")
for label, name in CLASS_NAMES.items():
    r, g, b = COLORS[label]
    print(f"  Class {label:2d} ({name:30s}) → RGB({r},{g},{b})")