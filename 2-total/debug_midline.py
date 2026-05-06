"""
debug_midline.py v3
Finds true brain midline from actual brain voxel positions
"""
import nibabel as nib
import numpy as np
from PIL import Image
from pathlib import Path

SEG_FILE   = Path(r"C:\Users\Lenovo\CTBrain\2-total\test_data\results\totalsegmentator\segmentations\595853990_seg.nii.gz")
OUTPUT_DIR = Path(r"C:\Users\Lenovo\CTBrain\2-total\debug_midline")
OUTPUT_DIR.mkdir(exist_ok=True)

seg_img  = nib.load(str(SEG_FILE))
seg_data = seg_img.get_fdata().astype(np.uint8)
affine   = seg_img.affine

print(f"Shape  : {seg_data.shape}")
print(f"Affine[0,0] = {affine[0,0]:.3f}  (negative = higher axis-0 idx → LEFT)")

# Find where brain voxels actually are on axis 0
brain_mask = seg_data > 0
axis0_positions = np.where(brain_mask.any(axis=(1,2)))[0]
print(f"\nBrain extent on axis 0: {axis0_positions[0]} → {axis0_positions[-1]}")
true_midline = (axis0_positions[0] + axis0_positions[-1]) // 2
print(f"True brain midline    : {true_midline}  (shape midline = {seg_data.shape[0]//2})")

# Test both midlines
for mid_name, mid in [("shape//2", seg_data.shape[0]//2), ("brain_midline", true_midline)]:
    ventricle_mask = (seg_data == 6) | (seg_data == 7)
    lm = ventricle_mask.copy(); lm[:mid, :, :] = False
    rm = ventricle_mask.copy(); rm[mid:, :, :] = False
    print(f"\n  {mid_name} (={mid}): L={int(np.sum(lm)):,}  R={int(np.sum(rm)):,}")

    mid_slice = seg_data.shape[2] // 2
    vis = np.zeros((seg_data.shape[0], seg_data.shape[1], 3), dtype=np.uint8)
    # Show full brain in gray
    vis[brain_mask[:,:,mid_slice]] = [80, 80, 80]
    vis[lm[:,:,mid_slice]] = [255, 0, 0]    # Red = Left
    vis[rm[:,:,mid_slice]] = [0, 0, 255]    # Blue = Right
    # Draw midline
    vis[mid, :, :] = [0, 255, 0]            # Green line = midline
    vis = np.rot90(vis, k=3)
    vis = np.fliplr(vis)
    Image.fromarray(vis, mode="RGB").save(OUTPUT_DIR / f"midline_{mid_name}.png")
    print(f"  Saved: midline_{mid_name}.png")

print("\n✅ Share both PNG files — pick the one where RED and BLUE are symmetric")