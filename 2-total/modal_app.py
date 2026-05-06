"""
TotalSegmentator V2 - Modal Deployment with brain_structures task
Fixed: volume mount, sync delay, thalamus L/R, correct midline split per structure
"""

import modal
import tempfile
import os

app = modal.App("totalsegmentator-brain-v2")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install(
        "fastapi",
        "python-multipart",
        "nibabel",
        "numpy",
        "torch",
        "Pillow",
        "TotalSegmentator"
    )
)

volume = modal.Volume.from_name("ct-segmentation-data", create_if_missing=True)
OUTPUT_PATH = "/data/outputs"

BRAIN_STRUCTURES_LABEL_MAP = {
    "brainstem":          1,
    "subarachnoid_space": 2,
    "venous_sinuses":     3,
    "septum_pellucidum":  4,
    "cerebellum":         5,
    "caudate_nucleus":    6,
    "lentiform_nucleus":  7,
    "insular_cortex":     8,
    "internal_capsule":   9,
    "ventricle":          10,
    "central_sulcus":     11,
    "frontal_lobe":       12,
    "parietal_lobe":      13,
    "occipital_lobe":     14,
    "temporal_lobe":      15,
    "thalamus":           16,
}


def get_class_names():
    return {
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


def map_totalseg_to_12_classes(seg_data, label_map=None):
    import numpy as np

    if label_map is None:
        label_map = BRAIN_STRUCTURES_LABEL_MAP

    id_to_name    = {v: k for k, v in label_map.items()}
    unique_labels = np.unique(seg_data)

    print(f"\n{'='*60}")
    print("TotalSegmentator brain_structures — Label Analysis")
    print(f"{'='*60}")
    print(f"  Volume shape: {seg_data.shape}")

    for label in unique_labels:
        if label > 0:
            voxels = int(np.sum(seg_data == label))
            pct    = (voxels / seg_data.size) * 100
            name   = id_to_name.get(int(label), f"unknown_{int(label)}")
            print(f"  Label {int(label):3d} ({name:30s}): {voxels:8d} voxels ({pct:.2f}%)")

    output = np.zeros_like(seg_data, dtype=np.uint8)

    def split_lr_by_structure(mask):
        """
        Split a bilateral structure into left/right
        using the structure's OWN centroid on axis 0
        — NOT the brain midline which is unreliable
        Affine confirmed: axis 0, negative = higher indices → LEFT
        """
        coords = np.where(mask)
        if len(coords[0]) == 0:
            return mask.copy(), mask.copy()

        # Use structure's own center on axis 0
        struct_midline = int(np.mean(coords[0]))

        lm = mask.copy(); lm[:struct_midline, :, :] = False  # LEFT  = higher axis-0
        rm = mask.copy(); rm[struct_midline:, :, :] = False  # RIGHT = lower axis-0

        l_count = int(np.sum(lm))
        r_count = int(np.sum(rm))

        # Sanity check — if split is very unbalanced, try median instead
        if l_count > 0 and r_count > 0:
            ratio = max(l_count, r_count) / min(l_count, r_count)
            if ratio > 3.0:
                print(f"    ⚠ Unbalanced split (ratio={ratio:.1f}), trying median")
                struct_midline = int(np.median(coords[0]))
                lm = mask.copy(); lm[:struct_midline, :, :] = False
                rm = mask.copy(); rm[struct_midline:, :, :] = False

        return lm, rm

    for label in unique_labels:
        if label == 0:
            continue

        name = id_to_name.get(int(label), "").lower()
        mask = seg_data == label

        if "frontal" in name:
            output[mask] = 1
        elif "parietal" in name:
            output[mask] = 2
        elif "temporal" in name:
            output[mask] = 3
        elif "occipital" in name:
            output[mask] = 4
        elif "cerebellum" in name:
            output[mask] = 5
        elif "ventricle" in name:
            lm, rm = split_lr_by_structure(mask)
            output[lm] = 6; output[rm] = 7
            print(f"  Ventricle     → L:{int(np.sum(lm)):,} / R:{int(np.sum(rm)):,}")
        elif "caudate" in name or "lentiform" in name:
            lm, rm = split_lr_by_structure(mask)
            output[lm] = 8; output[rm] = 9
            print(f"  Basal Ganglia → L:{int(np.sum(lm)):,} / R:{int(np.sum(rm)):,}")
        elif "brainstem" in name:
            output[mask] = 10
        elif "thalamus" in name:
            lm, rm = split_lr_by_structure(mask)
            output[lm] = 11; output[rm] = 12
            print(f"  Thalamus      → L:{int(np.sum(lm)):,} / R:{int(np.sum(rm)):,}")
        else:
            print(f"  Skipped: {name}")

    print(f"\nFinal class distribution:")
    for cid in range(1, 13):
        v = int(np.sum(output == cid))
        if v > 0:
            print(f"  Class {cid:2d} ({get_class_names()[cid]:30s}): {v:8,} voxels")

    return output


def find_output_file(output_base: str) -> str:
    for ext in [".nii.gz", ".nii"]:
        candidate = f"{output_base}{ext}"
        if os.path.exists(candidate):
            print(f"Found output: {candidate}")
            return candidate

    if os.path.isdir(output_base):
        files = [f for f in os.listdir(output_base) if f.endswith(('.nii.gz', '.nii'))]
        if files:
            found = os.path.join(output_base, files[0])
            print(f"Found output in directory: {found}")
            return found

    raise FileNotFoundError(
        f"No output found at '{output_base}'. "
        "Check stderr — likely a license or memory issue."
    )


@app.function(
    image=image,
    volumes={"/data": volume},
    secrets=[modal.Secret.from_name("totalseg-license")],
    timeout=900,
    memory=16384,
    gpu="any"
)
def _run_totalsegmentator(input_data: bytes, job_id: str) -> dict:
    import time
    import nibabel as nib
    import numpy as np
    import subprocess
    import shutil

    print(f"Job {job_id}: Starting TotalSegmentator V2 — brain_structures")

    license_key = os.environ.get("TOTALSEG_LICENSE_KEY")
    if not license_key:
        raise RuntimeError("TOTALSEG_LICENSE_KEY secret not found!")

    lic_result = subprocess.run(
        ["totalseg_set_license", "-l", license_key],
        capture_output=True, text=True
    )
    print(f"License setup: {lic_result.stdout.strip() or 'OK'}")

    os.makedirs(OUTPUT_PATH, exist_ok=True)

    input_path   = f"/tmp/{job_id}_input.nii.gz"
    output_base  = f"/tmp/{job_id}_output"
    output_final = f"{OUTPUT_PATH}/{job_id}_seg.nii.gz"

    with open(input_path, 'wb') as f:
        f.write(input_data)

    start_time = time.time()

    cmd = [
        "TotalSegmentator",
        "-i", input_path,
        "-o", output_base,
        "--task", "brain_structures",
        "--ml",
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    print(f"Return code: {result.returncode}")
    print(f"STDOUT:\n{result.stdout}")

    if result.returncode != 0:
        print(f"STDERR:\n{result.stderr}")
        raise RuntimeError(
            f"TotalSegmentator failed (code {result.returncode}).\n"
            f"STDERR: {result.stderr[-2000:]}"
        )

    inference_time = time.time() - start_time
    print(f"Inference done in {inference_time:.2f}s")

    seg_path = find_output_file(output_base)
    seg_img  = nib.load(seg_path)
    seg_data = seg_img.get_fdata()
    print(f"Shape: {seg_data.shape}, dtype: {seg_data.dtype}")

    seg_mapped = map_totalseg_to_12_classes(seg_data, BRAIN_STRUCTURES_LABEL_MAP)

    nib.save(nib.Nifti1Image(seg_mapped, seg_img.affine, seg_img.header), output_final)
    print(f"Saved: {output_final}")

    voxel_vol_ml     = float(np.prod(seg_img.header.get_zooms())) / 1000
    class_names      = get_class_names()
    structures_found = []

    for cid in range(1, 13):
        voxels = int(np.sum(seg_mapped == cid))
        if voxels > 0:
            structures_found.append({
                "class_id":  cid,
                "name":      class_names[cid],
                "voxels":    voxels,
                "volume_ml": round(voxels * voxel_vol_ml, 2)
            })

    for path in [input_path, seg_path]:
        if path and os.path.exists(path):
            os.unlink(path)
    if os.path.isdir(output_base):
        shutil.rmtree(output_base)

    volume.commit()
    import time as _time
    _time.sleep(3)

    return {
        "status":                 "success",
        "model":                  "TotalSegmentator V2 (brain_structures)",
        "inference_time_seconds": round(inference_time, 2),
        "num_structures_found":   len(structures_found),
        "total_volume_ml":        round(sum(s["volume_ml"] for s in structures_found), 2),
        "structures":             structures_found,
        "job_id":                 job_id,
    }


@app.function(
    image=image,
    volumes={"/data": volume},
)
def _get_file_from_volume(filepath: str) -> bytes:
    import time

    volume.reload()

    for attempt in range(5):
        if os.path.exists(filepath):
            with open(filepath, "rb") as f:
                return f.read()
        print(f"  File not found yet (attempt {attempt+1}/5), waiting 3s...")
        time.sleep(3)

    raise FileNotFoundError(
        f"File not found after retries: {filepath}\n"
        f"Contents of {os.path.dirname(filepath)}: "
        f"{os.listdir(os.path.dirname(filepath)) if os.path.exists(os.path.dirname(filepath)) else 'directory missing'}"
    )


@app.function(image=image)
@modal.asgi_app()
def fastapi_app():
    from fastapi import FastAPI, UploadFile, File, HTTPException
    from fastapi.responses import Response, JSONResponse
    import uuid
    import nibabel as nib
    import numpy as np

    web_app = FastAPI(
        title="TotalSegmentator V2 Brain Segmentation API",
        version="2.5.0"
    )

    @web_app.get("/")
    def root():
        return {
            "model":   "TotalSegmentator V2 — brain_structures task",
            "classes": get_class_names(),
            "endpoints": {
                "POST /segment":           "Upload .nii or .nii.gz CT scan",
                "GET  /download/{job_id}": "Download segmentation NIfTI",
                "GET  /info/{job_id}":     "Get per-structure volumes",
                "GET  /health":            "Health check"
            }
        }

    @web_app.get("/health")
    def health():
        return {"status": "healthy", "model": "TotalSegmentator V2",
                "task": "brain_structures", "classes": 12}

    @web_app.post("/segment")
    async def segment(file: UploadFile = File(...)):
        if not (file.filename.endswith(".nii") or file.filename.endswith(".nii.gz")):
            raise HTTPException(400, "Only .nii or .nii.gz files are supported")

        job_id    = str(uuid.uuid4())[:8]
        file_data = await file.read()

        if len(file_data) == 0:
            raise HTTPException(400, "Uploaded file is empty")

        print(f"Job {job_id}: received {file.filename} ({len(file_data)/1e6:.1f} MB)")

        try:
            result = _run_totalsegmentator.remote(file_data, job_id)
            result["download_url"] = f"/download/{job_id}"
            result["info_url"]     = f"/info/{job_id}"
            return JSONResponse(content=result)
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            raise HTTPException(500, f"Segmentation failed: {str(e)}")

    @web_app.get("/download/{job_id}")
    def download(job_id: str):
        filepath = f"{OUTPUT_PATH}/{job_id}_seg.nii.gz"
        try:
            data = _get_file_from_volume.remote(filepath)
            return Response(
                content=data,
                media_type="application/gzip",
                headers={"Content-Disposition": f"attachment; filename={job_id}_seg.nii.gz"}
            )
        except FileNotFoundError as e:
            print(f"404 for job {job_id}: {e}")
            raise HTTPException(404, f"No segmentation found for job {job_id}.")
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            raise HTTPException(500, str(e))

    @web_app.get("/info/{job_id}")
    def info(job_id: str):
        filepath = f"{OUTPUT_PATH}/{job_id}_seg.nii.gz"
        try:
            data = _get_file_from_volume.remote(filepath)

            with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp:
                tmp.write(data)
                tmp_path = tmp.name

            seg_img  = nib.load(tmp_path)
            seg_data = seg_img.get_fdata()
            os.unlink(tmp_path)

            voxel_vol_ml = float(np.prod(seg_img.header.get_zooms())) / 1000
            structures   = []

            for cid in range(1, 13):
                voxels = int(np.sum(seg_data == cid))
                if voxels > 0:
                    vol = round(voxels * voxel_vol_ml, 2)
                    structures.append({
                        "class_id":   cid,
                        "name":       get_class_names()[cid],
                        "voxels":     voxels,
                        "volume_ml":  vol,
                        "percentage": round((voxels / seg_data.size) * 100, 3)
                    })

            structures.sort(key=lambda x: x["volume_ml"], reverse=True)

            return {
                "job_id":           job_id,
                "total_structures": len(structures),
                "total_volume_ml":  round(sum(s["volume_ml"] for s in structures), 2),
                "image_shape":      list(seg_data.shape),
                "voxel_spacing_mm": [round(float(x), 3) for x in seg_img.header.get_zooms()],
                "structures":       structures
            }

        except FileNotFoundError as e:
            raise HTTPException(404, f"No segmentation found for job {job_id}")
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            raise HTTPException(500, str(e))

    return web_app