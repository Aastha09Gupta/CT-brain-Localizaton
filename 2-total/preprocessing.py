"""
Shared preprocessing for all CT segmentation models
Handles DICOM series and NIfTI files
"""

import os
import numpy as np
import nibabel as nib
import pydicom
import SimpleITK as sitk
from pathlib import Path
from typing import Union, Tuple, Optional


class CTPreprocessor:
    """CT scan preprocessing for brain segmentation"""
    
    def __init__(self, brain_window: Tuple[int, int] = (-100, 100)):
        """
        Args:
            brain_window: HU window for brain tissue (min, max)
        """
        self.brain_window = brain_window
    
    def load_dicom_series(self, dicom_folder: str) -> sitk.Image:
        """
        Load DICOM series from folder (e.g., 30 .dcm files)
        
        Args:
            dicom_folder: Path to folder containing DICOM files
            
        Returns:
            SimpleITK Image object
        """
        print(f"Loading DICOM series from: {dicom_folder}")
        
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(dicom_folder)
        
        if len(dicom_names) == 0:
            raise ValueError(f"No DICOM files found in {dicom_folder}")
        
        print(f"Found {len(dicom_names)} DICOM slices")
        
        reader.SetFileNames(dicom_names)
        image = reader.Execute()
        
        return image
    
    def load_nifti(self, nifti_path: str) -> sitk.Image:
        """
        Load NIfTI file
        
        Args:
            nifti_path: Path to .nii or .nii.gz file
            
        Returns:
            SimpleITK Image object
        """
        print(f"Loading NIfTI from: {nifti_path}")
        image = sitk.ReadImage(nifti_path)
        return image
    
    def auto_load(self, input_path: str) -> sitk.Image:
        """
        Automatically detect and load DICOM folder or NIfTI file
        
        Args:
            input_path: Path to DICOM folder or NIfTI file
            
        Returns:
            SimpleITK Image object
        """
        path = Path(input_path)
        
        if path.is_dir():
            # DICOM folder
            return self.load_dicom_series(str(path))
        elif path.suffix in ['.nii', '.gz']:
            # NIfTI file
            return self.load_nifti(str(path))
        else:
            raise ValueError(f"Unsupported format: {path.suffix}")
    
    def apply_brain_window(self, image: sitk.Image) -> sitk.Image:
        """
        Apply brain tissue windowing (HU range)
        
        Args:
            image: Input CT image
            
        Returns:
            Windowed image
        """
        print(f"Applying brain window: {self.brain_window}")
        
        # Convert to array
        array = sitk.GetArrayFromImage(image)
        
        # Apply windowing
        min_hu, max_hu = self.brain_window
        array = np.clip(array, min_hu, max_hu)
        
        # Convert back to image
        windowed = sitk.GetImageFromArray(array)
        windowed.CopyInformation(image)
        
        return windowed
    
    def normalize_intensity(self, image: sitk.Image, 
                          method: str = "zscore") -> sitk.Image:
        """
        Normalize image intensities
        
        Args:
            image: Input image
            method: "zscore", "minmax", or "brain"
            
        Returns:
            Normalized image
        """
        print(f"Normalizing with method: {method}")
        
        array = sitk.GetArrayFromImage(image)
        
        if method == "zscore":
            # Z-score normalization
            mean = np.mean(array)
            std = np.std(array)
            array = (array - mean) / (std + 1e-8)
            
        elif method == "minmax":
            # Min-max normalization to [0, 1]
            min_val = np.min(array)
            max_val = np.max(array)
            array = (array - min_val) / (max_val - min_val + 1e-8)
            
        elif method == "brain":
            # Specific for brain CT (-100 to 100 HU → 0 to 1)
            array = (array + 100) / 200.0
            array = np.clip(array, 0, 1)
        
        # Convert back to image
        normalized = sitk.GetImageFromArray(array)
        normalized.CopyInformation(image)
        
        return normalized
    
    def resample_image(self, image: sitk.Image, 
                      target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                      interpolator: int = sitk.sitkLinear) -> sitk.Image:
        """
        Resample image to target spacing
        
        Args:
            image: Input image
            target_spacing: Desired voxel spacing (x, y, z) in mm
            interpolator: SimpleITK interpolator
            
        Returns:
            Resampled image
        """
        original_spacing = image.GetSpacing()
        original_size = image.GetSize()
        
        print(f"Original spacing: {original_spacing}")
        print(f"Target spacing: {target_spacing}")
        
        # Calculate new size
        new_size = [
            int(round(original_size[i] * (original_spacing[i] / target_spacing[i])))
            for i in range(3)
        ]
        
        # Resample
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(target_spacing)
        resampler.SetSize(new_size)
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetInterpolator(interpolator)
        resampler.SetDefaultPixelValue(image.GetPixelIDValue())
        
        resampled = resampler.Execute(image)
        
        print(f"New size: {resampled.GetSize()}")
        return resampled
    
    def to_standard_orientation(self, image: sitk.Image) -> sitk.Image:
        """
        Convert to standard RAS orientation
        
        Args:
            image: Input image
            
        Returns:
            Reoriented image
        """
        print("Converting to RAS orientation")
        return sitk.DICOMOrient(image, 'RAS')
    
    def save_nifti(self, image: sitk.Image, output_path: str):
        """
        Save image as NIfTI format
        
        Args:
            image: Image to save
            output_path: Output file path (.nii.gz)
        """
        print(f"Saving NIfTI to: {output_path}")
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        sitk.WriteImage(image, output_path)
        print(f"Saved successfully")
    
    def preprocess_for_segmentation(self, 
                                   input_path: str,
                                   output_path: Optional[str] = None,
                                   apply_windowing: bool = True,
                                   normalize: bool = True,
                                   resample: bool = False,
                                   target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)) -> sitk.Image:
        """
        Complete preprocessing pipeline
        
        Args:
            input_path: Path to DICOM folder or NIfTI file
            output_path: Optional output path for preprocessed NIfTI
            apply_windowing: Apply brain windowing
            normalize: Normalize intensities
            resample: Resample to target spacing
            target_spacing: Target voxel spacing if resampling
            
        Returns:
            Preprocessed image
        """
        print("\n" + "="*50)
        print("Starting CT Preprocessing Pipeline")
        print("="*50 + "\n")
        
        # 1. Load image
        image = self.auto_load(input_path)
        print(f"Loaded image shape: {image.GetSize()}")
        print(f"Original spacing: {image.GetSpacing()}")
        
        # 2. Convert to standard orientation
        image = self.to_standard_orientation(image)
        
        # 3. Apply brain windowing (optional)
        if apply_windowing:
            image = self.apply_brain_window(image)
        
        # 4. Resample (optional)
        if resample:
            image = self.resample_image(image, target_spacing)
        
        # 5. Normalize intensities (optional)
        if normalize:
            image = self.normalize_intensity(image, method="brain")
        
        # 6. Save if output path provided
        if output_path:
            self.save_nifti(image, output_path)
        
        print("\n" + "="*50)
        print("Preprocessing Complete")
        print("="*50 + "\n")
        
        return image
    
    def convert_dicom_to_nifti(self, dicom_folder: str, output_path: str):
        """
        Simple DICOM to NIfTI conversion
        
        Args:
            dicom_folder: Path to folder with DICOM files
            output_path: Output NIfTI path
        """
        image = self.load_dicom_series(dicom_folder)
        image = self.to_standard_orientation(image)
        self.save_nifti(image, output_path)
        print(f"Converted {dicom_folder} → {output_path}")


# Utility functions for quick access
def dicom_to_nifti(dicom_folder: str, output_path: str):
    """Quick DICOM to NIfTI conversion"""
    preprocessor = CTPreprocessor()
    preprocessor.convert_dicom_to_nifti(dicom_folder, output_path)


def preprocess_ct(input_path: str, output_path: str, 
                 windowing: bool = True, normalize: bool = True):
    """Quick preprocessing"""
    preprocessor = CTPreprocessor()
    preprocessor.preprocess_for_segmentation(
        input_path, 
        output_path,
        apply_windowing=windowing,
        normalize=normalize
    )


# Test/Debug
if __name__ == "__main__":
    import os
    from pathlib import Path
    
    preprocessor = CTPreprocessor()
    
    # Paths (both as Path objects)
    base_path = Path(r"C:\Users\Lenovo\CTBrain\test_data\s3_downloads")
    output_dir = Path("test_data/nifti_volumes")  # ✅ Changed to Path
    
    output_dir.mkdir(parents=True, exist_ok=True)  # ✅ Create directory
    
    if not base_path.exists():
        print(f"❌ Path not found: {base_path}")
        exit(1)
    
    patient_folders = sorted([f for f in base_path.iterdir() if f.is_dir()])
    
    print(f"Found {len(patient_folders)} patients\n")
    
    converted = 0
    skipped = 0
    
    for patient_folder in patient_folders:
        patient_id = patient_folder.name
        
        scmc_folders = list(patient_folder.glob("SCMC*"))
        
        if not scmc_folders:
            print(f"⚠️  {patient_id}: No SCMC folder")
            skipped += 1
            continue
        
        scmc_folder = scmc_folders[0]
        
        ct_plain_patterns = [
            "CT 5mm Plain",
            "CT 55mm Plain",
            "CT*Plain",
            "CT Plain"
        ]
        
        dicom_folder = None
        
        for pattern in ct_plain_patterns:
            matches = list(scmc_folder.glob(pattern))
            if matches:
                dicom_folder = matches[0]
                break
        
        if not dicom_folder:
            plain_folders = [f for f in scmc_folder.iterdir() 
                           if f.is_dir() and "plain" in f.name.lower()]
            if plain_folders:
                dicom_folder = plain_folders[0]
        
        if not dicom_folder:
            available = [f.name for f in scmc_folder.iterdir() if f.is_dir()]
            print(f"⚠️  {patient_id}: No CT Plain folder")
            print(f"    Available: {', '.join(available[:3])}")
            skipped += 1
            continue
        
        output_path = output_dir / f"{patient_id}.nii.gz"  # ✅ Now works
        
        if output_path.exists():
            print(f"✓ {patient_id}.nii.gz (exists)")
            converted += 1
            continue
        
        try:
            print(f"Converting {patient_id}...")
            print(f"  Folder: {dicom_folder.name}")
            
            preprocessor.convert_dicom_to_nifti(str(dicom_folder), str(output_path))
            
            print(f"✅ {patient_id}.nii.gz\n")
            converted += 1
            
        except Exception as e:
            print(f"❌ {patient_id}: {e}\n")
            skipped += 1
    
    print(f"\n{'='*60}")
    print(f"✅ Conversion Complete!")
    print(f"{'='*60}")
    print(f"Converted: {converted}/{len(patient_folders)}")
    print(f"Skipped: {skipped}/{len(patient_folders)}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")