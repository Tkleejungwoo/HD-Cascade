"""Retain the largest 18-connected component of each organ in a 3D NIfTI mask."""

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
from skimage.measure import label, regionprops


def keep_largest_component_per_label(mask_data):
    """Match the existing per-label LCC algorithm: connectivity=2 in 3D (18 neighbors)."""
    mask_data = np.asarray(mask_data)
    if mask_data.ndim != 3:
        raise ValueError(f'Expected a 3D organ mask, got shape {mask_data.shape}')
    values = np.unique(mask_data)
    
    if not set(values.tolist()) <= {0, 1, 2, 3}:
        raise ValueError(f'Expected labels 0/1/2/3, found {values.tolist()}')
    clean_mask = np.zeros_like(mask_data, dtype=np.uint8)
    
    for value in values:
        if value == 0:
            continue
        labeled_mask, num_features = label(
            (mask_data == value).astype(np.uint8), return_num=True, connectivity=2,
        )
        if num_features:
            largest_region = max(regionprops(labeled_mask), key=lambda region: region.area)
            clean_mask[labeled_mask == largest_region.label] = value
    return clean_mask


def process_directory(input_dir, output_dir, overwrite=False):
    input_dir, output_dir = Path(input_dir).resolve(), Path(output_dir).resolve()
    if input_dir == output_dir:
        raise ValueError('Use separate raw-prediction and postprocessed output directories')
    files = sorted(input_dir.glob('*.nii.gz'))
    
    if not files:
        raise FileNotFoundError(f'No NIfTI masks found: {input_dir}')
    
    if not overwrite:
        existing = [output_dir / p.name for p in files if (output_dir / p.name).exists()]
        if existing:
            raise FileExistsError(f'Output already exists; use --overwrite: {existing[0]}')
    
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, path in enumerate(files, 1):
        image = nib.load(path)
        cleaned = keep_largest_component_per_label(np.asanyarray(image.dataobj))
        
        result = nib.Nifti1Image(cleaned, image.affine, image.header.copy())
        result.set_data_dtype(np.uint8)
        nib.save(result, output_dir / path.name)
        print(f'[{index}/{len(files)}] Saved {path.name}', flush=True)
    return len(files)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', '--input_path', type=Path, required=True)
    parser.add_argument('--output-dir', '--output_path', type=Path, required=True)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    count = process_directory(args.input_dir, args.output_dir, args.overwrite)
    print(f'Postprocessing complete: {count} masks in {args.output_dir}', flush=True)


if __name__ == '__main__':
    main()
