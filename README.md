# Explainability-Guided Multimodal Hierarchical Deep Learning for Robust Brain Tumor Segmentation

This project provides a complete PyTorch implementation template for the paper pipeline:

- preprocessing and normalization
- hierarchical 3D feature encoding
- modality-aware fusion
- decoder-based segmentation prediction
- explainability-guided refinement
- cross-dataset testing
- qualitative, error, and uncertainty visualization

## 1. Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Expected dataset structure

The data discovery script expects one folder per subject with files that contain these tokens in the file name:

- `t1`
- `t1ce`
- `t2`
- `flair`
- `seg`

Example:

```text
BraTS2020/
  BraTS20_Training_001/
    BraTS20_Training_001_t1.nii.gz
    BraTS20_Training_001_t1ce.nii.gz
    BraTS20_Training_001_t2.nii.gz
    BraTS20_Training_001_flair.nii.gz
    BraTS20_Training_001_seg.nii.gz
```

## 3. Create train/val/test splits

```bash
python utils/data_prep.py --root data/BraTS2020 --out_dir data_splits
```

This creates:

- `data_splits/train_subjects.json`
- `data_splits/val_subjects.json`
- `data_splits/test_subjects.json`

## 4. Train

```bash
python train.py --epochs 300
```

Outputs are saved in `outputs/`:

- `best_model.pt`
- `history.json`
- checkpoints
- `config.json`

## 5. Test

```bash
python test.py --checkpoint outputs/best_model.pt --split_json data_splits/test_subjects.json
```

## 6. Cross-dataset testing

After training on BraTS 2020, create a split file for BraTS 2019 and run:

```bash
python utils/data_prep.py --root data/BraTS2019 --out_dir data_splits_2019
python test.py --checkpoint outputs/best_model.pt --split_json data_splits_2019/test_subjects.json --out_json outputs/brats2019_metrics.json
```

## 7. Generate qualitative figures

```bash
python infer_and_visualize.py --checkpoint outputs/best_model.pt --split_json data_splits/test_subjects.json --out_dir outputs/figures --num_cases 5
```

This creates:

- qualitative overlays
- error maps
- uncertainty maps using MC dropout

## 8. Label convention

BraTS labels are remapped internally:

- `0 -> 0`
- `1 -> 1`
- `2 -> 2`
- `4 -> 3`

Region evaluation uses:

- `WT = label > 0`
- `TC = label in {1, 3}`
- `ET = label == 3`

## 9. Main files

- `config.py` — all main settings
- `utils/data_prep.py` — subject discovery and split creation
- `utils/dataset.py` — MONAI dataset and loaders
- `models/encoder.py` — hierarchical encoder
- `models/fusion.py` — modality-aware fusion
- `models/decoder.py` — segmentation decoder
- `models/explainability.py` — attribution and consistency losses
- `train.py` — main training loop
- `test.py` — evaluation script
- `infer_and_visualize.py` — figure generation

## 10. Notes

- The implementation is designed to be readable and modifiable for ablations.
- If your GPU memory is limited, reduce `patch_size` and `base_channels` in `config.py`.
- For production experiments, you may extend the model with deeper encoders, auxiliary heads, or MONAI sliding-window inference.
