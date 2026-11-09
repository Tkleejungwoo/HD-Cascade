# HD-Cascade-Dev

HD-loss training extensions built on
[nnU-Net v2](https://github.com/MIC-DKFZ/nnUNet).

This repository contains the custom loss, HD-weight scheduling, custom trainers,
and a polynomial learning-rate scheduler. It does **not** contain
the nnU-Net framework or patient data. The setup command installs nnU-Net as a dependency.

## Install

Clone this repository and activate a dedicated Python 3.11 environment. 

```bash
git clone https://github.com/Tkleejungwoo/HD-Cascade-Dev.git
cd HD-Cascade-Dev
python3.11 -m venv .venv
source .venv/bin/activate
bash shell/setup.shell
```

The setup command automatically:

1. Installs the dependencies listed in `requirements.txt`, including nnU-Net 2.6.2 and postprocessing dependencies.
2. Finds `nnunetv2` inside the active Python environment and installs the custom
   loss, schedulers, and trainers there.
3. Checks module imports, nnU-Net trainer discovery, and the HD schedule boundaries.

**Users do not move or copy any loss or trainer files manually.** The setup script
does not start training or inference.

For a specific CUDA/CPU build, install the appropriate PyTorch 2.7.1 build from
https://pytorch.org/get-started/locally/ before running setup. An already installed
matching build satisfies the pinned PyTorch requirement. A working GPU driver is
required for CUDA training; CPU-only environments can still pass setup checks.

Alternatively, obtain the official nnU-Net source from
https://github.com/MIC-DKFZ/nnUNet and install the matching **2.6.2** version in
editable mode. The installer above also supports an editable nnU-Net install.

The installer copies only the custom extension modules into the installed
`nnunetv2` package, because nnU-Net 2.6.2 discovers custom trainers there. It does
not copy a framework checkout and does not replace the built-in trainer, loss,
or `polylr.py`. Existing identical files are skipped; differing destination files
cause the installer to stop before writing anything.

## Training shell

Prepare a dataset and plans using the official nnU-Net documentation. Set
`nnUNet_raw`, `nnUNet_preprocessed`, and `nnUNet_results` to your own directories.
The shell scripts are portable Bash scripts for an installed Python environment;
they do not depend on a particular Slurm cluster or container image. Settings
near the top of each script can be edited or overridden through environment
variables.

```bash
export nnUNet_raw=/path/to/nnUNet_raw
export nnUNet_preprocessed=/path/to/nnUNet_preprocessed
export nnUNet_results=/path/to/nnUNet_results

DATASET=Dataset001_Temp FOLD=0 bash shell/train.shell all

# Run stages separately:
bash shell/train.shell lowres
bash shell/train.shell cascade
bash shell/train.shell base-hd
```

`all` trains **all five lowres folds** before the selected cascade fold. Lowres
validation exports out-of-fold predictions for the next stage. Training only
lowres fold 0 does not provide previous-stage masks for every case required by
cascade fold 0. The script checks the links between the lowres and cascade configurations,
preprocessing directories, and next-stage mask coverage before cascade training. Preprocess both lowres and
full-resolution data first; plans must define lowres/cascade links. Dataset
preparation and planning remain part of the official nnU-Net workflow.

## Model Weights

Download the pretrained model weights from [Google Drive](https://drive.google.com/file/d/1KIPTsrkuNEspdnISSkSYzcloECA6qhRc/view?usp=sharing).

## Prediction

Download the pretrained weights ZIP and install it:

```bash
python scripts/install_pretrained.py --archive /path/to/hd-cascade-normalized-fold0-v2.zip
```

The installer verifies SHA-256 hashes and places both stages under `nnUNet_results`
(default: `data/nnUNet_results`). No model files need to be moved manually.
These are inference-only exports of the lowres and cascade best checkpoints;
network tensors are preserved, and optimizer and training state are omitted.
Mirroring test-time augmentation is disabled during prediction. The prediction shell is preconfigured for the supplied weights.

```bash
bash shell/predict.shell /path/to/nii /path/to/predictions
```

The prediction shell creates:

```text
predictions/
  input/    # channel-suffixed symlinks to the original CT files
  lowres/   # low-resolution predictions
  cascade/  # raw cascade predictions
  mask/     # final masks, largest 18-connected component per organ
```

The postprocessing code can also be used on its own:

```bash
python scripts/postprocess_largest_components.py \
  --input-dir /path/to/cascade --output-dir /path/to/mask
```

## Attribution and licensing

nnU-Net remains an external dependency. Cite the nnU-Net publication and retain
its upstream attribution. `LICENSES/nnUNet-Apache-2.0.txt` preserves the upstream
license text for nnU-Net-derived components, including the LR scheduler and
trainer-derived implementation. See `THIRD_PARTY_NOTICES.md`.

