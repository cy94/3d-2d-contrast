# thesis

[[_TOC_]]

Contrastive learning in 3D

## Setup
- Create a conda environment and install conda requirements from `env.yaml` (includes pip installs)
- [MinkowskiEngine v0.5.4](https://github.com/NVIDIA/MinkowskiEngine/tree/v0.5.4) through pip

## Dataset
Download the full ScanNet dataset. Optionally select a subset of the scans by 
limiting the file list inside the script.
```
python scripts/download-scannet.py
```
- `.sens` files are used to obtain the color and depth images, and camera matrices
  - Run `python scripts/extract_sens.py` to extract color, depth and matrices from the .sens files.
- PLY files are used to create the voxel grid
- `label` and `label-filt` are the labels for color and depth images
  - Run `python scripts/extract_zip.py` to extract multiple zip files

## ScanNet Baselines
### 2D Semantic Segmentation
RGB semantic segmentation with ENet.

```
python scripts/sem_seg/train_enet.py configs/sem_seg/enet_train.yml
```

### 3D Semantic Segmentation
#### Voxel Grid
Prepare the occupancy grid using 
```
python scripts/sem_seg/prepare_occ_grid.py
```

The maximum grid size can be found with `max_grid_size.py` (not required when training on subvolumes).

```
python scripts/sem_seg/train_occgrid.py configs/sem_seg/occgrid_train.yml
```