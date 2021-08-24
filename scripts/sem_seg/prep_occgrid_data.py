
from datasets.scannet.sem_seg_3d import ScanNetPLYDataset
from transforms.grid_3d import RandomRotate
from tqdm import tqdm
import argparse
from pathlib import Path
from lib.misc import read_config
import h5py
import numpy as np
from torchvision.transforms import Compose

def main(args):
    cfg = read_config(args.cfg_path)

    transforms = []

    if args.split == 'train': 
        transforms.append(RandomRotate())
    t = Compose(transforms)

    dataset = ScanNetPLYDataset(cfg['data'], transform=t, split=args.split,
                                  full_scene=True)
    print(f'Dataset: {len(dataset)}')

    # sample N subvolumes from each scene, save to file
    print('Writing to file...')
    
    out_path = Path(args.out_path)
    out_path.parent.mkdir(exist_ok=True)

    subvols_per_scene = cfg['data']['subvols_per_scene']
    n_samples = len(dataset) * subvols_per_scene
    subvol_size = tuple(cfg['data']['subvol_size'])

    with h5py.File(out_path, 'w') as f:
        x_ds = f.create_dataset('x', (n_samples,) + subvol_size, dtype=np.float32)
        y_ds = f.create_dataset('y', (n_samples,) + subvol_size, dtype=np.int16)
        
        # iterate over each scene, read it only once
        for scene_ndx, scene in enumerate(tqdm(dataset)):
            scene_x, scene_y = scene['x'], scene['y']
            # sample N subvols from this scene
            for ndx in tqdm(range(subvols_per_scene), leave=False):
                subvol_x, subvol_y = dataset.sample_subvol(scene_x, scene_y)
                sample_ndx = scene_ndx * subvols_per_scene + ndx
                x_ds[sample_ndx], y_ds[sample_ndx] = subvol_x, subvol_y

    print('Read from file and iterate..')
    with h5py.File(out_path, 'r') as f:
        print('Samples:', len(f['x']))
        for ndx, x in enumerate(tqdm(f['x'])):
            y = f['y'][ndx]

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('cfg_path', help='Path to cfg')
    p.add_argument('split', help='Split to be used: train/val')
    p.add_argument('out_path', help='Path to output hdf5 file')
    args = p.parse_args()

    main(args)