
import argparse
import os
from pathlib import Path
from tqdm import tqdm

from lib.ScanNet.SensReader.python.SensorData import SensorData

def main(args):
    root = Path(args.scannet_dir)

    for scan_id in tqdm(sorted(os.listdir(root))[5:100]):
        out_path = root / scan_id
        sens_file = root / scan_id / f'{scan_id}.sens'

        sd = SensorData(sens_file)

        if args.export_depth_images:
            sd.export_depth_images(os.path.join(out_path, 'depth')) 
        if args.export_color_images:
            sd.export_color_images(os.path.join(out_path, 'color')) 
        if args.export_poses:
            sd.export_poses(os.path.join(out_path, 'pose'))
        if args.export_intrinsics:
            sd.export_intrinsics(os.path.join(out_path, 'intrinsic'))


if __name__ == '__main__':
    # params
    parser = argparse.ArgumentParser()
    # data paths
    parser.add_argument('scannet_dir', help='path to sens file to read')
    parser.add_argument('--export_depth_images', dest='export_depth_images', action='store_true')
    parser.add_argument('--export_color_images', dest='export_color_images', action='store_true')
    parser.add_argument('--export_poses', dest='export_poses', action='store_true')
    parser.add_argument('--export_intrinsics', dest='export_intrinsics', action='store_true')
    parser.set_defaults(export_depth_images=False, export_color_images=False, export_poses=False, export_intrinsics=False)

    args = parser.parse_args()

    assert any((args.export_depth_images, args.export_color_images, args.export_poses, 
                args.export_intrinsics)), 'Pick any file to be exported!'
    main(args)