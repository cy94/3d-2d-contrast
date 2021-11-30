'''
project 2d predictions to 3d and get 3d IOU on the val set
'''
from tqdm import tqdm
import torch
import numpy as np
from pathlib import Path

from datasets.scannet.utils import get_scan_name
from datasets.scannet.utils_3d import ProjectionHelper, adjust_intrinsic, load_color, \
             load_intrinsic, load_pose, load_depth
from transforms.image_2d import Normalize
from datasets.scannet.sem_seg_3d import ScanNet2D3DH5
from lib.misc import get_args, read_config
from models.sem_seg.utils import MODEL_MAP_2D
from eval.common import ConfMat
from datasets.scannet.common import VALID_CLASSES


def main(args):
    cfg = read_config(args.cfg_path)

    transform_2d = Normalize()
    dataset = ScanNet2D3DH5(cfg['data'], 'val')
    print(f'Dataset size: {len(dataset)}')

    root = Path(cfg['data']['root'])
    subvol_size = cfg['data']['subvol_size']
    img_size = tuple(cfg['data']['rgb_img_size'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # pick the 2d model
    model = MODEL_MAP_2D[cfg['model']['name_2d']].load_from_checkpoint(cfg['model']['ckpt_2d'])
    model.to(device)

    # init metric
    confmat = ConfMat(cfg['data']['num_classes'])
    not_occupied, skipped = 0, 0

    # go over each 2d3d sample
    for sample_ndx, sample in enumerate(tqdm(dataset)):
        w2g, sceneid, scanid, frames = sample['world_to_grid'], sample['scene_id'], \
                                     sample['scan_id'], sample['frames']
        x = sample['x']
        if (x == 1).sum() == 0:
            not_occupied += 1
            continue       

        subvol_y = sample['y']
        # per-scene basics
        scan_name = get_scan_name(sceneid, scanid)
        frame_ndx = 0                         
        # val set - no frame, then skip
        if frames[frame_ndx] == -1:
            skipped += 1
            continue     
        # load the nearest 1-view
        pose_path = root / scan_name / 'pose' / f'{frames[frame_ndx]}.txt'
        pose = load_pose(pose_path).numpy()
        depth_path = root / scan_name / 'depth' / f'{frames[frame_ndx]}.png' 
        depth = load_depth(depth_path, img_size)
        # get projection
        intrinsic_path = root / scan_name / 'intrinsic/intrinsic_color.txt'
        intrinsic = load_intrinsic(intrinsic_path)
        # adjust for smaller image size
        intrinsic = adjust_intrinsic(intrinsic, [1296, 968], img_size)

        projection = ProjectionHelper(
                    intrinsic, 
                    0.4, 4.0,
                    img_size,
                    subvol_size, cfg['data']['voxel_size']
                )

        proj = projection.compute_projection(torch.Tensor(depth), torch.Tensor(pose), torch.Tensor(w2g))
        if proj is None: 
            continue
        proj3d, proj2d = proj
        num_inds = proj3d[0]

        ind3d = proj3d[1:1+num_inds]
        ind2d = proj2d[1:1+num_inds]

        # load rgb
        rgb_path = root / scan_name / 'color' / f'{frames[frame_ndx]}.jpg' 
        # load H, W, C
        rgb = load_color(rgb_path, img_size).transpose(1, 2, 0)
        # apply transform on rgb and back to C,H,W
        rgb = transform_2d({'x': rgb})['x'].transpose(2, 1, 0)
        # convert to tensor, add batch dim, get (1,C,H,W)
        rgb = torch.Tensor(rgb).unsqueeze(0).to(device)

        # get preds on this view
        pred2d = model(rgb).argmax(dim=1).squeeze().cpu()
        labels = pred2d.view(-1)[ind2d]
        # get the label volume - DHW
        # create empty volume with zeros
        output = torch.zeros(subvol_size[2], \
                            subvol_size[1], \
                            subvol_size[0], dtype=int)
        # project the preds to 3d
        output.view(-1)[ind3d] = labels.T
        # update metric
        subvol_y = torch.LongTensor(subvol_y).permute(2, 1, 0)
        confmat.update(output, subvol_y)

    # calculate metric
    class_subset = np.array(VALID_CLASSES) - 1
    iou_subset = np.nanmean(confmat.ious[class_subset])
    acc_subset = np.nanmean(confmat.accs[class_subset])
    print(f'iou: {iou_subset:.3f}, acc: {acc_subset:.3f}')

  
if __name__ == '__main__':
    args = get_args()
    main(args)