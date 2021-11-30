'''
project 2d predictions to 3d and get 3d IOU on the val set
'''
from tqdm import tqdm
import torch
import numpy as np
from pathlib import Path
import imageio
import cv2
import argparse

from datasets.scannet.utils import get_scan_name
from datasets.scannet.utils_3d import ProjectionHelper, adjust_intrinsic, load_color, \
             load_intrinsic, load_pose, load_depth
from transforms.image_2d import Normalize
from datasets.scannet.sem_seg_3d import ScanNet2D3DH5
from lib.misc import get_args, read_config
from models.sem_seg.utils import MODEL_MAP_2D
from eval.common import ConfMat
from datasets.scannet.common import VALID_CLASSES
from datasets.scannet.common import read_label_mapping, map_labels, nyu40_to_continuous


def main(args):
    cfg = read_config(args.cfg_path)

    transform_2d = Normalize()
    dataset = ScanNet2D3DH5(cfg['data'], 'val')
    print(f'Dataset size: {len(dataset)}')

    scannet_to_nyu40 = read_label_mapping(cfg['data']['label_file'])

    root = Path(cfg['data']['root'])
    subvol_size = cfg['data']['subvol_size']
    img_size = tuple(cfg['data']['rgb_img_size'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not args.use_gt:
        # pick the 2d model
        model = MODEL_MAP_2D[cfg['model']['name_2d']].load_from_checkpoint(cfg['model']['ckpt_2d'])
        model.eval()
        model.to(device)

    # init metric
    confmat = ConfMat(cfg['data']['num_classes'])
    if not args.use_gt:
        confmat2d = ConfMat(cfg['data']['num_classes'])
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

        label_path = root / scan_name / 'label-filt' / f'{frames[frame_ndx]}.png' 
        label_scannet = np.array(imageio.imread(label_path))
        label_nyu40 = map_labels(label_scannet, scannet_to_nyu40)
        # map from NYU40 labels to 0-39 + 40 (ignored) labels, H,W
        y2d = nyu40_to_continuous(label_nyu40, ignore_label=cfg['data']['num_classes'], 
                                            num_classes=cfg['data']['num_classes'])
        # resize label image here using the proper interpolation - no artifacts  
        # dims: H,W                                     
        y2d = cv2.resize(y2d, img_size, interpolation=cv2.INTER_NEAREST)
        y2d = torch.LongTensor(y2d.astype(np.int32))

        if args.use_gt:
            # labels at the required locations, index into H,W image
            labels = y2d.view(-1)[ind2d]
            # 40/ignore labels in 2d -> dont project 
            invalid_2d = (labels == cfg['data']['num_classes'])
            valid_2d = torch.logical_not(invalid_2d)
            
            ind2d = ind2d[valid_2d]
            ind3d = ind3d[valid_2d]

            labels = y2d.view(-1)[ind2d]
        else:
            # load rgb
            rgb_path = root / scan_name / 'color' / f'{frames[frame_ndx]}.jpg' 
            # load H, W, C
            rgb = load_color(rgb_path, img_size).transpose(1, 2, 0)
            # apply transform on rgb and back to C,H,W
            rgb = transform_2d({'x': rgb})['x'].transpose(2, 0, 1)
            # convert to tensor, add batch dim, get (1,C,H,W)
            rgb = torch.Tensor(rgb).unsqueeze(0).to(device)

            # get preds on this view
            with torch.no_grad():
                pred2d = model(rgb).argmax(dim=1).squeeze().cpu()
            confmat2d.update(pred2d, y2d)
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

    iou_subset2d = np.nanmean(confmat2d.ious[class_subset])
    acc_subset2d = np.nanmean(confmat2d.accs[class_subset])
    print(f'2d iou: {iou_subset2d:.3f}, 2d acc: {acc_subset2d:.3f}')

  
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('cfg_path', help='Path to cfg')
    p.add_argument('--use-gt', action='store_true', dest='use_gt', 
                    default=False, help='Use ground truth labels')
    args = p.parse_args()
    main(args)