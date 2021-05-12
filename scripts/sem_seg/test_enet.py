import os
import argparse
from pathlib import Path

from tqdm import tqdm

import numpy as np
import imageio

import torch
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader
from torchvision.transforms import Compose

from lib.misc import read_config
from datasets.scannet.sem_seg_2d import ScanNetSemSeg2D, collate_func
from datasets.scannet.utils import continous_to_nyu40, viz_labels, CLASS_NAMES
from transforms.image_2d import Normalize, TransposeChannels, Resize
from models.sem_seg.enet import ENet2
from models.sem_seg.utils import count_parameters


def main(args):
    cfg = read_config(args.cfg_path)

    # create transforms list
    transforms = []
    if cfg['data']['img_size'] is not None:
        transforms.append(Resize(cfg['data']['img_size']))
    transforms.append(Normalize())
    transforms.append(TransposeChannels())

    t = Compose(transforms)

    test_set = ScanNetSemSeg2D(cfg['data']['root'], cfg['data']['label_file'],
                                cfg['data']['limit_scans'],
                                transform=t)
    if cfg['test']['test_size']:
        test_set = Subset(test_set, range(cfg['test']['test_size']))

    print(f'Test set: {len(test_set)}')

    print(test_set[0]['img_path'])

    ckpt = torch.load(cfg['test']['ckpt'])

    test_loader = DataLoader(test_set, batch_size=cfg['test']['batch_size'],
                            shuffle=False, num_workers=4, collate_fn=collate_func)        

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ENet2(num_classes=21).to(device)
    model.load_state_dict(ckpt['model_state_dict'])

    print(f'Num params: {count_parameters(model)}')

    loss = 0    

    model.eval()
    num_batches = 0

    out_dir = cfg['test']['out_dir']
    os.makedirs(out_dir, exist_ok=True)

    with torch.no_grad():
        for batch in tqdm(test_loader):
            num_batches += 1
            img, label = batch['img'].to(device), batch['label'].to(device)
            out = model(img)
            loss += (F.cross_entropy(out, label) / len(batch))

            preds = out.argmax(dim=1).cpu().numpy()
            label = label.cpu().numpy()

            if args.viz:
                viz_preds(preds, label, out_dir, batch['label_path'])

    print(f'Test loss: {loss / num_batches}')
            
def viz_preds(preds, gt, out_dir, filenames):
    '''
    preds: (n, h, w) predictions over images with continuous labels
    gt: (n, h, w) GT with continuous labels
    out_dir: directory to write colored images to
    filenames: names of files
    '''
    # convert continous labels back to NYU40 labels
    # map NYU40 labels to colors
    pred_nyu, gt_nyu = continous_to_nyu40(preds), continous_to_nyu40(gt)
    pred_rgb, gt_rgb = viz_labels(pred_nyu), viz_labels(gt_nyu)

    out_dir = Path(out_dir)

    for (fname, pred, gt) in zip(filenames, pred_rgb, gt_rgb):
        stem = Path(fname).stem
        pred_path = out_dir / f'{stem}_pred.png'
        gt_path = out_dir / f'{stem}_gt.png'

        imageio.imwrite(pred_path, pred)
        imageio.imwrite(gt_path, gt)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('cfg_path', help='Path to cfg')
    p.add_argument('--viz', dest='viz', help='Visualize the predictions?', action='store_true')
    args = p.parse_args()

    main(args)