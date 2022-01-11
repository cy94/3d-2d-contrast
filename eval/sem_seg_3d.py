import torch
from collections import defaultdict
from tqdm import tqdm

def get_dataset_split_scans(dataset):
    '''
    dataset: ScanNet2D3DH5 with 'scan_id' and 'scene_id' fields

    return: dict with key = (scene_id, scan_id): [list of sample indices for this scan]
    '''
    splits = defaultdict(list)
    print('Split dataset into scenes')
    for ndx, sample in enumerate(tqdm(dataset)):
        key = (sample['scene_id'], sample['scan_id'])
        splits[key].append(ndx)

    return splits


def gen_predictions(model, loader, ckpt_path):
    '''
    get predictions of a sem seg 3d model on a loader
    model: the model object
    loader: 3d/2d3d data loader
    ckpt_path: path to checkpoint
    '''
    # load the ckpt
    model.load_state_dict(torch.load(ckpt_path, map_location=model.device)['state_dict'])
    # eval mode
    model.eval()
    # store all preds
    all_preds = []

    with torch.no_grad():
        for batch in loader:
            preds = model.common_step(batch)[0]
            all_preds.append(preds)

    return all_preds