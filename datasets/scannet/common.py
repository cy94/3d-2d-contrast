'''
common and internally used dataset utils
'''

import os
from pathlib import Path
import csv

import numpy as np

from plyfile import PlyData


# 20 valid NYU40 class IDs as present in the TSV file
VALID_CLASSES = [1, 2, 3, 
            4, 5, 6, 7, 
            8, 9, 10, 11, 
            12, 14, 16, 24, 
            28, 33, 34, 36, 
            39] 

# train on 40 classes
VALID_CLASSES_ALL = list(range(1, 41))        

# full class names
CLASS_NAMES = ['wall', 'floor', 'cabinet', 
            'bed', 'chair', 'sofa', 'table', 
            'door', 'window', 'bookshelf', 'picture', 
            'counter', 'desk', 'curtain', 'refridgerator', 
            'shower_curtain', 'toilet', 'sink', 'bathtub', 
            'otherfurniture']

# 40 class names
CLASS_NAMES_ALL = ['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 
'door', 'window', 'bookshelf', 'picture', 'counter', 'blinds', 'desk', 'shelves', 
'curtain', 'dresser', 'pillow', 'mirror', 'floor mat', 'clothes', 'ceiling', 
'books', 'refridgerator', 'television', 'paper', 'towel', 'shower_curtain', 
'box', 'whiteboard', 'person', 'night stand', 'toilet', 'sink', 'lamp', 
'bathtub', 'bag', 'otherstructure', 'otherfurniture', 'otherprop']


# short class names, max 5 chars
CLASS_NAMES_SHORT = ['wall', 'floor', 'cab',
            'bed', 'chair', 'sofa', 'tab',
            'door', 'wind', 'bksf', 'pic',
            'cntr', 'desk', 'curt', 'refg',
            'show', 'toil', 'sink', 'bath',
            'othr']

# all class names, max 5 chars
CLASS_NAMES_ALL_SHORT = ['wall', 'floor', 'cabnt', 'bed', 'chair', 'sofa', 'table', 
'door', 'wind', 'bkslf', 'pic', 'cntr', 'blind', 'desk', 'shelvs', 
'curtn', 'dresr', 'pillo', 'mirrr', 'flmat', 'cloths', 'ceil', 
'books', 'fridg', 'tv', 'paper', 'towel', 'scurt', 
'box', 'wbord', 'persn', 'nstnd', 'toil', 'sink', 'lamp', 
'btub', 'bag', 'struc', 'furn', 'prop']            
    
CLASS_WEIGHTS = [0.0014, 0.0017, 0.0104,
         0.0155, 0.0064, 0.0169, 0.0121, 
         0.0097, 0.013 , 0.0198, 0.0861, 
         0.0875, 0.0223, 0.0213, 0.0821, 
         0.1485, 0.1428, 0.1729, 0.1166, 
         0.0129]

CLASS_WEIGHTS_ALL = [2.6176, 2.7981, 4.6702, 4.9019, 4.3257, 
4.9262, 4.7336, 4.6696, 4.9046, 5.0631, 
5.3792, 5.3724, 5.4624, 5.0697, 5.0463, 
5.1065, 5.3753, 5.3845, 5.4471, 5.4784,
 5.3949, 4.9176, 5.3059, 5.3655, 5.4272, 
 5.4813, 5.449, 5.4293, 5.3499, 5.3159, 
 5.4709, 5.4184, 5.4187, 5.4311, 5.442, 
 5.4013, 5.4573, 5.0262, 4.8396, 4.6717]

CLASS_WEIGHTS_ALL_2D = [2.7306, 2.9981, 4.466, 4.6582, 4.314, 
4.8346, 4.4833, 4.5789, 5.1207, 5.1898, 
5.3727, 5.2827, 5.4743, 4.9813, 5.0898, 
5.1984, 5.3052, 5.3427, 5.4561, 5.4651, 
5.342, 5.3668, 5.3136, 5.2745, 5.4141, 
5.479, 5.3924, 5.3595, 5.3544, 5.3541, 
5.4719, 5.4193, 5.3105, 5.3665, 5.4554, 
5.2974, 5.4478, 5.01, 4.7633, 4.4952]


def nyu40_to_continuous(img, ignore_label=20, num_classes=20):
    '''
    map NYU40 labels 0-40 in VALID_CLASSES to continous labels 0-20
    img: h,w array
    ignore_label: the label to assign all the non-valid classes
    '''
    new_img = img.copy()
    # pick 20 classes or all 40
    valid_classes = VALID_CLASSES if num_classes == 20 else VALID_CLASSES_ALL
    valid_to_cts = dict(zip(valid_classes, range(len(valid_classes))))

    # map valid NYU labelsNYU has classes 1-40
    for nyu_cls in range(41):
        if nyu_cls in valid_classes:
            new_img[img == nyu_cls] = valid_to_cts[nyu_cls]
        else:
            new_img[img == nyu_cls] = ignore_label

    # bugs in labels - some labels are 50 or 149
    # negative values = missing labels
    # larger values = anything else
    new_img[(img < 0) | (img == 50) | (img == 149) | (img > ignore_label)] = ignore_label      

    return new_img

def continous_to_nyu40(img):
    '''
    map continous labels 0-20 NYU40 labels 0-40 in VALID_CLASSES
    img: single image (h, w) or batch (n, h, w)
    '''
    new_img = img.copy()

    cts_to_valid = dict(zip(range(len(VALID_CLASSES)), VALID_CLASSES))

    for cts_cls in range(21):
        new_img[img == cts_cls] = cts_to_valid[cts_cls]
    
    return new_img

def load_ply(path, read_label=False, read_colors=True):
    ply_path = Path(path)
    plydata = PlyData.read(ply_path)
    
    data = plydata.elements[0].data

    coords = np.array([data['x'], data['y'], data['z']], dtype=np.float32).T
    
    if read_colors:
        feats = np.array([data['red'], data['green'], data['blue']], dtype=np.float32).T
    else:
        feats = None
    
    if read_label:  
      labels = np.array(data['label'], dtype=np.int32)
    else:
      labels = None

    return coords, feats, labels

def viz_labels(img):
    '''
    Map NYU40 labels to corresponding RGB values
    img: single image (h, w) or batch (n, h, w)
    '''
    vis_image = np.zeros(img.shape +(3,), dtype=np.uint8)

    color_palette = create_color_palette()

    for idx, color in enumerate(color_palette):
        vis_image[img == idx] = color

    return vis_image

# NYU labels
def create_color_palette():
    colors =  [
       (0, 0, 0), # first index=0
       (174, 199, 232),  # 1.wall
       (152, 223, 138),  # 2.floor
       (31, 119, 180),   # 3.cabinet
       (255, 187, 120),  # 4.bed
       (188, 189, 34),   # 5.chair
       (140, 86, 75),    # 6.sofa
       (255, 152, 150),  # 7.table
       (214, 39, 40),    # 8.door
       (197, 176, 213),  # 9.window
       (148, 103, 189),  # 10.bookshelf
       (196, 156, 148),  # 11.picture
       (23, 190, 207),   # 12.counter
       (178, 76, 76),  
       (247, 182, 210),  # 14.desk
       (66, 188, 102), 
       (219, 219, 141),  # 16.curtain
       (140, 57, 197), 
       (202, 185, 52), 
       (51, 176, 203), 
       (200, 54, 131), 
       (92, 193, 61),  
       (78, 71, 183),  
       (172, 114, 82), 
       (255, 127, 14),   # 24.refrigerator
       (91, 163, 138), 
       (153, 98, 156), 
       (140, 153, 101),
       (158, 218, 229),  # 28.shower curtain
       (100, 125, 154),
       (178, 127, 135),
       (120, 185, 128),
       (146, 111, 194),
       (44, 160, 44),    # 33.toilet
       (112, 128, 144),  # 34.sink
       (96, 207, 209), 
       (227, 119, 194),  # 36.bathtub
       (213, 92, 176), 
       (94, 106, 211), 
       (82, 84, 163),    # 39.otherfurn
       (100, 85, 144)    # last index=40
    ]
    return colors

# mappings not present in the dict will be kept as-is
# map scannet -> nyu40 - using the mapping from the TSV, all the labels should 
# be changed except label 0 (none/unannotated)
def map_labels(arr, label_mapping):
    mapped = np.copy(arr)
    for k,v in label_mapping.items():
        mapped[arr == k] = v
    return mapped

# check if string s represents an int
def represents_int(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False

# read the TSV file
def read_label_mapping(filename, label_from='id', label_to='nyu40id'):
    assert os.path.isfile(filename)
    mapping = dict()
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')
        for row in reader:
            mapping[row[label_from]] = int(row[label_to])
    # if ints convert 
    if represents_int(list(mapping.keys())[0]):
        mapping = {int(k):v for k,v in mapping.items()}
    return mapping

def read_list(path):
    '''
    read list of lines from a file
    path: Path object or str
    '''
    with open(path) as f:
        lines = f.readlines()

    cleanlines = [line.strip() for line in lines]

    return cleanlines

def get_scene_scan_ids(scan_name):
    '''
    scan_name: scene0673_05
    output: 673, 05 (ints)
    '''
    return int(scan_name[5:9]), int(scan_name[10:12])