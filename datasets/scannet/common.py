'''
common and internally used dataset utils
'''

import os
import numpy as np

import csv

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
            'counter', 'desk', 'curtain', 'refrigerator', 
            'shower_curtain', 'toilet', 'sink', 'bathtub', 
            'otherfurn']

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

CLASS_WEIGHTS_ALL = []


def nyu40_to_continuous(img, ignore_label=20):
    '''
    map NYU40 labels 0-40 in VALID_CLASSES to continous labels 0-20
    img: h,w array
    ignore_label: the label to assign all the non-valid classes
    '''
    new_img = img.copy()
    valid_to_cts = dict(zip(VALID_CLASSES, range(len(VALID_CLASSES))))

    for nyu_cls in range(41):
        if nyu_cls in VALID_CLASSES:
            new_img[img == nyu_cls] = valid_to_cts[nyu_cls]
        else:
            new_img[img == nyu_cls] = ignore_label

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

# map scannet -> nyu40
# mappings not present in the dict will be kept as-is
def map_labels(arr, label_mapping):
    mapped = np.copy(arr)
    for k,v in label_mapping.items():
        mapped[arr == k] = v
    return mapped.astype(np.uint8)

# if string s represents an int
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