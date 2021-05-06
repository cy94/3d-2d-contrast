import os
import csv
import numpy as np


VALID_CLASSES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39] 

def nyu40_to_continuous(img):
    '''
    map NYU40 labels 0-40 in VALID_CLASSES to continous labels 0-20
    '''
    new_img = img.copy()
    valid_to_cts = dict(zip(VALID_CLASSES, range(len(VALID_CLASSES))))

    for nyu_cls in range(41):
        if nyu_cls in VALID_CLASSES:
            new_img[img == nyu_cls] = valid_to_cts[nyu_cls]
        else:
            new_img[img == nyu_cls] = 0

    return new_img

# NYU labels
def create_color_palette():
    colors =  [
       (0, 0, 0), #index=0
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
       (100, 85, 144) #index=40
    ]
    return colors

# map scannet -> nyu40
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