from os import read
import argparse

from lib.misc import read_config
from datasets.scannet.sem_seg_2d import ScanNetSemSeg2D


def main(args):
    cfg = read_config(args.cfg_path)
    dataset = ScanNetSemSeg2D(cfg['data']['root'], cfg['data']['label_file'],
                                cfg['data']['limit_scans'])
    print(dataset[0])                                        

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('cfg_path', help='Path to cfg')
    args = p.parse_args()

    main(args)