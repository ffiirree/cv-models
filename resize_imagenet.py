import os
import argparse
import shutil
from cvm.utils import *
import cv2
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='ImageNet Resizing')
    parser.add_argument('--src', type=str, default='/datasets/ILSVRC2012')
    parser.add_argument('--dst', type=str, default='/datasets/ILSVRC2012_R')
    parser.add_argument('--max-size', type=int, default=256)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print(args)

    dirs = os.listdir(os.path.join(args.src, 'train'))
    dirs.sort()
    for i, cls in enumerate(dirs):
        files = os.listdir(os.path.join(args.src, 'train', cls))

        if not os.path.exists(os.path.join(args.dst, 'train', cls)):
            os.makedirs(os.path.join(args.dst, 'train', cls))

        for f in tqdm(files, desc=f'Resizing [{i:>3}/{len(dirs)}]', unit='images', leave=False, ascii=True):
            src_file, dst_file = os.path.join(args.src, 'train', cls, f), os.path.join(args.dst, 'train', cls, f)
            image = cv2.imread(src_file)

            if min(image.shape[0], image.shape[1]) <= args.max_size:
                shutil.copyfile(src_file, dst_file)
            else:
                if image.shape[0] < image.shape[1]:
                    size = (int((image.shape[1] / image.shape[0]) * args.max_size), args.max_size)
                else:
                    size = (args.max_size, int((image.shape[0] / image.shape[1]) * args.max_size))

                image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
                cv2.imwrite(dst_file, image)
