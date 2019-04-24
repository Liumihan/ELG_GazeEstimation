import glob
import os
import cv2
import pprint
import numpy as np

def find_wrong_imgs(img_dir):
    """
    找出该文件夹下面所有对应错误图片的json
    :param img_dir: str, path to the data directory
    :return: void
    """
    json_filenames = glob.glob(img_dir + "/*.json")
    wrong_img = []
    for json_filename in json_filenames:
        img_name = json_filename[:-5] + ".jpg"
        img = cv2.imread(img_name)
        if img is None:
            wrong_img.append(img_name)
            os.popen('rm -f {}'.format(json_filename))
            print('{} has no correspond img, deleted'.format(json_filename))
    print('there are {} wrong imgs'.format(len(wrong_img)))
    pprint.pprint(wrong_img)


if __name__ == "__main__":
    find_wrong_imgs('/media/liumihan/HDD_Documents/Datesets/UnityEyes/imgs')
