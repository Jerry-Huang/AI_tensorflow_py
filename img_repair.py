import os

import cv2
from PIL import Image


def repair(img_dir_path):

    for dir_path, dir_names, file_names in os.walk(img_dir_path):
        # print(dir_path)
        # print(dir_names)
        # print(file_names)
        # print("----")
        for file_name in file_names:
            img_path_name = os.path.join(dir_path, file_name)
            print(img_path_name)

            # img = cv2.imread(img_path_name)
            # os.remove(img_path_name)
            # cv2.imwrite(img_path_name, img)

            img = Image.open(img_path_name)
            img = img.convert('RGB')
            img.verify()
            img.save(img_path_name)


if __name__ == '__main__':
    repair("./catsdogs")