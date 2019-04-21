from torch.utils.data import Dataset
import numpy as np
import cv2
from glob import glob
import os
import json

class UnityEyeDataset(Dataset):
    def __init__(self, data_dir, eye_size=(150, 90)):
        """
        :param data_dir: str, path to the data directory
        :param eye_size: tuple, (W, H) the size of the eye image which will be
                                sent to the network
        """
        data_dir = os.path.abspath(data_dir)
        self.json_filenames = glob(data_dir + "/*.json")
        self.eye_size = eye_size
    def __len__(self):
        return len(self.json_filenames)

    def __getitem__(self, idx):
        json_filename = self.json_filenames[idx]
        img = cv2.imread("{}.jpg".format(json_filename[:-5]))
        data_file = open(json_filename)
        data = json.load(data_file)

        def process_json_list(json_list):
            ldmks = [eval(s) for s in json_list]
            return np.array([(x, img.shape[0] - y, z) for (x, y, z) in ldmks])

        # 只取了其中的一部分数据
        ldmks_interior_margin = process_json_list(data['interior_margin_2d'])
        ldmks_caruncle = process_json_list(data['caruncle_2d'])
        ldmks_iris = process_json_list(data['iris_2d'])
        ldmks_iris_center = np.mean(ldmks_iris, axis=0)

        # 对眼睛区域进行裁剪
        # 1.根据GazeML里面的处理方法
        left_corner = np.mean(ldmks_caruncle, axis=0)
        right_corner = ldmks_interior_margin[8]
        eye_width = 1.5 * abs(left_corner[0] - right_corner[0])
        eye_middle = np.mean((np.amin(ldmks_interior_margin, axis=0), # 寻找这个矩形框的中心
                              np.amax(ldmks_interior_margin, axis=0)), axis=0)
        eye_height = self.eye_size[1] / self.eye_size[0] * eye_width
        # 2.从中心向两边寻找裁剪的边界
        lowx = np.max(eye_middle[0] - eye_width / 2, 0)
        lowy = np.max(eye_middle[1] - eye_height / 2, 0)
        high_x = np.min([eye_middle[0] + eye_width / 2, img.shape[1]])
        high_y = np.min([eye_middle[1] + eye_height / 2, img.shape[0]])
        # 2.1 关键点的坐标也得要相应的减
        ldmks_interior_margin = ldmks_interior_margin - np.array([lowx, lowy, 0])
        ldmks_caruncle = ldmks_caruncle - np.array([lowx, lowy, 0])
        ldmks_iris = ldmks_iris - np.array([lowx, lowy, 0])
        ldmks_iris_center = ldmks_iris_center - np.array([lowx, lowy, 0])
        # 3.开始crop
        eye_image = img[int(lowy):int(high_y), int(lowx):int(high_x), :]
        # 4.准备rescale
        # 4.1 关键点也要相应的缩放
        scale_factor = self.eye_size[0] / eye_image.shape[1]
        ldmks_iris_center = scale_factor * ldmks_iris_center
        ldmks_iris = scale_factor * ldmks_iris
        ldmks_caruncle = scale_factor * ldmks_caruncle
        ldmks_interior_margin = scale_factor * ldmks_interior_margin
        # 4.2 图像进行缩放
        eye_image = cv2.resize(eye_image, self.eye_size)
        # 5. 将图像变为灰度图
        eye_image = cv2.cvtColor(eye_image, cv2.COLOR_BGR2GRAY)
        # 只取其中的一部分点
        return {"image": eye_image.copy(), "ldmks_interior_margin": ldmks_interior_margin[::2, :2],
                "ldmks_caruncle": ldmks_caruncle[:, :2], "ldmks_iris": ldmks_iris[::4, :2],
                "ldmks_iris_center": ldmks_iris_center[:2]}

if __name__ == "__main__":
    dataset = UnityEyeDataset("/media/liumihan/Document/DSM的数据集/数据集/瞳孔数据集/UnityEyes/UnityEyes_Windows/imgs")
    for sample in dataset:
        img = sample["image"]
        interior_margin = sample["ldmks_interior_margin"] # 8 个
        caruncle = sample["ldmks_caruncle"]
        iris = sample["ldmks_iris"] # 8 个
        iris_center = sample["ldmks_iris_center"]
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # 这里是为了让图片能够显示彩色的点
        for ldmk in np.vstack((interior_margin, iris, iris_center)):
            img = cv2.circle(img, center=(int(ldmk[0]), int(ldmk[1])),
                       radius=1, thickness=2, color=(0, 255, 0))
        cv2.polylines(img, np.array([interior_margin], dtype=int), isClosed=True, color=(0,0,255), thickness=2)
        cv2.polylines(img, np.array([iris], dtype=int), isClosed=True, color=(0, 255, 0), thickness=2)

        cv2.imshow('img', img)

        if cv2.waitKey(0) & 0xff == ord("q"):
            break
    cv2.destroyAllWindows()
