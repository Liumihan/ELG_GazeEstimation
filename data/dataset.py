from torch.utils.data import Dataset
import numpy as np
import cv2
from glob import glob
import os
import json
import torch
from torchvision import transforms
from config import opt


class UnityEyeDataset(Dataset):
    def __init__(self, data_dir, eye_size=(160, 96), transform=None):
        """
        :param data_dir: str, path to the data directory
        :param eye_size: tuple, (W, H) the size of the eye image which will be
                                sent to the network
        """
        data_dir = os.path.abspath(data_dir)
        self.json_filenames = glob(data_dir + "/*.json")
        self.eye_size = eye_size
        self.transform = transform

    def __len__(self):
        return len(self.json_filenames)

    def _get_gauss_point(self, size=5):
        """
        生成一个高斯分布的区域,中心点的大小是1
        :param size: int, 3 or 5 高斯点的大小
        :return: gauss
        """
        assert size == 3 or size == 5
        gauss_5x5 = np.array([[1, 4,  7,  4,  1],
                              [4, 16, 26, 16, 5],
                              [7, 26, 41, 26, 7],
                              [4, 16, 26, 16, 4],
                              [1, 4,  7,  4,  1]], dtype=np.float32) * (1 / 41)
        gauss_3x3 = np.array([[1, 2, 1],
                              [2, 4, 1],
                              [1, 2, 1]], dtype=np.float32) * (1 / 4)
        if size == 5:
            return gauss_5x5
        else:
            return gauss_3x3


    def _generate_target_heatmap(self, keypoint, shape, size=0):
        """
        根据关键点生成一个heatmap, 现在生成的还不是高斯分布的样子
        :param keypoint:tuple, (x, y) 关键点的坐标
        :param shape: tuple, (W, H) 想要生成的heatmap的形状
        :return: np.array, heatmap x
        """
        W, H = shape
        x, y = keypoint
        heatmap = np.ones(shape=(H, W), dtype=np.float32) * -1
        if size == 5:
            # heatmap[y-2:y+3, x-2:x+3] = self._get_gauss_point(size=size) # 直接这样地址可能会越界
            for i in range(-2, 3, 1):
                try:
                 heatmap[y+i, x+i] = self._get_gauss_point(size)[i+2, i+2]  # 负数地址赋值错误之后后面会自己改过来
                except IndexError:
                    continue
        elif size == 3:
            # heatmap[y-1:y+2, x-1:x+2] = self._get_gauss_point(size=size) # 同理
            for i in range(-1, 2, 1):
                try:
                    heatmap[y+i, x+i] = self._get_gauss_point(size)[i+1, i+1]
                except IndexError:
                    continue
        else:
            heatmap[y-2:y+3, x-2:x+3] = 1

        return heatmap

    def __getitem__(self, idx):
        json_filename = self.json_filenames[idx]
        img_filename = json_filename[:-5]+".jpg"
        img = cv2.imread(img_filename)
        h, w = img.shape[:2]
        data_file = open(json_filename)
        data = json.load(data_file)

        def process_json_list(json_list):
            ldmks = [eval(s) for s in json_list]
            return np.array([(x, img.shape[0] - y, z) for (x, y, z) in ldmks])

        # 获取关键点的数据
        ldmks_interior_margin = process_json_list(data['interior_margin_2d'])
        ldmks_caruncle = process_json_list(data['caruncle_2d'])
        ldmks_iris = process_json_list(data['iris_2d'])
        ldmks_iris_center = np.mean(ldmks_iris, axis=0)
        ldmks_eyeball_center = np.array([[w/2, h/2]], dtype=np.float32)

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
        ldmks_eyeball_center = ldmks_eyeball_center - np.array([lowx, lowy])
        # 3.开始crop
        eye_image = img[int(lowy):int(high_y), int(lowx):int(high_x), :]
        # 4.准备rescale
        # 4.1 关键点也要相应的缩放
        scale_factor = self.eye_size[0] / eye_image.shape[1]
        ldmks_iris_center = scale_factor * ldmks_iris_center
        ldmks_iris = scale_factor * ldmks_iris
        ldmks_caruncle = scale_factor * ldmks_caruncle
        ldmks_interior_margin = scale_factor * ldmks_interior_margin
        ldmks_eyeball_center = scale_factor * ldmks_eyeball_center
        # 只取其中一部分的点
        ldmks_interior_margin = ldmks_interior_margin[::2, :2]
        ldmks_caruncle = ldmks_caruncle[:, :2]
        ldmks_iris = ldmks_iris[::4, :2]
        ldmks_iris_center = ldmks_iris_center[:2]
        # 4.2 图像进行缩放
        eye_image = cv2.resize(eye_image, self.eye_size)
        # 5. 将图像变为灰度图
        eye_image = cv2.cvtColor(eye_image, cv2.COLOR_BGR2GRAY)
        # 只取其中的一部分点
        # 6. 生成heatmaps
        heatmaps = []
        for ldmk in np.vstack((ldmks_iris_center, ldmks_eyeball_center,   ldmks_iris, ldmks_interior_margin)):
            heatmap = self._generate_target_heatmap(keypoint=(int(ldmk[0]), int(ldmk[1])), shape=(160, 96))
            heatmaps.append(heatmap)
        # 0: iris_center, 1:eyeball_center, 2~9: iris, 10~17: interior
        heatmaps = np.array(heatmaps, dtype=np.float32)
        # 7. 获取geze vector 基于GazeML
        look_vec = np.array(eval(data["eye_details"]['look_vec']))[:3]
        look_vec[1] = -look_vec[1]
        gaze = vector_to_pitchyaw(look_vec.reshape((1, 3))).flatten()
        if gaze[1] > 0.0:
            gaze[1] = np.pi - gaze[1]
        elif gaze[1] < 0.0:
            gaze[1] = -(np.pi + gaze[1])

        sample = {"image": eye_image, "ldmks_interior_margin": ldmks_interior_margin,
                "ldmks_caruncle": ldmks_caruncle, "ldmks_iris": ldmks_iris,
                "ldmks_iris_center": ldmks_iris_center, "ldmks_eyeball_center": ldmks_eyeball_center,
                  "heatmaps": heatmaps, "image_filename": img_filename, "look_vec": look_vec, "gaze": gaze}
        if self.transform:
            sample = self.transform(sample)
        return sample


class ToTensor(object):

    def __call__(self, sample):

        image, heatmaps = sample["image"], sample["heatmaps"]
        look_vec, gaze = sample["look_vec"], sample["gaze"]

        if len(image.shape) < 3:
            image = image[np.newaxis, ...]
        image = torch.from_numpy(image.astype(np.float32))
        heatmaps = torch.from_numpy(heatmaps.astype(np.float32))
        gaze = torch.from_numpy(gaze.astype(np.float32))
        look_vec = torch.from_numpy(look_vec.astype(np.float32))
        sample["image"] = image
        sample["heatmaps"] = heatmaps
        sample["gaze"] = gaze
        sample["look_vec"] = look_vec
        return sample


class ZeroMean(object):

    def __call__(self, sample):
        image = sample["image"]
        image = image / 255.0 * 2 - 1
        sample["image"] = image

        return sample

class Blur(object):
    def __init__(self):
        pass
    def __call__(self, *args, **kwargs):
        pass

class HistogramEqual(object):

    def __init__(self):
        pass
    def __call__(self, *args, **kwargs):
        pass

class RGBNoise(object):

    def __init__(self, difficulty):
        self.difficulty = difficulty

    def __call__(self, *args, **kwargs):
        pass

class LineNoise(object):
    def __init__(self):
        pass
    def __call__(self, *args, **kwargs):
        pass


def visualize_dataset(dataset = UnityEyeDataset(data_dir=opt.dev_data_dir)):

    for sample in dataset:
        img = sample["image"]
        interior_margin = sample["ldmks_interior_margin"] # 8 个
        caruncle = sample["ldmks_caruncle"]
        iris = sample["ldmks_iris"] # 8 个
        iris_center = sample["ldmks_iris_center"]
        eyeball_center = sample["ldmks_eyeball_center"]
        gaze = sample["gaze"]
        look_vec = sample["look_vec"]
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # 这里是为了让图片能够显示彩色的点
        # 画区域关键点以及连线
        # for ldmk in np.vstack((interior_margin, iris, iris_center, eyeball_center)):
        #     img = cv2.circle(img, center=(int(ldmk[0]), int(ldmk[1])),
        #                radius=1, thickness=2, color=(0, 255, 0))
        img = cv2.circle(img, center=(int(eyeball_center[0][0]), int(eyeball_center[0][1])), radius=2, thickness=3, color=(255, 255, 255))
        cv2.polylines(img, np.array([interior_margin], dtype=int), isClosed=True, color=(0,0,255), thickness=2)
        cv2.polylines(img, np.array([iris], dtype=int), isClosed=True, color=(0, 255, 0), thickness=2)
        # 画gaze 的方向
        ox, oy = int(iris_center[0]), int(iris_center[1])
        cv2.arrowedLine(img, (ox, oy), (int(ox+80*look_vec[0]), int(oy+80*look_vec[1])),
                        color=(0, 255, 255), thickness=2)
        cv2.imshow('img', img)

        if cv2.waitKey(0) & 0xff == ord("q"):
            break
    cv2.destroyAllWindows()

def vector_to_pitchyaw(vectors):
    r"""Convert given gaze vectors to yaw (:math:`\theta`) and pitch (:math:`\phi`) angles.

    Args:
        vectors (:obj:`numpy.array`): gaze vectors in 3D :math:`(n\times 3)`.

    Returns:
        :obj:`numpy.array` of shape :math:`(n\times 2)` with values in radians.
    """
    n = vectors.shape[0]
    out = np.empty((n, 2))
    vectors = np.divide(vectors, np.linalg.norm(vectors, axis=1).reshape(n, 1))
    out[:, 0] = np.arcsin(vectors[:, 1])  # theta
    out[:, 1] = np.arctan2(vectors[:, 0], vectors[:, 2])  # phi
    return out

def pitchyaw_to_vector(pitchyaws):
    r"""Convert given yaw (:math:`\theta`) and pitch (:math:`\phi`) angles to unit gaze vectors.

    Args:
        pitchyaws (:obj:`numpy.array`): yaw and pitch angles :math:`(n\times 2)` in radians.

    Returns:
        :obj:`numpy.array` of shape :math:`(n\times 3)` with 3D vectors per row.
    """
    n = pitchyaws.shape[0]
    sin = np.sin(pitchyaws)
    cos = np.cos(pitchyaws)
    out = np.empty((n, 3))
    out[:, 0] = np.multiply(cos[:, 0], sin[:, 1])
    out[:, 1] = sin[:, 0]
    out[:, 2] = np.multiply(cos[:, 0], cos[:, 1])
    return out


def test_transform():
    # ToTensor
    # dataset = UnityEyeDataset("/home/liumihan/Desktop/DSM-FinalDesign/驾驶数据集/imgs", transform=ToTensor())
    # ZeroMean
    dataset = UnityEyeDataset("/home/liumihan/Desktop/DSM-FinalDesign/驾驶数据集/imgs", transform=ZeroMean())
    # Compose
    dataset = UnityEyeDataset("/home/liumihan/Desktop/DSM-FinalDesign/驾驶数据集/imgs",
                              transform=transforms.Compose([ZeroMean(), ToTensor()]))
    for sample in dataset:
        a = sample

if __name__ == "__main__":
    visualize_dataset()
    # test_transform()