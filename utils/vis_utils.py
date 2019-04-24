from data.dataset import ZeroMean, ToTensor
from torchvision import transforms
import cv2
from utils.utils import get_peek_points
import numpy as np
from config import opt


def vis_images(sample, net, vis, title):
    """
    生成一组visdom中显示的images
    :param sample:
    :return: tuple(np.ndarray, np.ndarray), 两张图片前一张是有标记的， 后一张是原图
    """
    tsf = transforms.Compose([ZeroMean(), ToTensor()])
    img = sample['image']
    vis.image(img[np.newaxis, ...], win='ori_img' + title, opts={'title': title})
    img_input = tsf(sample)['image']
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    net.eval()
    pred_heatmaps = net.forward(img_input.unsqueeze(0).to(opt.device))
    pred_heatmaps_numpy = pred_heatmaps.cpu().detach().numpy()
    pred_heatmaps_numpy = pred_heatmaps_numpy.squeeze()

    draw_points(img, pred_heatmaps_numpy)

    vis.image(img.transpose(2, 0, 1), win='dotted_img' + title, opts={'title': title})


def draw_points(img, heatmaps, contain_eyeball_center=True):
    """
    根据所给的图片和heatmaps 画出关键点和连线来
    :param img: np.array, 通过opencv 读入的图片, 应该是一个BGR图片
    :param heatmaps: np.array, 从网络中得到的heatmaps,转换成的numpy 的格式
    :return: img: np.array, 已经画好了的图像
    """
    ldmks_iris = []
    ldmks_interior = []
    if contain_eyeball_center:
        for i, heatmap in enumerate(heatmaps):
            x, y = get_peek_points(heatmap)
            # 在图像上面画出来
            if i == 0:
                # 第0位是iris center
                cv2.circle(img, center=(int(x), int(y)), radius=1, thickness=2, color=(255, 0, 0))
            elif i == 1:
                # 第1位是 eyeball center
                cv2.circle(img, center=(int(x), int(y)), radius=2, thickness=2, color=(255, 255, 255))
            elif i > 1 and i <= 9:
                # 2~9位是iris_edge
                cv2.circle(img, center=(int(x), int(y)), radius=1, thickness=2, color=(0, 255, 0))
                ldmks_interior.append((int(x), int(y)))
            else:
                # 大于9的是interior edge
                cv2.circle(img, center=(int(x), int(y)), radius=1, thickness=2, color=(0, 0, 255))
                ldmks_iris.append((int(x), int(y)))
    else:
        for i, heatmap in enumerate(heatmaps):
            x, y = get_peek_points(heatmap)
            # 在图像上面画出来
            if i == 0:
                # 第0位是iris center
                cv2.circle(img, center=(int(x), int(y)), radius=1, thickness=2, color=(255, 0, 0))
            elif i >= 1 and i <= 8:
                # 1~8位是iris edge
                cv2.circle(img, center=(int(x), int(y)), radius=1, thickness=2, color=(0, 255, 0))
                ldmks_interior.append((int(x), int(y)))
            else:
                # 大于8的是interior edge
                cv2.circle(img, center=(int(x), int(y)), radius=1, thickness=2, color=(0, 0, 255))
                ldmks_iris.append((int(x), int(y)))
    # 画iris
    cv2.polylines(img, np.array([ldmks_iris]), True, color=(0, 0, 255), thickness=1)
    # draw interior
    cv2.polylines(img, np.array([ldmks_interior]), True, color=(0, 255, 0), thickness=1)
    return img

def visualize_heatmaps(heatmaps, image_filename):
    heatmaps = heatmaps.squeeze()
    ori_image = cv2.imread(image_filename)
    cv2.imshow('ori_image', ori_image)
    for heatmap in heatmaps:
        max = heatmap.max()
        frame = np.where(heatmap == max, np.ones(heatmap.shape) * 255, np.zeros(heatmap.shape))
        cv2.imshow('frame', frame)
        if cv2.waitKey(0) & 0xff == ord("q"):
            break
    cv2.destroyAllWindows()