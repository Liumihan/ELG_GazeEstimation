from model.elg import ELGNetwork
from data.dataset import UnityEyeDataset, ZeroMean, ToTensor
from config import opt
from torchvision import transforms
import torch
import cv2
from utils.vis_utils import draw_points

def visualize_prediction(dataset=UnityEyeDataset(data_dir=opt.val_data_dir), contrain_eyeball_center=False):
    net = ELGNetwork(output_shape=(17, 96, 160))
    checkpoint = torch.load('weights/ELG_best_so_far.pth')
    net.load_state_dict(checkpoint['net_state_dict'])
    net.cuda()
    net.eval()
    tsf = transforms.Compose([ZeroMean(), ToTensor()])
    for sample in dataset:
        img = sample["image"]
        img_input = tsf(sample)["image"]
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        pred_heatmaps = net.forward(img_input.unsqueeze(0).to(opt.device))
        pred_heatmaps_numpy = pred_heatmaps.cpu().detach().numpy()
        pred_heatmaps_numpy = pred_heatmaps_numpy.squeeze()

        draw_points(img, pred_heatmaps_numpy, contain_eyeball_center=contrain_eyeball_center)

        cv2.imshow('img', img)
        if cv2.waitKey(0) & 0xff == ord('q'):
            break
    cv2.destroyAllWindows()



if __name__ == "__main__":
    visualize_prediction()

