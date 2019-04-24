import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from data.dataset import UnityEyeDataset, ToTensor, ZeroMean
from model.elg import ELGNetwork
from torch.nn import MSELoss
from torch import optim
from config import opt
from utils.vis_utils import vis_images
import visdom  # 使用visdom来进行训练时期的可视化
import math


def train():
    vis = visdom.Visdom(env='ElgNet')
    tsf = transforms.Compose([ZeroMean(), ToTensor()])
    train_dataset = UnityEyeDataset(data_dir=opt.dev_data_dir, eye_size=(160, 96), transform=tsf)
    train_dataset_for_vis = UnityEyeDataset(data_dir=opt.dev_data_dir, eye_size=(160, 96))
    sample_num = len(train_dataset)
    iter_per_epoch = math.ceil(sample_num / opt.batch_size)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4)

    val_dataset = UnityEyeDataset(data_dir=opt.dev_data_dir, eye_size=(160, 96))

    net = ELGNetwork(HGNet_num=3, input_shape=(1, 96, 160), output_shape=(18, 96, 160), feature_channels=64)
    if opt.checkpoint_path:
        checkpoint = torch.load(opt.checkpoint_path)
        net.load_state_dict(checkpoint['net_state_dict'])
    net.float().to(opt.device)
    net.train()
    criterion = MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)

    best_epoch_loss = 10000
    v_max, v_min = 0.0, 0.0

    for epoch in range(opt.epochs):
        epoch_loss = 0.0
        for itr, batch in enumerate(train_dataloader):
            inputs = batch["image"].to(opt.device)
            heatmaps_targets = batch["heatmaps"].to(opt.device)

            optimizer.zero_grad()
            net.train()
            heatmaps_pred = net.forward(inputs)
            # 多loss 回传, 效果不好放弃使用了
            loss = criterion(heatmaps_targets, heatmaps_pred)

            loss.backward()
            optimizer.step()
            epoch_loss = (epoch_loss * itr + loss.item()) / (itr + 1)

            # 统计最大值和最小值
            v_max = torch.max(heatmaps_pred)
            v_min = torch.min(heatmaps_pred)

            print('[Epoch {} / {}, iter {} / {}] train_loss: {:.6f} max: {:.4f} min: {:.4f}'.format(
                epoch, opt.epochs, itr * opt.batch_size, sample_num, loss, v_max, v_min
            ))
            # 可视化

            vis.line(Y=np.array([loss.item()]), X=np.array([(epoch * iter_per_epoch) + (itr + 1)]), win='Train_loss',
                     update='append' if (epoch + 1) * (itr + 1) > 0 else None, opts=dict(title='train_loss'))
            vis.line(Y=np.array([v_max.item()]), X=np.array([(epoch * iter_per_epoch) + (itr + 1)]), win='v_max',
                     update='append' if (epoch + 1) * (itr + 1) != 0 else None, opts=dict(title='v_max'))
            vis.line(Y=np.array([v_min.item()]), X=np.array([(epoch * iter_per_epoch) + (itr + 1)]), win="v_min",
                     update='append' if (epoch + 1) * (itr + 1) != 0 else None, opts=dict(title='v_min'))
            # 图片显示
            if itr % opt.plot_every_iter == 0:

                random_val_idx = np.random.randint(0, len(val_dataset))
                random_train_idx = np.random.randint(0, len(train_dataset_for_vis))
                vis_val_sample = val_dataset[random_val_idx]
                vis_train_sample = train_dataset_for_vis[random_train_idx]
                vis_images(vis_val_sample, net, vis, title='val_sample')
                vis_images(vis_train_sample, net, vis, title='train_sample')

        vis.line(Y=np.array([epoch_loss]), X=np.array([epoch]),
                 win='epoch_loss', update='append' if epoch > 0 else None)
        print('epoch_loss: {:.5f} old_best_epoch_loss: {:.5f}'.format(epoch_loss, best_epoch_loss))
        if epoch_loss < best_epoch_loss:
            print('epoch loss < best_epoch_loss, save these weights.')
            best_epoch_loss = epoch_loss
            torch.save({
                'epoch': epoch,
                'epoch_loss': epoch_loss,
                'v_max': v_max,
                'v_min': v_min,
                'net_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, opt.weight_save_dir+'ELG_epoch{}.pth'.format(epoch))


if __name__ == "__main__":
    # 打开visdom 服务器
    # os.popen('python -m visdom.server')
    train()
