class Config(object):

    epochs = 2000
    lr = 4e-5
    weight_decay = 1e-5
    dev_data_dir = '/home/liumihan/Desktop/DSM-FinalDesign/驾驶数据集/imgs'
    train_data_dir = '/media/liumihan/HDD_Documents/Datesets/UnityEyes/imgs'
    val_data_dir = '/media/liumihan/HDD_Documents/Datesets/UnityEyes/val_imgs'

    device = 'cuda:0'

    weight_save_dir = './weights/'
    # 如果想从头训练的话就将他置为None
    checkpoint_path = 'weights/ELG_epoch323.pth'

    batch_size = 4

    plot_every_iter = 35


opt = Config()
