import argparse
import os
from dataset import get_loader
from deep_supervision_solver import Solver


def get_test_info(sal_mode='e'):
    if sal_mode == 'e':
        image_root = '/home/omnisky/diskB/datasets/SOD_Datasets/ECSSD/images/'
        image_source = '/home/omnisky/diskB/datasets/SOD_Datasets/ECSSD/test.lst'
    elif sal_mode == 'p':
        image_root = '/home/omnisky/diskB/datasets/SOD_Datasets/PASCALS/images/'
        image_source = '/home/omnisky/diskB/datasets/SOD_Datasets/PASCALS/test.lst'
    elif sal_mode == 'd':
        image_root = '/home/omnisky/diskB/datasets/SOD_Datasets/DUTOMRON/images/'
        image_source = '/home/omnisky/diskB/datasets/SOD_Datasets/DUTOMRON/test.lst'
    elif sal_mode == 'h':
        image_root = '/home/omnisky/diskB/datasets/SOD_Datasets/HKU-IS/images/'
        image_source = '/home/omnisky/diskB/datasets/SOD_Datasets/HKU-IS/test.lst'
    elif sal_mode == 's':
        image_root = '/home/omnisky/diskB/datasets/SOD_Datasets/SOD/images/'
        image_source = '/home/omnisky/diskB/datasets/SOD_Datasets/SOD/test.lst'
    elif sal_mode == 't':
        image_root = '/home/omnisky/diskB/datasets/SOD_Datasets/DUTS-TE/images/'
        image_source = '/home/omnisky/diskB/datasets/SOD_Datasets/DUTS-TE/test.lst'

    return image_root, image_source


def main(config):
    if config.mode == 'train':
        train_loader = get_loader(config)
        run = 0
        while os.path.exists("%s/run-%d" % (config.save_folder, run)):
            run += 1
        os.makedirs("%s/run-%d" % (config.save_folder, run))
        os.makedirs("%s/run-%d/models" % (config.save_folder, run))
        config.save_folder = "%s/run-%d" % (config.save_folder, run)
        train = Solver(train_loader, None, config)
        train.train()

        # save hyperparameters
        with open('%s/args.txt' % (config.save_folder), 'w') as f:
            for arg in vars(config):
                print('%s: %s' % (arg, getattr(config, arg)), file=f)

    elif config.mode == 'test':
        config.test_root, config.test_list = get_test_info(config.sal_mode)
        test_loader = get_loader(config, mode='test')
        if not os.path.exists(config.test_fold): os.makedirs(config.test_fold)
        test = Solver(None, test_loader, config)
        test.test()
    else:
        raise IOError("illegal input!!!")


if __name__ == '__main__':

    vgg_path = './pretrained/vgg16_20M.pth'
    resnet_path = './pretrained/resnet50_caffe.pth'

    parser = argparse.ArgumentParser(description="Attention Guided Boundary-aware Network")

    # Hyper-parameters
    parser.add_argument('--n_color', type=int, default=3)
    parser.add_argument('--lr', type=float, default=3e-5)  # Learning rate resnet:5e-5, vgg:1e-4
    parser.add_argument('--wd', type=float, default=0.0005)  # Weight decay
    parser.add_argument('--no-cuda', dest='cuda', action='store_false')

    # Training settings
    parser.add_argument('--arch', type=str, default='resnet')  # resnet or vgg
    parser.add_argument('--pretrained_model', type=str, default=resnet_path)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--num_thread', type=int, default=8)
    parser.add_argument('--load', type=str, default='')
    parser.add_argument('--save_folder', type=str, default='./deep_supervision_results')
    parser.add_argument('--epoch_save', type=int, default=50)
    parser.add_argument('--iter_size', type=int, default=10)
    parser.add_argument('--show_every', type=int, default=500)
    parser.add_argument('--weight', type=float, default=5)
    parser.add_argument('--sigma', type=float, default=4)

    # Train data
    parser.add_argument('--train_root', type=str, default='./data/DUTS-TR')
    parser.add_argument('--train_list', type=str,
                        default='./data/DUTS-TR/mypair.lst')

    # Testing settings
    parser.add_argument('--model', type=str, default=None)  # Snapshot
    parser.add_argument('--test_fold', type=str, default=None)  # Test results saving folder
    parser.add_argument('--sal_mode', type=str, default='e')  # Test image dataset

    # Misc
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])

    # Device
    config = parser.parse_args()


    if not os.path.exists(config.save_folder):
        os.mkdir(config.save_folder)

    # Get test set info
    test_root, test_list = get_test_info(config.sal_mode)
    config.test_root = test_root
    config.test_list = test_list

    main(config)
