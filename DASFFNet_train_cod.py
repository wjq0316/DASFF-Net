import logging
import sys
import torch.nn.functional as F
import torch
import numpy as np
import os, argparse
from datetime import datetime
from tqdm import tqdm
import torch.multiprocessing as mp  # 确保导入多进程模块
from DASFCNet_SMT_MobileViT import DASFCNet
from data import get_loader, test_dataset
from utils import clip_gradient, adjust_lr, opt_save, iou_loss
import torch.backends.cudnn as cudnn
import pytorch_iou

os.environ["CUDA_VISIBLE_DEVICES"] = '1'


# 定义训练函数（保持不变）
def train(train_loader, model, optimizer, epoch, opt, CE, IOU):
    model.train()
    loss_list = []
    total_step = len(train_loader)
    for i, pack in enumerate(train_loader, start=1):
        optimizer.zero_grad()
        (images, gts, depth) = pack
        images = images.cuda()
        gts = gts.cuda()
        depth = depth.cuda().repeat(1, 3, 1, 1)
        #
        pred_1, pred_2, pred_3, pred_4 = model(images, depth)

        loss1 = CE(pred_1, gts) + iou_loss(pred_1, gts)
        loss2 = CE(pred_2, gts) + iou_loss(pred_2, gts)
        loss3 = CE(pred_3, gts) + iou_loss(pred_3, gts)
        loss4 = CE(pred_4, gts) + iou_loss(pred_4, gts)

        loss = loss1 + loss2 + loss3 + loss4

        loss.backward()
        clip_gradient(optimizer, opt.clip)
        optimizer.step()

        loss_list.append(float(loss))

        if i % 100 == 0 or i == total_step:
            msg = '{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Learning Rate: {}, Loss: {:.4f}, Loss1: {:.4f}, Loss2: {:.4f}, Loss3: {:.4f}, Loss4: {:.4f}'.format(
                datetime.now(), epoch, opt.epoch, i, total_step,
                opt.lr * opt.decay_rate ** (epoch // opt.decay_epoch), loss.data, loss1.data,
                loss2.data, loss3.data, loss4.data)
            print(msg)
            logging.info(msg)

    return np.mean(loss_list)  # 返回epoch平均损失


# 定义验证函数（保持不变）
def validate(test_dataset, model, epoch, opt):
    global best_mae, best_epoch
    model.eval().cuda()
    mae_sum = 0
    test_loader = test_dataset(opt.val_rgb, opt.val_gt, opt.val_depth, opt.trainsize)
    with torch.no_grad():
        for i in tqdm(range(test_loader.size), desc="Validating", file=sys.stdout):
            image, gt, depth, name, image_for_post = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()
            depth = depth.repeat(1, 3, 1, 1).cuda()

            res, _, _, _ = model(image, depth)
            # res = model(image, depth)
            res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()

            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_train = np.mean(np.abs(res - gt))
            mae_sum += mae_train
    mae = mae_sum / test_loader.size

    if epoch == 0:
        best_mae = mae
    else:
        if mae < best_mae:
            best_mae = round(mae, 5)
            best_epoch = epoch
            torch.save(model.state_dict(), opt.save_path + 'DASFFNet_mae_best.pth',
                       _use_new_zipfile_serialization=False)
            print('best epoch:{}'.format(epoch))
    msg = 'Epoch: {} MAE: {:.5f} ####  bestMAE: {} bestEpoch: {}'.format(epoch, mae, best_mae, best_epoch)
    print(msg)
    logging.info(msg)
    with open(f"{opt.save_path}mae.log", "a", encoding='utf-8') as f:
        f.write('Epoch: {:03d} MAE: {:.5f} ####  bestMAE: {:.5f} bestEpoch: {:03d}\n'.format(epoch, mae, best_mae,
                                                                                             best_epoch))
    return mae


# 主函数：封装所有主逻辑
def main():
    # 解析参数（移到main函数内）
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=200, help='epoch number')
    parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
    parser.add_argument('--batchsize', type=int, default=16, help='training batch size')
    parser.add_argument('--trainsize', type=int, default=384, help='training image size')
    parser.add_argument('--continue_train', type=bool, default=False, help='continue training')
    parser.add_argument('--continue_train_path', type=str, default='', help='continue training path')
    parser.add_argument('--gpu_id', type=str, default='1', help='train use gpu')
    parser.add_argument('--rgb_root', type=str,
                        default='./CodDataset/TrainDataset/Imgs/',
                        help='the training rgb images root')

    parser.add_argument('--depth_root', type=str,
                        default='./CodDataset/TrainDataset/depth/',
                        help='the training depth images root')

    parser.add_argument('--gt_root', type=str, default='./CodDataset/TrainDataset/GT/',
                        help='the training gt images root')
    parser.add_argument('--val_rgb', type=str,
                        default="./CodDataset/TestDataset/CAMO/Imgs/",
                        help='validate rgb path')

    parser.add_argument('--val_depth', type=str,
                        default="./CodDataset/TestDataset/CAMO/depth/",
                        help='validate depth path')

    parser.add_argument('--val_gt', type=str,
                        default="./CodDataset/TestDataset/CAMO/GT/",
                        help='validate gt path')

    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int, default=100, help='every n epochs decay learning rate')
    parser.add_argument('--save_path', type=str, default="ckps/DASFF-Net/", help='checkpoint path')
    opt = parser.parse_args()

    opt_save(opt)

    # 设置GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
    print(f'USE GPU {opt.gpu_id}')

    cudnn.benchmark = True

    # 初始化日志
    logging.basicConfig(
        filename=opt.save_path + 'log.log',
        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
        level=logging.INFO,
        filemode='a',
        datefmt='%Y-%m-%d %H:%M:%S %p'
    )
    logging.info("Net-Train")

    # 初始化模型
    model = DASFCNet()
    if os.path.exists("ckps/smt/smt_tiny.pth"):
        model.rgb_backbone.load_state_dict(
            torch.load("./ckps/smt_tiny.pth")['model'])

        model.d_backbone.load_state_dict(torch.load("./ckps/mobilevit_s.pt"
                                                    ), strict=False)
        print(f"loaded imagenet pretrained smt_tiny.pth from ckps")
        print(f"loaded imagenet pretrained mobilevit_s.pth from ckps")
    else:
        raise Exception("please put smt_tiny.pth under ckps/smt/folder")

    model.cuda()

    # 优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), opt.lr)
    CE = torch.nn.BCEWithLogitsLoss()
    IOU = pytorch_iou.IOU(size_average=True)

    # 加载数据（移到main函数内，避免子进程重复加载）
    print('load data...')
    train_loader = get_loader(
        opt.rgb_root,
        opt.gt_root,
        opt.depth_root,
        batchsize=opt.batchsize,
        trainsize=opt.trainsize
    )
    total_step = len(train_loader)

    # 训练相关变量初始化
    best_loss = 1.0
    global best_mae, best_epoch
    best_mae = 1.0
    best_epoch = 0

    # 训练循环
    print("Let's go!")
    for epoch in range(opt.epoch):
        adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        epoch_loss = train(train_loader, model, optimizer, epoch, opt, CE, IOU)

        # 保存最佳损失模型
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(
                model.state_dict(),
                opt.save_path + f'Net_rgb_d.pth.{epoch}_{epoch_loss:.3f}',
                _use_new_zipfile_serialization=False
            )

        # 如果周期数是5的倍数
        if epoch >= 50:
            # 保存模型的状态字典到指定路径
            torch.save(
                model.state_dict(),
                opt.save_path + f'Net_rgb_d.pth.{epoch}_{epoch_loss:.3f}',
                _use_new_zipfile_serialization=False
            )
        # 验证
        validate(test_dataset, model, epoch, opt)


# 保护主模块，避免多进程递归创建
if __name__ == '__main__':
    mp.freeze_support()  # Windows多进程支持（必须在main()前调用）
    main()  # 执行主函数

