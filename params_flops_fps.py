import os
from thop import profile
import numpy as np
import torch


def fps(model, epoch_num, size, gpu=0, count=2):
    # dummy_input = torch.randn(1, 3, size, size).cuda()
    # # flops, params = profile(model, (dummy_input, ))
    # flops, params = profile(model, (dummy_input, dummy_input))
    # print('GFLOPs: %.2f , params: %.2f M' % (flops / 1.0e9, params / 1.0e6))

    # torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    assert count in [1, 2]

    ls = []  # 每次计算得到的fps
    iterations = 300  # 重复计算的轮次

    model.eval().to(f'cuda:{gpu}' if gpu >= 0 else 'cpu')
    random_input = torch.randn(1, 3, size, size).to(f'cuda:{gpu}' if gpu >= 0 else 'cpu')
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    # GPU预热
    for _ in range(50):
        if count == 2:
            _ = model(random_input, random_input)
        else:
            _ = model(random_input)

    for i in range(epoch_num + 1):
        # 测速
        times = torch.zeros(iterations)  # 存储每轮iteration的时间
        with torch.no_grad():
            for iter in range(iterations):

                if count == 2:
                    starter.record()
                    _ = model(random_input, random_input)
                    ender.record()

                else:
                    starter.record()
                    _ = model(random_input)
                    ender.record()

                # 同步GPU时间
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)  # 计算时间
                times[iter] = curr_time
                # print(curr_time)

        mean_time = times.mean().item()

        if i == 0:
            print("Initialization Inference time: {:.2f} ms, FPS: {:.2f} ".format(mean_time, 1000 / mean_time))
        if i != 0:
            ls.append(1000 / mean_time)
            print("{}/{} Inference time: {:.2f} ms, FPS: {:.2f} ".format(i, epoch_num, mean_time, 1000 / mean_time))
    print(f"平均fps为 {np.mean(ls):.2f}")
    print(f"最大fps为 {np.max(ls):.2f}")


def flops(model, size, gpu=0, count=2):
    assert count in [1, 2], 'please input correct param number !'
    model.cuda().eval()
    os.environ['CUDA_VISIBLE_DEVICES'] = f'{gpu}'
    dummy_input = torch.randn(1, 3, size, size).cuda()
    if count == 2:
        flops, params = profile(model, (dummy_input, dummy_input))
    else:
        flops, params = profile(model, (dummy_input,))
    print('GFLOPs: %.3f , params: %.2f M' % (flops / 1.0e9, params / 1.0e6))


def param(model):
    param_sum = 0
    for i in model.named_modules():
        if '.' not in i[0] and i[0] != '':
            layer = getattr(model, i[0])
            temp = sum(p.numel() for p in layer.parameters() if p.requires_grad) / 1e6
            param_sum += temp
            print(i[0], temp, "M")
    print(param_sum, 'M')


if __name__ == '__main__':
    # from utils import param, flops, fps
    # from models.PRNet import Net

    from model.DASFCNet_SMT_MobileViT import DASFCNet

    TRAIN_SIZE = 384

    model = DASFCNet().cuda().eval()
    x = torch.randn(1, 3, TRAIN_SIZE, TRAIN_SIZE).cuda()
    d = torch.randn(1, 3, TRAIN_SIZE, TRAIN_SIZE).cuda()

    f_ls = model(x, d)
    # for i in f_ls:
    #     print(i.shape)

    # param(model)
    # flops(model, TRAIN_SIZE, count=2)

    fps(model=model, epoch_num=10, size=TRAIN_SIZE, gpu=1, count=2)
