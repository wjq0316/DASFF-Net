# import math
#
# from thop import profile
#
# from model.smt import smt_t  # 导入自定义的SMT模块
# from model.MobileNetV2 import mobilenet_v2  # 导入MobileNetV2模型
# from model.mobilevit import mobile_vit_small
# import torch.nn as nn  # 导入PyTorch的神经网络模块
# import torch  # 导入PyTorch库
# import torch.nn.functional as F  # 导入PyTorch的函数模块
# from timm.models.layers import trunc_normal_  # 导入timm库中的trunc_normal_函数
#
# TRAIN_SIZE = 384  # 定义训练图像的大小为384x384
#
# import torch.nn.functional as F
#
#
# # 定义一个名为BasicConv2d的类，它继承自nn.Module
# class BasicConv2d(nn.Module):
#     # 定义一个构造函数__init__，它接受六个参数：输入通道数in_planes、输出通道数out_planes、卷积核大小kernel_size、步长stride（默认为1）、填充padding（默认为0）和膨胀dilation（默认为1）
#     def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
#         # 调用父类nn.Module的构造函数
#         super(BasicConv2d, self).__init__()
#         # 初始化一个卷积层，使用指定的参数。nn.Conv2d是PyTorch中的一个函数，用于创建二维卷积层
#         self.conv = nn.Conv2d(in_planes, out_planes,
#                               kernel_size=kernel_size, stride=stride,
#                               padding=padding, dilation=dilation, bias=False)
#         # 初始化一个批量归一化层，使用指定的输出通道数
#         self.bn = nn.BatchNorm2d(out_planes)
#         # 初始化一个ReLU激活函数，使用in-place模式
#         self.gelu = nn.GELU()
#
#     # 定义一个前向传播方法forward，它接受一个输入x
#     def forward(self, x):
#         # 应用卷积层处理输入x
#         x = self.conv(x)
#         # 应用批量归一化层处理卷积层的输出
#         x = self.bn(x)
#
#         x = self.gelu(x)
#
#         # 返回处理后的特征x
#         return x
#
#
# class SpatialAttention(nn.Module):
#     # 定义一个构造函数__init__，它接受一个参数kernel_size，表示卷积核的大小，默认为7
#     def __init__(self, kernel_size=7):
#         # 调用父类nn.Module的构造函数
#         super(SpatialAttention, self).__init__()
#         # 检查kernel_size是否为3或7，如果不是，则抛出异常
#         assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
#         # 计算padding值，用于卷积层的padding参数
#         padding = 3 if kernel_size == 7 else 1
#
#         # 初始化卷积层，用于计算空间注意力权重
#         self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
#         # 初始化Sigmoid函数，用于生成最终的注意力权重
#         self.sigmoid = nn.Sigmoid()
#
#     # 定义一个前向传播方法forward，它接受一个输入x
#     def forward(self, x):
#         # 计算平均池化后的特征
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         # 计算最大池化后的特征
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         # 将平均池化后的特征和最大池化后的特征拼接在一起
#         x = torch.cat([avg_out, max_out], dim=1)
#         # 应用卷积层处理拼接后的特征
#         x = self.conv1(x)
#         # 应用Sigmoid函数生成最终的注意力权重
#         return self.sigmoid(x)
#
#
# class CAFM(nn.Module):
#     def __init__(self, in_channel, out_channel):
#         super(CAFM, self).__init__()
#
#         self.branch0 = nn.Sequential(
#             BasicConv2d(in_channel, in_channel // 4, kernel_size=3, padding=1),
#             BasicConv2d(in_channel // 4, in_channel // 4, kernel_size=3, padding=1, dilation=1)
#         )
#
#         self.branch1 = nn.Sequential(
#             BasicConv2d(in_channel, in_channel // 4, kernel_size=3, padding=1),
#             BasicConv2d(in_channel // 4, in_channel // 4, kernel_size=3, padding=3, dilation=3)
#         )
#
#         self.branch2 = nn.Sequential(
#             BasicConv2d(in_channel, in_channel // 4, kernel_size=3, padding=1),
#             BasicConv2d(in_channel // 4, in_channel // 4, kernel_size=3, padding=5, dilation=5)
#         )
#
#         self.branch3 = nn.Sequential(
#             BasicConv2d(in_channel, in_channel // 4, kernel_size=3, padding=1),
#             BasicConv2d(in_channel // 4, in_channel // 4, kernel_size=3, padding=7, dilation=7)
#         )
#
#         # 改进的翻转注意力模块
#         self.reverse_attention = nn.Sequential(
#             nn.Conv2d(in_channel, in_channel // 4, kernel_size=1),
#             nn.ReLU(),
#             nn.Conv2d(in_channel // 4, in_channel, kernel_size=1),
#             nn.Sigmoid()
#         )
#
#         # 注意力融合门控机制
#         self.attention_gate = nn.Sequential(
#             nn.Conv2d(in_channel * 2, in_channel // 2, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(in_channel // 2, in_channel, kernel_size=1),
#             nn.Sigmoid()
#         )
#
#         self.conv_cat = BasicConv2d(in_channel, in_channel, kernel_size=3, padding=1)
#
#         self.sa = SpatialAttention()
#
#         self.fusion = BasicConv2d(in_channel, in_channel, kernel_size=3, padding=1)
#         self.conv_down = BasicConv2d(in_channel, out_channel, kernel_size=3, padding=1)
#
#     def forward(self, x_pre):
#         # 多尺度处理
#         x0 = self.branch0(x_pre)
#         x1 = self.branch1(x_pre)
#         x2 = self.branch2(x_pre)
#         x3 = self.branch3(x_pre)
#
#         x_fused = self.conv_cat(torch.cat((x0, x1, x2, x3), dim=1))
#
#         # 空间注意力
#         sa_map = self.sa(x_fused)  # 获取注意力图
#         x_sa = sa_map * x_fused
#
#         reverse_att = self.reverse_attention(x_fused)
#
#         # 2. 应用翻转注意力（聚焦被忽略区域）
#         x_reverse = reverse_att * (1 - sa_map) * x_fused
#
#         # 3. 门控融合机制
#         gate = self.attention_gate(torch.cat([x_sa, x_reverse], dim=1))
#         x_dual_att = gate * x_sa + (1 - gate) * x_reverse
#
#         # 残差连接
#         x_ff = x_pre + x_dual_att
#
#         # 输出处理
#         x_out = self.fusion(x_ff)
#         out = self.conv_down(x_out)
#
#         return out
#
#
# class EnhancedCA(nn.Module):
#     def __init__(self, in_planes, ratio=16):
#         super().__init__()
#         self.shared_conv = nn.Sequential(
#             nn.Conv2d(2 * in_planes, 2 * in_planes // ratio, 1, bias=False),
#             nn.ReLU(),
#             nn.Conv2d(2 * in_planes // ratio, in_planes, 1, bias=False)
#         )
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, rgb, depth):
#         # 双模态特征拼接
#         x = torch.cat([rgb, depth], dim=1)
#
#         # 双路径池化
#         max_pool = self.shared_conv(nn.AdaptiveMaxPool2d(1)(x))
#         avg_pool = self.shared_conv(nn.AdaptiveAvgPool2d(1)(x))
#
#         # 注意力权重生成
#         weight = self.sigmoid(max_pool + avg_pool)
#         return depth * weight
#
#
# class EnhancedSA(nn.Module):
#     def __init__(self, in_planes, kernel_size=7):
#         super().__init__()
#         padding = kernel_size // 2
#         self.conv = nn.Sequential(
#             nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False),
#             nn.BatchNorm2d(1),
#             nn.Sigmoid()
#         )
#
#     def forward(self, rgb, depth):
#         # 跨模态空间特征提取
#         rgb_max, _ = torch.max(rgb, dim=1, keepdim=True)
#         depth_max, _ = torch.max(depth, dim=1, keepdim=True)
#
#         # 空间特征融合
#         x = torch.cat([rgb_max, depth_max], dim=1)
#         return self.conv(x)
#
#
# class DAMv2(nn.Module):
#     def __init__(self, in_planes, ratio=16):
#         super().__init__()
#         self.ca = EnhancedCA(in_planes, ratio)
#         self.sa = EnhancedSA(in_planes)
#         self.cross_att = nn.Sequential(
#             nn.Conv2d(in_planes * 2, in_planes, 3, padding=1, bias=False),
#             nn.ReLU()
#         )
#
#     def forward(self, rgb, depth):
#         # 并行双注意力
#         ca_out = self.ca(rgb, depth)
#         sa_mask = self.sa(rgb, depth)
#
#         # 注意力融合
#         sa_out = depth * sa_mask
#
#         # 交叉模态融合
#         fused = torch.cat([ca_out, sa_out], dim=1)
#         return self.cross_att(fused) + depth
#
#
# # 坐标注意力模块 (适合方向性边缘)
# class h_sigmoid(nn.Module):
#     def __init__(self, inplace=True):
#         # 调用基类的构造函数
#         super(h_sigmoid, self).__init__()
#         # ReLU6激活函数，其输出范围被限制在0到6之间
#         self.relu = nn.ReLU6(inplace=inplace)
#
#     # 定义前向传播函数，实现h_sigmoid激活函数
#     def forward(self, x):
#         # 通过将输入x加上3，然后通过ReLU6激活函数，再除以6来模拟sigmoid函数
#         return self.relu(x + 3) / 6
#
#
# # 定义h_swish类，这是一个自定义的激活函数，模仿swish函数的形状
# class h_swish(nn.Module):
#     def __init__(self, inplace=True):
#         # 调用基类的构造函数
#         super(h_swish, self).__init__()
#         # 使用h_sigmoid作为sigmoid激活函数的实现
#         self.sigmoid = h_sigmoid(inplace=inplace)
#
#     # 定义前向传播函数，实现h_swish激活函数
#     def forward(self, x):
#         # 通过将输入x与通过h_sigmoid的x相乘来实现swish操作
#         return x * self.sigmoid(x)
#
#
# # 这段代码定义了一个名为 CoordAtt 的类，它是一个用于图像处理的注意力模块，继承自 nn.Module  LFE
# class CoordAtt(nn.Module):
#     # 定义CoordAtt类，这是一个坐标注意力模块
#     def __init__(self, inp, oup, reduction=32):
#         # 调用基类的构造函数
#         super(CoordAtt, self).__init__()
#         # 定义两个自适应平均池化层，分别对高度和宽度进行池化
#         self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
#         self.pool_w = nn.AdaptiveAvgPool2d((1, None))
#
#         # 计算多输入通道（mip），至少为8，最大为inp除以reduction
#         mip = max(8, inp // reduction)
#
#         # 定义一个1x1卷积层，后接批量归一化层和h_swish激活函数
#         self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
#         self.bn1 = nn.BatchNorm2d(mip)
#         self.act = h_swish()
#
#         # 定义两个1x1卷积层，用于生成高度和宽度方向的注意力图
#         self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
#         self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
#
#     def forward(self, x):
#         # 保存输入x作为残差连接
#         identity = x
#
#         # 获取输入x的尺寸
#         n, c, h, w = x.size()
#         # 对x的高度进行池化，得到x_h
#         x_h = self.pool_h(x)
#         # 对x的宽度进行池化，然后交换H和W的维度，得到x_w
#         x_w = self.pool_w(x).permute(0, 1, 3, 2)
#
#         # 沿着宽度拼接x_h和x_w，得到y
#         y = torch.cat([x_h, x_w], dim=2)
#         # 通过第一个1x1卷积层
#         y = self.conv1(y)
#         # 进行批量归一化
#         y = self.bn1(y)
#         # 通过激活函数
#         y = self.act(y)
#
#         # 将y按照原始的高度和宽度拆分成x_h和x_w
#         x_h, x_w = torch.split(y, [h, w], dim=2)
#         # 交换x_w的高度和宽度维度，以匹配原始输入的维度
#         x_w = x_w.permute(0, 1, 3, 2)
#
#         # 通过conv_h和conv_w生成高度和宽度的注意力图，并应用sigmoid函数
#         a_h = self.conv_h(x_h).sigmoid()
#         a_w = self.conv_w(x_w).sigmoid()
#
#         # 将输入x与两个方向的注意力图相乘，实现注意力机制
#         out = identity * a_w * a_h
#
#         # 返回注意力加权的输出
#         return out
#
#
# class ChannelAttention(nn.Module):
#     def __init__(self, in_planes, reduction=1):
#         super(ChannelAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#
#         self.fc = nn.Sequential(
#             nn.Conv2d(in_planes, in_planes // reduction, 1, bias=False),
#             nn.ReLU(),
#             nn.Conv2d(in_planes, in_planes, 1, bias=False)
#         )
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         avg_out = self.fc(self.avg_pool(x))
#         max_out = self.fc(self.max_pool(x))
#         out = avg_out + max_out
#         return self.sigmoid(out)
#
#
# class FPEM(nn.Module):
#     def __init__(self, in_channels, reduction_ratio=2):
#         super(FPEM, self).__init__()
#
#         self.in_channels = in_channels
#
#         self.reduced_channels = max(in_channels // reduction_ratio, 8)  # 通道压缩至合理范围
#
#         # 小波正变换
#         self.dwt = DWT()
#
#         # 小波逆变换
#         self.iwt = IWT()
#
#         # 通道压缩：统一小波变换前的特征维度
#         self.channel_reducer = nn.Sequential(
#             nn.Conv2d(in_channels, self.reduced_channels, 1),
#             nn.BatchNorm2d(self.reduced_channels),
#             nn.ReLU(inplace=True)
#         )
#
#         # 低频处理模块
#         self.ll_processor = nn.Sequential(
#             nn.Conv2d(self.reduced_channels, self.reduced_channels, 3, padding=1),
#             nn.BatchNorm2d(self.reduced_channels),
#             nn.GELU()
#         )
#
#         self.ca = ChannelAttention(self.reduced_channels)
#         self.sa = SpatialAttention()
#
#         self.lh_processor = nn.Sequential(
#             nn.Conv2d(self.reduced_channels, self.reduced_channels, (1, 3), padding=(0, 1)),  # 小核3x1
#             nn.BatchNorm2d(self.reduced_channels),
#             nn.GELU(),
#         )
#
#         # 垂直高频(hl)：垂直方向小核卷积
#         self.hl_processor = nn.Sequential(
#             nn.Conv2d(self.reduced_channels, self.reduced_channels, (3, 1), padding=(1, 0)),  # 小核1x3
#             nn.BatchNorm2d(self.reduced_channels),
#             nn.GELU(),
#         )
#
#         # 对角高频(hh)：简单3x3卷积
#         self.hh_processor = nn.Sequential(
#             nn.Conv2d(self.reduced_channels, self.reduced_channels, 3, padding=1),
#             nn.BatchNorm2d(self.reduced_channels),
#             nn.GELU(),
#         )
#
#         # 重构后处理
#         # self.cat_processor = nn.Sequential(
#         #     # 坐标注意力增强
#         #     CoordAtt(3 * self.reduced_channels, 3 * self.reduced_channels),
#         #     # 特征融合
#         #     nn.Conv2d(3 * self.reduced_channels, 3 * self.reduced_channels, 1),
#         #     nn.BatchNorm2d(3 * self.reduced_channels),
#         #     nn.GELU()
#         # )
#
#         self.pre_att = nn.Sequential(
#             nn.Conv2d(in_channels * 2, in_channels * 2, kernel_size=3, padding=1, groups=in_channels * 2),
#             nn.BatchNorm2d(in_channels * 2),
#             nn.GELU(),
#             nn.Conv2d(in_channels * 2, in_channels, kernel_size=1),
#             nn.BatchNorm2d(in_channels),
#             nn.GELU()
#         )
#
#         # 特征融合 (恢复原始通道)
#         self.fusion = nn.Sequential(
#             CoordAtt(self.reduced_channels, self.reduced_channels),
#             nn.Conv2d(self.reduced_channels, self.reduced_channels, 3, padding=1),
#             nn.BatchNorm2d(self.reduced_channels),
#             nn.GELU(),
#             nn.Conv2d(self.reduced_channels, in_channels, 1),
#             nn.BatchNorm2d(in_channels),
#             nn.GELU()
#         )
#
#         # 残差连接
#         self.residual = nn.Identity()
#
#     def forward(self, x, d=None):
#         if d is not None:
#             fused = torch.cat([x, d], dim=1)
#             x = self.pre_att(fused)
#
#         identity = self.residual(x)
#
#         orignal_size = x.size()[2:]
#         B, C, H, W = x.size()
#
#         if H % 2 != 0 or W % 2 != 0:
#             H_new = H if H % 2 == 0 else H - 1
#             W_new = W if W % 2 == 0 else W - 1
#             x_resized = F.interpolate(x, size=(H_new, W_new), mode="bilinear", align_corners=False)
#
#         else:
#             x_resized = x
#
#         x = self.channel_reducer(x_resized)
#
#         # 通道压缩
#         # 小波分解
#         ll, lh, hl, hh = self.dwt(x)
#         # 拆分四个子带
#         # 1. 低频处理
#         ll_processed = self.ll_processor(ll)
#         ll_ca = ll_processed * self.ca(ll_processed)
#         ll_sa = ll_ca * self.sa(ll_ca)  # [1,256,6,6]
#
#         ll_mean = torch.mean(ll_sa, dim=(2, 3), keepdim=True)  # [1,256,1,1] # 低频全局指导信号
#
#         lh_processed = self.lh_processor(lh) * ll_mean  # [1,256,6,6]
#
#         hl_processed = self.hl_processor(hl) * ll_mean
#
#         hh_processed = self.hh_processor(hh) * ll_mean
#
#         iwt_input = torch.cat([
#             ll_sa,  # 处理后的低频 [B, C', H/2, W/2]
#             lh_processed,  # 处理后的水平高频 [B, C', H/2, W/2]
#             hl_processed,  # 处理后的垂直高频 [B, C', H/2, W/2]
#             hh_processed  # 处理后的对角高频 [B, C', H/2, W/2]
#         ], dim=0)  # 关键：在批次维度 (dim=0) 拼接 -> [4*B, C', H/2, W/2]
#
#         # 2. 高频合并处理
#         # hf_cat = torch.cat([lh_processed, hl_processed, hh_processed], dim=1)
#         #
#         # hf_processed = self.cat_processor(hf_cat)  # [1,768,6,6]
#         # 拆分处理后的高频
#         # lh_p, hl_p, hh_p = torch.chunk(hf_processed, 3, dim=1)
#         # 3. 小波重构
#         recon = self.iwt(iwt_input)
#
#         if H % 2 != 0 or W % 2 != 0:
#             recon = F.interpolate(recon, size=orignal_size, mode="bilinear", align_corners=False)
#
#         # 特征融合
#         fused1 = self.fusion(recon)
#
#         return fused1 + identity
#
#
# # class DWT(nn.Module):
# #     def __init__(self):
# #         super(DWT, self).__init__()
# #         self.requires_grad = False
# #
# #     def forward(self, x):
# #         x01 = x[:, :, 0::2, :] / 2
# #         x02 = x[:, :, 1::2, :] / 2
# #         x1 = x01[:, :, :, 0::2]
# #         x2 = x02[:, :, :, 0::2]
# #         x3 = x01[:, :, :, 1::2]
# #         x4 = x02[:, :, :, 1::2]
# #         ll = x1 + x2 + x3 + x4
# #         lh = -x1 + x2 - x3 + x4
# #         hl = -x1 - x2 + x3 + x4
# #         hh = x1 - x2 - x3 + x4
# #         return ll, lh, hl, hh
# #
# #
# # class IWT(nn.Module):
# #     def __init__(self):
# #         super(IWT, self).__init__()
# #         self.requires_grad = False
# #
# #     def forward(self, ll, lh, hl, hh):
# #         device = ll.device
# #         batch, channel, height, width = ll.shape
# #         recon = torch.zeros(batch, channel, height * 2, width * 2).float().to(device)
# #         recon[:, :, 0::2, 0::2] = ll - lh - hl + hh
# #         recon[:, :, 1::2, 0::2] = ll - lh + hl - hh
# #         recon[:, :, 0::2, 1::2] = ll + lh - hl - hh
# #         recon[:, :, 1::2, 1::2] = ll + lh + hl + hh
# #         return recon
#
#
# #  缩小hw和扩大b到4b
# def dwt_init(x):
#     # 第一步：在高度方向上进行下采样（隔行采样）并归一化
#     x01 = x[:, :, 0::2, :] / 2  # 选择偶数行并除以2（归一化）
#     x02 = x[:, :, 1::2, :] / 2  # 选择奇数行并除以2（归一化）
#
#     # 第二步：在宽度方向上进行下采样（隔列采样）
#     x1 = x01[:, :, :, 0::2]  # 从偶数行中选择偶数列（左上象限）
#     x2 = x02[:, :, :, 0::2]  # 从奇数行中选择偶数列（左下象限）
#     x3 = x01[:, :, :, 1::2]  # 从偶数行中选择奇数列（右上象限）
#     x4 = x02[:, :, :, 1::2]  # 从奇数行中选择奇数列（右下象限）
#
#     # 第三步：通过加减组合生成四个子带
#     x_LL = x1 + x2 + x3 + x4  # 低频近似：四个象限相加（保留主要信息）
#     x_HL = -x1 - x2 + x3 + x4  # 水平细节：垂直边缘信息（奇数行贡献为负）
#     x_LH = -x1 + x2 - x3 + x4  # 垂直细节：水平边缘信息（奇数列贡献为负）
#     x_HH = x1 - x2 - x3 + x4  # 对角细节：对角边缘和纹理信息
#
#     # 第四步：将四个子带在批次维度上拼接
#     return x_LL, x_LH, x_HL, x_HH
#     # 第四步：将四个子带在批次维度上拼接
#
#
# # 使用哈尔 haar 小波变换来实现二维离散小波
# # 还原 b和hw
# def iwt_init(x):
#     # 设置缩放因子，r=2表示将图像尺寸扩大2倍
#     r = 2
#     # 获取输入张量的维度信息：批次大小、通道数、高度和宽度
#     in_batch, in_channel, in_height, in_width = x.size()
#     # 计算输出张量的维度：批次大小缩小r²倍，通道数不变，高度和宽度扩大r倍
#     out_batch, out_channel, out_height, out_width = int(in_batch / (r ** 2)), in_channel, r * in_height, r * in_width
#
#     # 将输入张量分割为四个子带，并分别除以2（归一化处理）
#     # 假设输入是按批次排列的四个子带：LL, LH, HL, HH
#     x1 = x[0:out_batch, :, :, :] / 2  # 低频子带LL
#     x2 = x[out_batch:out_batch * 2, :, :, :] / 2  # 水平高频子带LH
#     x3 = x[out_batch * 2:out_batch * 3, :, :, :] / 2  # 垂直高频子带HL
#     x4 = x[out_batch * 3:out_batch * 4, :, :, :] / 2  # 对角高频子带HH
#
#     # 创建全零输出张量，并将其放置在与输入相同的设备上（CPU或GPU）
#     h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().to(x.device)
#
#     # 根据逆小波变换的重构规则，将四个子带重新组合到输出张量的不同位置
#     # 偶数行偶数列位置：低频信息减去水平、垂直和对角高频信息
#     h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
#     # 奇数行偶数列位置：低频信息减去水平高频，加上垂直高频，减去对角高频
#     h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
#     # 偶数行奇数列位置：低频信息加上水平高频，减去垂直高频，减去对角高频
#     h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
#     # 奇数行奇数列位置：低频信息加上所有高频信息
#     h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4
#
#     return h  # 返回重构后的图像
#
#
# class DWT(nn.Module):
#     def __init__(self):
#         super(DWT, self).__init__()
#         self.requires_grad = False  # 信号处理，非卷积运算，不需要进行梯度求导
#
#     def forward(self, x):
#         return dwt_init(x)
#
#
# class IWT(nn.Module):
#     def __init__(self):
#         super(IWT, self).__init__()
#         self.requires_grad = False
#
#     def forward(self, x):
#         return iwt_init(x)
#
#
# class Decoder(nn.Module):
#     def __init__(self, channel=32):
#         super(Decoder, self).__init__()
#         self.predict_layer = nn.Sequential(
#             nn.Conv2d(channel, 1, kernel_size=1, padding=0),
#         )
#
#     def forward(self, x):
#         prediction = self.predict_layer(x)
#
#         return prediction
#
#
# class Trans(nn.Module):
#     def __init__(self, inc, outc):
#         super().__init__()
#         self.trans = nn.Sequential(
#             nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=1),  # 1x1卷积层，用于调整通道数
#             nn.BatchNorm2d(outc),  # 批归一化层，用于稳定训练
#             nn.GELU()  # GELU激活函数，用于非线性变换
#         )
#
#     def forward(self, d):
#         return self.trans(d)  # 前向传播，通过定义的变换模块
#
#
# class DASFCNet(nn.Module):
#     def __init__(self, pretrained=True):
#         super().__init__()
#         self.rgb_backbone = smt_t(pretrained)  # RGB主干网络
#         # self.rgb_backbone = mobile_vit_small()  # RGB主干网络
#         # self.d_backbone = mobilenet_v2(pretrained)  # 深度主干网络
#         self.d_backbone = mobile_vit_small()
#
#         # self.depth_enhance4 = DAM_module(512*2)
#         # self.depth_enhance3 = DAM_module(256*2)
#         # self.depth_enhance2 = DAM_module(128*2)
#         # self.depth_enhance1 = DAM_module(64*2)
#
#         # self.depth_enhance4 = DAMv2(512)
#         # self.depth_enhance3 = DAMv2(256)
#         # self.depth_enhance2 = DAMv2(128)
#         # self.depth_enhance1 = DAMv2(64)
#
#         self.d_trans_4 = Trans(160, 512)  # 深度特征变换层4
#         self.d_trans_3 = Trans(128, 256)  # 深度特征变换层3
#         self.d_trans_2 = Trans(96, 128)  # 深度特征变换层2
#         self.d_trans_1 = Trans(64, 64)  # 深度特征变换层1
#
#         # Fuse
#         self.Trans1 = BasicConv2d(256, 256, kernel_size=3, padding=1)
#         self.Trans2 = BasicConv2d(128, 128, kernel_size=3, padding=1)
#         self.Trans3 = BasicConv2d(64, 64, kernel_size=3, padding=1)
#
#         self.decoder = nn.ModuleList([
#             Decoder(channel=32),
#             Decoder(channel=64),
#             Decoder(channel=128),
#             Decoder(channel=256),
#         ])
#
#         # 创建一个包含RFB_modified模块的列表，用于特征的增强处理
#         self.cafm = nn.ModuleList([
#             CAFM(64, 32),
#             CAFM(128, 64),
#             CAFM(256, 128),
#             CAFM(512, 256)
#         ])
#
#         self.fpem = nn.ModuleList([
#             FPEM(64),
#             FPEM(128),
#             FPEM(256),
#             FPEM(512)
#         ])
#
#         self.deconv_layer_44 = nn.Sequential(
#             nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(512),
#             nn.GELU(),
#         )
#
#         #
#         self.deconv_layer_43 = nn.Sequential(
#             nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(256),
#             nn.GELU(),
#         )
#
#         self.deconv_layer_33 = nn.Sequential(
#             nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(256),
#             nn.GELU(),
#         )
#
#         self.deconv_layer_32 = nn.Sequential(
#             nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(128),
#             nn.GELU(),
#         )
#
#         self.deconv_layer_22 = nn.Sequential(
#             nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(128),
#             nn.GELU(),
#         )
#
#         self.deconv_layer_21 = nn.Sequential(
#             nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(64),
#             nn.GELU(),
#         )
#
#         self.deconv_layer_11 = nn.Sequential(
#             nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(64),
#             nn.GELU(),
#         )
#
#     def forward(self, x_rgb, x_d, shape=None):
#         shape = x_rgb.size()[2:] if shape is None else shape
#         # rgb [1,64,96,96] [1,96,48,48] [1,128,24,24] [1,160,12,12]
#         _, (rgb_1, rgb_2, rgb_3, rgb_4) = self.rgb_backbone(x_rgb)  # 提取RGB特征
#
#         # d [1,64,96,96] [1,96,48,48] [1,128,24,24] [1,160,12,12]
#         _, d_1, d_2, d_3, d_4 = self.d_backbone(x_d)  # 提取深度特征
#
#         d_4 = self.d_trans_4(d_4)  # 变换深度特征4
#         d_3 = self.d_trans_3(d_3)  # 变换深度特征3
#         d_2 = self.d_trans_2(d_2)  # 变换深度特征2
#         d_1 = self.d_trans_1(d_1)  # 变换深度特征1
#
#         # d_4 = self.depth_enhance4(rgb_4, d_4)
#         # d_3 = self.depth_enhance3(rgb_3, d_3)
#         # d_2 = self.depth_enhance2(rgb_2, d_2)
#         # d_1 = self.depth_enhance1(rgb_1, d_1)
#
#         rgb_43_up = F.interpolate(self.deconv_layer_43(rgb_4), size=rgb_3.size()[2:], mode='bilinear',
#                                   align_corners=False)
#         d_43_up = F.interpolate(self.deconv_layer_43(d_4), size=d_3.size()[2:], mode='bilinear', align_corners=False)
#
#         rgb_32_up = F.interpolate(self.deconv_layer_32(rgb_3), size=rgb_2.size()[2:], mode='bilinear',
#                                   align_corners=False)
#         d_32_up = F.interpolate(self.deconv_layer_32(d_3), size=d_2.size()[2:], mode='bilinear', align_corners=False)
#
#         rgb_21_up = F.interpolate(self.deconv_layer_21(rgb_2), size=rgb_1.size()[2:], mode='bilinear',
#                                   align_corners=False)
#         d_21_up = F.interpolate(self.deconv_layer_21(d_2), size=d_1.size()[2:], mode='bilinear', align_corners=False)
#
#         rgb4_con = torch.cat((rgb_4, rgb_4), dim=1)
#         rgb4_con = self.deconv_layer_44(rgb4_con)
#         d4_con = torch.cat((d_4, d_4), dim=1)
#         d4_con = self.deconv_layer_44(d4_con)
#
#         rgb3_con = torch.cat((rgb_3, rgb_43_up), dim=1)
#         rgb3_con = self.deconv_layer_33(rgb3_con)
#         d3_con = torch.cat((d_3, d_43_up), dim=1)
#         d3_con = self.deconv_layer_33(d3_con)
#
#         rgb2_con = torch.cat((rgb_2, rgb_32_up), dim=1)
#         rgb2_con = self.deconv_layer_22(rgb2_con)
#         d2_con = torch.cat((d_2, d_32_up), dim=1)
#         d2_con = self.deconv_layer_22(d2_con)
#
#         rgb1_con = torch.cat((rgb_1, rgb_21_up), dim=1)
#         rgb1_con = self.deconv_layer_11(rgb1_con)
#         d1_con = torch.cat((d_1, d_21_up), dim=1)
#         d1_con = self.deconv_layer_11(d1_con)
#
#         bk_edge4 = self.fpem[3](rgb4_con, d4_con)  # [1,160,12,12]
#         bk_edge3 = self.fpem[2](rgb3_con, d3_con)  # [1,128,24,24]
#         bk_edge2 = self.fpem[1](rgb2_con, d2_con)  # [1,96,48,48]
#         bk_edge1 = self.fpem[0](rgb1_con, d1_con)  # [1,64,96,96]
#
#         ################################################ CAEM ####################################
#
#         out4 = self.cafm[3](bk_edge4)
#
#         prediction4 = F.interpolate(self.decoder[3](out4), size=shape, mode='bilinear', align_corners=False)
#
#         out4_up = F.interpolate(out4, bk_edge3.size()[2:], mode='bilinear', align_corners=False)
#         out4_up, _ = torch.chunk(out4_up, 2, dim=1)
#         bk_edge3_up, _ = torch.chunk(bk_edge3, 2, dim=1)
#         out3 = torch.cat((out4_up, bk_edge3_up), dim=1)
#         out3 = self.Trans1(out3)
#         out3 = self.cafm[2](out3)
#         prediction3 = F.interpolate(self.decoder[2](out3), size=shape, mode='bilinear', align_corners=False)
#
#         out3_up = F.interpolate(out3, bk_edge2.size()[2:], mode='bilinear', align_corners=False)
#         out3_up, _ = torch.chunk(out3_up, 2, dim=1)
#         bk_edge2_up, _ = torch.chunk(bk_edge2, 2, dim=1)
#         out2 = torch.cat((out3_up, bk_edge2_up), dim=1)
#         out2 = self.Trans2(out2)
#         out2 = self.cafm[1](out2)
#         prediction2 = F.interpolate(self.decoder[1](out2), size=shape, mode='bilinear', align_corners=False)
#
#         out2_up = F.interpolate(out2, bk_edge1.size()[2:], mode='bilinear', align_corners=False)
#         out2_up, _ = torch.chunk(out2_up, 2, dim=1)
#         bk_edge1_up, _ = torch.chunk(bk_edge1, 2, dim=1)
#         out1 = torch.cat((out2_up, bk_edge1_up), dim=1)
#         out1 = self.Trans3(out1)
#         out1 = self.cafm[0](out1)
#         prediction1 = F.interpolate(self.decoder[0](out1), size=shape, mode='bilinear', align_corners=False)
#
#         return prediction1, prediction2, prediction3, prediction4,
#
#     def load_pre(self, load_rgb, load_depth):
#         self.rgb_backbone.load_state_dict(torch.load("ckps/smt/smt_tiny.pth")['model'])
#         #
#         self.d_backbone.load_state_dict(torch.load(load_depth), strict=False)
#
#
# if __name__ == '__main__':
#     # 创建Net类的实例，传递配置参数
#     net = DASFCNet()
#
#     # 创建一个形状为 (1, 3, 320, 320) 的随机张量
#     x = torch.randn(1, 3, 384, 384)
#
#     d = torch.randn(1, 3, 384, 384)
#
#     # 使用Net类的实例和输入张量x来计算FLOPs和参数数量
#     flops, params = profile(net, (x, d))
#
#     # 打印FLOPs和参数数量
#     print('flops: %.2f G, parms: %.2f M' % (flops / 1000000000.0, params / 1000000.0))


import math

from thop import profile

from smt import smt_t  # 导入自定义的SMT模块
# from MobileNetV2 import mobilenet_v2  # 导入MobileNetV2模型
from mobilevit import mobile_vit_small
import torch.nn as nn  # 导入PyTorch的神经网络模块
import torch  # 导入PyTorch库
import torch.nn.functional as F  # 导入PyTorch的函数模块
from timm.models.layers import trunc_normal_  # 导入timm库中的trunc_normal_函数

TRAIN_SIZE = 384  # 定义训练图像的大小为384x384

import torch.nn.functional as F


# 定义一个名为BasicConv2d的类，它继承自nn.Module
class BasicConv2d(nn.Module):
    # 定义一个构造函数__init__，它接受六个参数：输入通道数in_planes、输出通道数out_planes、卷积核大小kernel_size、步长stride（默认为1）、填充padding（默认为0）和膨胀dilation（默认为1）
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        # 调用父类nn.Module的构造函数
        super(BasicConv2d, self).__init__()
        # 初始化一个卷积层，使用指定的参数。nn.Conv2d是PyTorch中的一个函数，用于创建二维卷积层
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        # 初始化一个批量归一化层，使用指定的输出通道数
        self.bn = nn.BatchNorm2d(out_planes)
        # 初始化一个ReLU激活函数，使用in-place模式
        self.gelu = nn.GELU()

    # 定义一个前向传播方法forward，它接受一个输入x
    def forward(self, x):
        # 应用卷积层处理输入x
        x = self.conv(x)
        # 应用批量归一化层处理卷积层的输出
        x = self.bn(x)

        x = self.gelu(x)

        # 返回处理后的特征x
        return x


class SpatialAttention(nn.Module):
    # 定义一个构造函数__init__，它接受一个参数kernel_size，表示卷积核的大小，默认为7
    def __init__(self, kernel_size=7):
        # 调用父类nn.Module的构造函数
        super(SpatialAttention, self).__init__()
        # 检查kernel_size是否为3或7，如果不是，则抛出异常
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        # 计算padding值，用于卷积层的padding参数
        padding = 3 if kernel_size == 7 else 1

        # 初始化卷积层，用于计算空间注意力权重
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        # 初始化Sigmoid函数，用于生成最终的注意力权重
        self.sigmoid = nn.Sigmoid()

    # 定义一个前向传播方法forward，它接受一个输入x
    def forward(self, x):
        # 计算平均池化后的特征
        avg_out = torch.mean(x, dim=1, keepdim=True)
        # 计算最大池化后的特征
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # 将平均池化后的特征和最大池化后的特征拼接在一起
        x = torch.cat([avg_out, max_out], dim=1)
        # 应用卷积层处理拼接后的特征
        x = self.conv1(x)
        # 应用Sigmoid函数生成最终的注意力权重
        return self.sigmoid(x)


class CAFM(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(CAFM, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, in_channel // 4, kernel_size=3, padding=1),
            BasicConv2d(in_channel // 4, in_channel // 4, kernel_size=3, padding=1, dilation=1)
        )

        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, in_channel // 4, kernel_size=3, padding=1),
            BasicConv2d(in_channel // 4, in_channel // 4, kernel_size=3, padding=3, dilation=3)
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, in_channel // 4, kernel_size=3, padding=1),
            BasicConv2d(in_channel // 4, in_channel // 4, kernel_size=3, padding=5, dilation=5)
        )

        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, in_channel // 4, kernel_size=3, padding=1),
            BasicConv2d(in_channel // 4, in_channel // 4, kernel_size=3, padding=7, dilation=7)
        )

        # 改进的翻转注意力模块
        self.reverse_attention = nn.Sequential(
            nn.Conv2d(in_channel, in_channel // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channel // 4, in_channel, kernel_size=1),
            nn.Sigmoid()
        )

        # 注意力融合门控机制
        self.attention_gate = nn.Sequential(
            nn.Conv2d(in_channel * 2, in_channel // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channel // 2, in_channel, kernel_size=1),
            nn.Sigmoid()
        )

        self.conv_cat = BasicConv2d(in_channel, in_channel, kernel_size=3, padding=1)

        self.sa = SpatialAttention()

        self.fusion = BasicConv2d(in_channel, in_channel, kernel_size=3, padding=1)
        self.conv_down = BasicConv2d(in_channel, out_channel, kernel_size=3, padding=1)

    def forward(self, x_pre):
        # 多尺度处理
        x0 = self.branch0(x_pre)
        x1 = self.branch1(x_pre)
        x2 = self.branch2(x_pre)
        x3 = self.branch3(x_pre)

        x_fused = self.conv_cat(torch.cat((x0, x1, x2, x3), dim=1))

        # 空间注意力
        sa_map = self.sa(x_fused)  # 获取注意力图
        x_sa = sa_map * x_fused

        reverse_att = self.reverse_attention(x_fused)

        # 2. 应用翻转注意力（聚焦被忽略区域）
        x_reverse = reverse_att * (1 - sa_map) * x_fused

        # 3. 门控融合机制
        gate = self.attention_gate(torch.cat([x_sa, x_reverse], dim=1))
        x_dual_att = gate * x_sa + (1 - gate) * x_reverse

        # 残差连接
        x_ff = x_pre + x_dual_att

        # 输出处理
        x_out = self.fusion(x_ff)
        out = self.conv_down(x_out)

        return out


class EnhancedCA(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.shared_conv = nn.Sequential(
            nn.Conv2d(2 * in_planes, 2 * in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(2 * in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, rgb, depth):
        # 双模态特征拼接
        x = torch.cat([rgb, depth], dim=1)

        # 双路径池化
        max_pool = self.shared_conv(nn.AdaptiveMaxPool2d(1)(x))
        avg_pool = self.shared_conv(nn.AdaptiveAvgPool2d(1)(x))

        # 注意力权重生成
        weight = self.sigmoid(max_pool + avg_pool)
        return depth * weight


class EnhancedSA(nn.Module):
    def __init__(self, in_planes, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, rgb, depth):
        # 跨模态空间特征提取
        rgb_max, _ = torch.max(rgb, dim=1, keepdim=True)
        depth_max, _ = torch.max(depth, dim=1, keepdim=True)

        # 空间特征融合
        x = torch.cat([rgb_max, depth_max], dim=1)
        return self.conv(x)


class DAMv2(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.ca = EnhancedCA(in_planes, ratio)
        self.sa = EnhancedSA(in_planes)
        self.cross_att = nn.Sequential(
            nn.Conv2d(in_planes * 2, in_planes, 3, padding=1, bias=False),
            nn.ReLU()
        )

    def forward(self, rgb, depth):
        # 并行双注意力
        ca_out = self.ca(rgb, depth)
        sa_mask = self.sa(rgb, depth)

        # 注意力融合
        sa_out = depth * sa_mask

        # 交叉模态融合
        fused = torch.cat([ca_out, sa_out], dim=1)
        return self.cross_att(fused) + depth


# 坐标注意力模块 (适合方向性边缘)
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        # 调用基类的构造函数
        super(h_sigmoid, self).__init__()
        # ReLU6激活函数，其输出范围被限制在0到6之间
        self.relu = nn.ReLU6(inplace=inplace)

    # 定义前向传播函数，实现h_sigmoid激活函数
    def forward(self, x):
        # 通过将输入x加上3，然后通过ReLU6激活函数，再除以6来模拟sigmoid函数
        return self.relu(x + 3) / 6


# 定义h_swish类，这是一个自定义的激活函数，模仿swish函数的形状
class h_swish(nn.Module):
    def __init__(self, inplace=True):
        # 调用基类的构造函数
        super(h_swish, self).__init__()
        # 使用h_sigmoid作为sigmoid激活函数的实现
        self.sigmoid = h_sigmoid(inplace=inplace)

    # 定义前向传播函数，实现h_swish激活函数
    def forward(self, x):
        # 通过将输入x与通过h_sigmoid的x相乘来实现swish操作
        return x * self.sigmoid(x)


# 这段代码定义了一个名为 CoordAtt 的类，它是一个用于图像处理的注意力模块，继承自 nn.Module  LFE
class CoordAtt(nn.Module):
    # 定义CoordAtt类，这是一个坐标注意力模块
    def __init__(self, inp, oup, reduction=32):
        # 调用基类的构造函数
        super(CoordAtt, self).__init__()
        # 定义两个自适应平均池化层，分别对高度和宽度进行池化
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        # 计算多输入通道（mip），至少为8，最大为inp除以reduction
        mip = max(8, inp // reduction)

        # 定义一个1x1卷积层，后接批量归一化层和h_swish激活函数
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        # 定义两个1x1卷积层，用于生成高度和宽度方向的注意力图
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # 保存输入x作为残差连接
        identity = x

        # 获取输入x的尺寸
        n, c, h, w = x.size()
        # 对x的高度进行池化，得到x_h
        x_h = self.pool_h(x)
        # 对x的宽度进行池化，然后交换H和W的维度，得到x_w
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        # 沿着宽度拼接x_h和x_w，得到y
        y = torch.cat([x_h, x_w], dim=2)
        # 通过第一个1x1卷积层
        y = self.conv1(y)
        # 进行批量归一化
        y = self.bn1(y)
        # 通过激活函数
        y = self.act(y)

        # 将y按照原始的高度和宽度拆分成x_h和x_w
        x_h, x_w = torch.split(y, [h, w], dim=2)
        # 交换x_w的高度和宽度维度，以匹配原始输入的维度
        x_w = x_w.permute(0, 1, 3, 2)

        # 通过conv_h和conv_w生成高度和宽度的注意力图，并应用sigmoid函数
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        # 将输入x与两个方向的注意力图相乘，实现注意力机制
        out = identity * a_w * a_h

        # 返回注意力加权的输出
        return out


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, reduction=1):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class FPEM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=2):
        super(FPEM, self).__init__()

        self.in_channels = in_channels

        self.reduced_channels = max(in_channels // reduction_ratio, 8)  # 通道压缩至合理范围

        # 小波正变换
        self.dwt = DWT()

        # 小波逆变换
        self.iwt = IWT()

        # 通道压缩：统一小波变换前的特征维度
        self.channel_reducer = nn.Sequential(
            nn.Conv2d(in_channels, self.reduced_channels, 1),
            nn.BatchNorm2d(self.reduced_channels),
            nn.GELU()
        )

        # 低频处理模块
        self.ll_processor = nn.Sequential(
            nn.Conv2d(self.reduced_channels, self.reduced_channels, 3, padding=1),
            nn.BatchNorm2d(self.reduced_channels),
            nn.GELU()
        )

        self.ca = ChannelAttention(self.reduced_channels)
        self.sa = SpatialAttention()

        self.lh_processor = nn.Sequential(
            nn.Conv2d(self.reduced_channels, self.reduced_channels, (1, 3), padding=(0, 1)),  # 小核3x1
            nn.BatchNorm2d(self.reduced_channels),
            nn.GELU(),
        )

        # 垂直高频(hl)：垂直方向小核卷积
        self.hl_processor = nn.Sequential(
            nn.Conv2d(self.reduced_channels, self.reduced_channels, (3, 1), padding=(1, 0)),  # 小核1x3
            nn.BatchNorm2d(self.reduced_channels),
            nn.GELU(),
        )

        # 对角高频(hh)：简单3x3卷积
        self.hh_processor = nn.Sequential(
            nn.Conv2d(self.reduced_channels, self.reduced_channels, 3, padding=1),
            nn.BatchNorm2d(self.reduced_channels),
            nn.GELU(),
        )

        # 重构后处理
        # self.cat_processor = nn.Sequential(
        #     # 坐标注意力增强
        #     CoordAtt(3 * self.reduced_channels, 3 * self.reduced_channels),
        #     # 特征融合
        #     nn.Conv2d(3 * self.reduced_channels, 3 * self.reduced_channels, 1),
        #     nn.BatchNorm2d(3 * self.reduced_channels),
        #     nn.GELU()
        # )

        self.pre_att = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels * 2, kernel_size=3, padding=1, groups=in_channels * 2),
            nn.BatchNorm2d(in_channels * 2),
            nn.GELU(),
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.GELU()
        )

        # 特征融合 (恢复原始通道)
        self.fusion = nn.Sequential(
            CoordAtt(self.reduced_channels, self.reduced_channels),
            nn.Conv2d(self.reduced_channels, self.reduced_channels, 3, padding=1),
            nn.BatchNorm2d(self.reduced_channels),
            nn.GELU(),
            nn.Conv2d(self.reduced_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.GELU()
        )

        # 残差连接
        self.residual = nn.Identity()

    def forward(self, x, d=None):
        if d is not None:
            fused = torch.cat([x, d], dim=1)
            x = self.pre_att(fused)

        identity = self.residual(x)

        x = self.channel_reducer(x)

        # 通道压缩
        # 小波分解
        ll, lh, hl, hh = self.dwt(x)
        # 拆分四个子带
        # 1. 低频处理
        ll_processed = self.ll_processor(ll)
        ll_ca = ll_processed * self.ca(ll_processed)
        ll_sa = ll_ca * self.sa(ll_ca)  # [1,256,6,6]

        ll_mean = torch.mean(ll_sa, dim=(2, 3), keepdim=True)  # [1,256,1,1] # 低频全局指导信号

        lh_processed = self.lh_processor(lh) * ll_mean  # [1,256,6,6]

        hl_processed = self.hl_processor(hl) * ll_mean

        hh_processed = self.hh_processor(hh) * ll_mean

        iwt_input = torch.cat([
            ll_sa,  # 处理后的低频 [B, C', H/2, W/2]
            lh_processed,  # 处理后的水平高频 [B, C', H/2, W/2]
            hl_processed,  # 处理后的垂直高频 [B, C', H/2, W/2]
            hh_processed  # 处理后的对角高频 [B, C', H/2, W/2]
        ], dim=1)  # 关键：在批次维度 (dim=0) 拼接 -> [4*B, C', H/2, W/2]

        # 2. 高频合并处理
        # hf_cat = torch.cat([lh_processed, hl_processed, hh_processed], dim=1)
        #
        # hf_processed = self.cat_processor(hf_cat)  # [1,768,6,6]
        # 拆分处理后的高频
        # lh_p, hl_p, hh_p = torch.chunk(hf_processed, 3, dim=1)
        # 3. 小波重构
        recon = self.iwt(iwt_input)

        # 特征融合
        fused1 = self.fusion(recon)

        return fused1 + identity


# class DWT(nn.Module):
#     def __init__(self):
#         super(DWT, self).__init__()
#         self.requires_grad = False
#
#     def forward(self, x):
#         x01 = x[:, :, 0::2, :] / 2
#         x02 = x[:, :, 1::2, :] / 2
#         x1 = x01[:, :, :, 0::2]
#         x2 = x02[:, :, :, 0::2]
#         x3 = x01[:, :, :, 1::2]
#         x4 = x02[:, :, :, 1::2]
#         ll = x1 + x2 + x3 + x4
#         lh = -x1 + x2 - x3 + x4
#         hl = -x1 - x2 + x3 + x4
#         hh = x1 - x2 - x3 + x4
#         return ll, lh, hl, hh
#
#
# class IWT(nn.Module):
#     def __init__(self):
#         super(IWT, self).__init__()
#         self.requires_grad = False
#
#     def forward(self, ll, lh, hl, hh):
#         device = ll.device
#         batch, channel, height, width = ll.shape
#         recon = torch.zeros(batch, channel, height * 2, width * 2).float().to(device)
#         recon[:, :, 0::2, 0::2] = ll - lh - hl + hh
#         recon[:, :, 1::2, 0::2] = ll - lh + hl - hh
#         recon[:, :, 0::2, 1::2] = ll + lh - hl - hh
#         recon[:, :, 1::2, 1::2] = ll + lh + hl + hh
#         return recon


#  缩小hw和扩大b到4b
def dwt_init(x):
    # 第一步：在高度方向上进行下采样（隔行采样）并归一化
    x01 = x[:, :, 0::2, :] / 2  # 选择偶数行并除以2（归一化）
    x02 = x[:, :, 1::2, :] / 2  # 选择奇数行并除以2（归一化）

    # 第二步：在宽度方向上进行下采样（隔列采样）
    x1 = x01[:, :, :, 0::2]  # 从偶数行中选择偶数列（左上象限）
    x2 = x02[:, :, :, 0::2]  # 从奇数行中选择偶数列（左下象限）
    x3 = x01[:, :, :, 1::2]  # 从偶数行中选择奇数列（右上象限）
    x4 = x02[:, :, :, 1::2]  # 从奇数行中选择奇数列（右下象限）

    # 第三步：通过加减组合生成四个子带
    x_LL = x1 + x2 + x3 + x4  # 低频近似：四个象限相加（保留主要信息）
    x_HL = -x1 - x2 + x3 + x4  # 水平细节：垂直边缘信息（奇数行贡献为负）
    x_LH = -x1 + x2 - x3 + x4  # 垂直细节：水平边缘信息（奇数列贡献为负）
    x_HH = x1 - x2 - x3 + x4  # 对角细节：对角边缘和纹理信息

    # 第四步：将四个子带在批次维度上拼接
    return x_LL, x_LH, x_HL, x_HH
    # 第四步：将四个子带在批次维度上拼接


# 使用哈尔 haar 小波变换来实现二维离散小波
# 还原 b和hw
def iwt_init(x):
    # # 设置缩放因子，r=2表示将图像尺寸扩大2倍
    # r = 2
    # # 获取输入张量的维度信息：批次大小、通道数、高度和宽度
    # in_batch, in_channel, in_height, in_width = x.size()
    # # 计算输出张量的维度：批次大小缩小r²倍，通道数不变，高度和宽度扩大r倍
    # out_batch, out_channel, out_height, out_width = int(in_batch / (r ** 2)), in_channel, r * in_height, r * in_width
    #
    # # 将输入张量分割为四个子带，并分别除以2（归一化处理）
    # # 假设输入是按批次排列的四个子带：LL, LH, HL, HH
    # x1 = x[0:out_batch, :, :, :] / 2  # 低频子带LL
    # x2 = x[out_batch:out_batch * 2, :, :, :] / 2  # 水平高频子带LH
    # x3 = x[out_batch * 2:out_batch * 3, :, :, :] / 2  # 垂直高频子带HL
    # x4 = x[out_batch * 3:out_batch * 4, :, :, :] / 2  # 对角高频子带HH
    #
    # # 创建全零输出张量，并将其放置在与输入相同的设备上（CPU或GPU）
    # h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().to(x.device)
    #
    # # 根据逆小波变换的重构规则，将四个子带重新组合到输出张量的不同位置
    # # 偶数行偶数列位置：低频信息减去水平、垂直和对角高频信息
    # h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    # # 奇数行偶数列位置：低频信息减去水平高频，加上垂直高频，减去对角高频
    # h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    # # 偶数行奇数列位置：低频信息加上水平高频，减去垂直高频，减去对角高频
    # h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    # # 奇数行奇数列位置：低频信息加上所有高频信息
    # h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    # batch_size, channels, height, width = x.size()
    # assert channels % 4 == 0, f"通道数必须是4的倍数，当前是{channels}"
    #
    # # 每个子带的通道数
    # sub_channels = channels // 4
    #
    # # 分割四个子带
    # ll = x[:, 0:sub_channels, :, :] / 2  # 低频子带LL
    # lh = x[:, sub_channels:sub_channels * 2, :, :] / 2  # 水平高频子带LH
    # hl = x[:, sub_channels * 2:sub_channels * 3, :, :] / 2  # 垂直高频子带HL
    # hh = x[:, sub_channels * 3:sub_channels * 4, :, :] / 2  # 对角高频子带HH
    #
    # # 计算输出尺寸
    # out_height = height * 2
    # out_width = width * 2
    #
    # # 创建输出张量
    # h = torch.zeros([batch_size, sub_channels, out_height, out_width]).float().to(x.device)
    #
    # # 重构图像
    # h[:, :, 0::2, 0::2] = ll - lh - hl + hh
    # h[:, :, 1::2, 0::2] = ll - lh + hl - hh
    # h[:, :, 0::2, 1::2] = ll + lh - hl - hh
    # h[:, :, 1::2, 1::2] = ll + lh + hl + hh

    # x的形状应为 [B, 4*C, H/2, W/2]
    batch_size, channels, height, width = x.size()
    assert channels % 4 == 0, f"通道数必须是4的倍数，当前是{channels}"

    # 每个子带的通道数
    sub_channels = channels // 4

    # 在通道维度分割四个子带
    ll = x[:, 0:sub_channels, :, :] / 2
    lh = x[:, sub_channels:sub_channels * 2, :, :] / 2
    hl = x[:, sub_channels * 2:sub_channels * 3, :, :] / 2
    hh = x[:, sub_channels * 3:sub_channels * 4, :, :] / 2

    # 计算输出尺寸
    out_height = height * 2
    out_width = width * 2

    # 创建输出张量
    h = torch.zeros([batch_size, sub_channels, out_height, out_width]).float().to(x.device)

    # 重构图像
    h[:, :, 0::2, 0::2] = ll - lh - hl + hh
    h[:, :, 1::2, 0::2] = ll - lh + hl - hh
    h[:, :, 0::2, 1::2] = ll + lh - hl - hh
    h[:, :, 1::2, 1::2] = ll + lh + hl + hh

    return h  # 返回重构后的图像


class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False  # 信号处理，非卷积运算，不需要进行梯度求导

    def forward(self, x):
        return dwt_init(x)


class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)


class Decoder(nn.Module):
    def __init__(self, channel=32):
        super(Decoder, self).__init__()
        self.predict_layer = nn.Sequential(
            nn.Conv2d(channel, 1, kernel_size=1, padding=0),
        )

    def forward(self, x):
        prediction = self.predict_layer(x)

        return prediction


class Trans(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()
        self.trans = nn.Sequential(
            nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=1),  # 1x1卷积层，用于调整通道数
            nn.BatchNorm2d(outc),  # 批归一化层，用于稳定训练
            nn.GELU()  # GELU激活函数，用于非线性变换
        )

    def forward(self, d):
        return self.trans(d)  # 前向传播，通过定义的变换模块


class DASFCNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.rgb_backbone = smt_t(pretrained)  # RGB主干网络
        # self.rgb_backbone = mobile_vit_small()  # RGB主干网络
        # self.d_backbone = mobilenet_v2(pretrained)  # 深度主干网络
        self.d_backbone = mobile_vit_small()

        # self.depth_enhance4 = DAM_module(512*2)
        # self.depth_enhance3 = DAM_module(256*2)
        # self.depth_enhance2 = DAM_module(128*2)
        # self.depth_enhance1 = DAM_module(64*2)

        # self.depth_enhance4 = DAMv2(512)
        # self.depth_enhance3 = DAMv2(256)
        # self.depth_enhance2 = DAMv2(128)
        # self.depth_enhance1 = DAMv2(64)

        self.d_trans_4 = Trans(160, 512)  # 深度特征变换层4
        self.d_trans_3 = Trans(128, 256)  # 深度特征变换层3
        self.d_trans_2 = Trans(96, 128)  # 深度特征变换层2
        self.d_trans_1 = Trans(64, 64)  # 深度特征变换层1

        # Fuse
        self.Trans1 = BasicConv2d(256, 256, kernel_size=3, padding=1)
        self.Trans2 = BasicConv2d(128, 128, kernel_size=3, padding=1)
        self.Trans3 = BasicConv2d(64, 64, kernel_size=3, padding=1)

        self.decoder = nn.ModuleList([
            Decoder(channel=32),
            Decoder(channel=64),
            Decoder(channel=128),
            Decoder(channel=256),
        ])

        # 创建一个包含RFB_modified模块的列表，用于特征的增强处理
        self.cafm = nn.ModuleList([
            CAFM(64, 32),
            CAFM(128, 64),
            CAFM(256, 128),
            CAFM(512, 256)
        ])

        self.fpem = nn.ModuleList([
            FPEM(64),
            FPEM(128),
            FPEM(256),
            FPEM(512)
        ])

        self.deconv_layer_44 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.GELU(),
        )

        #
        self.deconv_layer_43 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.GELU(),
        )

        self.deconv_layer_33 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.GELU(),
        )

        self.deconv_layer_32 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
        )

        self.deconv_layer_22 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
        )

        self.deconv_layer_21 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )

        self.deconv_layer_11 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )

    def forward(self, x_rgb, x_d, shape=None):
        shape = x_rgb.size()[2:] if shape is None else shape
        # rgb [1,64,96,96] [1,96,48,48] [1,128,24,24] [1,160,12,12]

        _, (rgb_1, rgb_2, rgb_3, rgb_4) = self.rgb_backbone(x_rgb)
        _, d_1, d_2, d_3, d_4 = self.d_backbone(x_d)

        d_4 = self.d_trans_4(d_4)
        d_3 = self.d_trans_3(d_3)
        d_2 = self.d_trans_2(d_2)
        d_1 = self.d_trans_1(d_1)

        rgb_43_up = F.interpolate(self.deconv_layer_43(rgb_4), size=rgb_3.size()[2:], mode='bilinear',
                                  align_corners=False)
        d_43_up = F.interpolate(self.deconv_layer_43(d_4), size=d_3.size()[2:], mode='bilinear', align_corners=False)

        rgb3_con = torch.cat((rgb_3, rgb_43_up), dim=1)
        rgb3_con = self.deconv_layer_33(rgb3_con)
        d3_con = torch.cat((d_3, d_43_up), dim=1)
        d3_con = self.deconv_layer_33(d3_con)

        rgb_32_up = F.interpolate(self.deconv_layer_32(rgb_3), size=rgb_2.size()[2:], mode='bilinear',
                                  align_corners=False)
        d_32_up = F.interpolate(self.deconv_layer_32(d_3), size=d_2.size()[2:], mode='bilinear', align_corners=False)

        rgb_21_up = F.interpolate(self.deconv_layer_21(rgb_2), size=rgb_1.size()[2:], mode='bilinear',
                                  align_corners=False)
        d_21_up = F.interpolate(self.deconv_layer_21(d_2), size=d_1.size()[2:], mode='bilinear', align_corners=False)

        rgb4_con = torch.cat((rgb_4, rgb_4), dim=1)
        rgb4_con = self.deconv_layer_44(rgb4_con)
        d4_con = torch.cat((d_4, d_4), dim=1)
        d4_con = self.deconv_layer_44(d4_con)

        rgb2_con = torch.cat((rgb_2, rgb_32_up), dim=1)
        rgb2_con = self.deconv_layer_22(rgb2_con)
        d2_con = torch.cat((d_2, d_32_up), dim=1)
        d2_con = self.deconv_layer_22(d2_con)

        rgb1_con = torch.cat((rgb_1, rgb_21_up), dim=1)
        rgb1_con = self.deconv_layer_11(rgb1_con)
        d1_con = torch.cat((d_1, d_21_up), dim=1)
        d1_con = self.deconv_layer_11(d1_con)

        bk_edge4 = self.fpem[3](rgb4_con, d4_con)
        bk_edge3 = self.fpem[2](rgb3_con, d3_con)
        bk_edge2 = self.fpem[1](rgb2_con, d2_con)
        bk_edge1 = self.fpem[0](rgb1_con, d1_con)

        ################################################ CAEM ####################################

        out4 = self.cafm[3](bk_edge4)

        prediction4 = F.interpolate(self.decoder[3](out4), size=shape, mode='bilinear', align_corners=False)

        out4_up = F.interpolate(out4, bk_edge3.size()[2:], mode='bilinear', align_corners=False)
        out4_up, _ = torch.chunk(out4_up, 2, dim=1)
        bk_edge3_up, _ = torch.chunk(bk_edge3, 2, dim=1)
        out3 = torch.cat((out4_up, bk_edge3_up), dim=1)
        out3 = self.Trans1(out3)
        out3 = self.cafm[2](out3)
        prediction3 = F.interpolate(self.decoder[2](out3), size=shape, mode='bilinear', align_corners=False)

        out3_up = F.interpolate(out3, bk_edge2.size()[2:], mode='bilinear', align_corners=False)
        out3_up, _ = torch.chunk(out3_up, 2, dim=1)
        bk_edge2_up, _ = torch.chunk(bk_edge2, 2, dim=1)
        out2 = torch.cat((out3_up, bk_edge2_up), dim=1)
        out2 = self.Trans2(out2)
        out2 = self.cafm[1](out2)
        prediction2 = F.interpolate(self.decoder[1](out2), size=shape, mode='bilinear', align_corners=False)

        out2_up = F.interpolate(out2, bk_edge1.size()[2:], mode='bilinear', align_corners=False)
        out2_up, _ = torch.chunk(out2_up, 2, dim=1)
        bk_edge1_up, _ = torch.chunk(bk_edge1, 2, dim=1)
        out1 = torch.cat((out2_up, bk_edge1_up), dim=1)
        out1 = self.Trans3(out1)
        out1 = self.cafm[0](out1)
        prediction1 = F.interpolate(self.decoder[0](out1), size=shape, mode='bilinear', align_corners=False)

        # 收集所有中间特征
        # features = {
        #     'rgb_1': rgb_1, 'rgb_2': rgb_2, 'rgb_3': rgb_3, 'rgb_4': rgb_4,
        #     'd_1': d_1, 'd_2': d_2, 'd_3': d_3, 'd_4': d_4,
        #     'rgb1_con': rgb1_con, 'rgb2_con': rgb2_con, 'rgb3_con': rgb3_con, 'rgb4_con': rgb4_con,
        #     'd1_con': d1_con, 'd2_con': d2_con, 'd3_con': d3_con, 'd4_con': d4_con,
        #     'bk_edge1': bk_edge1, 'bk_edge2': bk_edge2, 'bk_edge3': bk_edge3, 'bk_edge4': bk_edge4,
        #     'out1': out1, 'out2': out2, 'out3': out3, 'out4': out4,
        #     'prediction1': prediction1, 'prediction2': prediction2,
        #     'prediction3': prediction3, 'prediction4': prediction4
        # }

        # visualize_feature_map(out1, "out1")
        # visualize_feature_map(prediction1, "prediction1")

        return prediction1, prediction2, prediction3, prediction4


if __name__ == '__main__':
    # 创建Net类的实例，传递配置参数
    net = DASFCNet()

    # 创建一个形状为 (1, 3, 320, 320) 的随机张量
    x = torch.randn(1, 3, 384, 384)

    d = torch.randn(1, 3, 384, 384)

    # 使用Net类的实例和输入张量x来计算FLOPs和参数数量
    flops, params = profile(net, (x, d))

    # 打印FLOPs和参数数量
    print('flops: %.2f G, parms: %.2f M' % (flops / 1000000000.0, params / 1000000.0))
