import numpy as np
import matplotlib.pyplot as plt

# 加载 .npz 文件
# data = np.load('/home/Chengtai_Li/文档/AVR-PredRNet-main/datasets/RAVEN/distribute_nine/RAVEN_118_test.npz')
# data = np.load('/home/Chengtai_Li/文档/AVR-PredRNet-main/datasets/I-RAVEN/distribute_nine/RAVEN_9888_test.npz')
data = np.load('/home/Chengtai_Li/文档/AVR-PredRNet-main/datasets/RAVEN/in_distribute_four_out_center_single/RAVEN_9999_test.npz')
# 假设 .npz 文件里只有一个数组，用以下方式获取数据
print(data['target'])
array = data['image']

# 创建图片
for i in range(16):
    plt.figure(figsize=(10, 10))  # 设置画布尺寸以确保高分辨率
    plt.imshow(array[i], cmap='gray')  # 可更换颜色映射
    plt.axis('off')  # 关闭坐标轴以获得纯图像

    # 保存为高清 .png
    plt.savefig(f'output{i}.png', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()
