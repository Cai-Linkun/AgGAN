import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import re


def save_grey_diff(r, f, output_path):
    # 计算像素差
    diff = cv2.absdiff(r, f)
    cv2.imwrite(output_path, diff)
    print(f"save at {output_path}")


def save_color_diff(r, f, output_path):
    # 计算像素差
    diff = cv2.absdiff(r, f)
    diff_norm = diff / 255.0  # 归一化到 0-1

    x = []
    y = []
    c = []

    for i in range(diff.shape[0]):
        for j in range(diff.shape[1]):
            if diff[i][j] > 0.5:
                x.append(j)
                y.append(i)
                c.append(diff_norm[i][j])

    # 颜色映射：相同区域白色，差异区域蓝色
    fig = plt.figure(figsize=(6, 6))
    plt.scatter(x, y, c=c, s=(72.0 / fig.dpi) ** 2, cmap="Blues", marker=".", vmin=0, vmax=1)  # s 控制点的大小

    # 反转 Y 轴，使得 (0,0) 在左上角
    plt.colorbar()
    # plt.gca().invert_yaxis()
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.xlim(0, 63)
    plt.ylim(63, 0)

    # 关闭坐标轴
    plt.axis("off")

    # 保存图片
    plt.savefig(output_path)
    print(f"save at {output_path}")
    # plt.show()


def gen_all2(path):
    all_files = {}
    for file in os.listdir(path):
        f = os.path.join(path, file)
        match = re.search(r"(testee\d+_layer\d+)", file)
        if match:
            key = match.group(1)
            if key not in all_files:
                all_files[key] = [0, 0]
            if "fake" in file:
                all_files[key][1] = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            elif "real" in file:
                all_files[key][0] = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            print("match key: ", key)
    print("all files: ", len(all_files))

    for k, (r, f) in all_files.items():

        of = os.path.join(path, k)
        grey = of + "_error_map_grey_2.png"
        color = of + "_error_map_color_2.png"
        old = of + "_error_map.png"
        if os.path.exists(grey):
            os.remove(grey)
        if os.path.exists(color):
            os.remove(color)
        if os.path.exists(old):
            os.remove(old)
        save_grey_diff(r, f, grey)
        save_color_diff_2(r, f, color)


def gen_all(path):
    all_files = {}
    for file in os.listdir(path):
        f = os.path.join(path, file)
        match = re.search(r"(testee\d+_layer\d+)", file)
        if match:
            key = match.group(1)
            if key not in all_files:
                all_files[key] = [0, 0]
            if "fake" in file:
                all_files[key][1] = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            elif "real" in file:
                all_files[key][0] = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            print("match key: ", key)
    print("all files: ", len(all_files))

    cols = 3
    rows = len(all_files)

    plt.figure()  # 控制画布大小

    i = 1
    for k, (r, f) in all_files.items():
        plt.subplot(i, 3, 1)
        plt.imshow(r, cmap="grey")
        plt.axis("off")
        plt.subplot(i, 3, 2)
        plt.imshow(f, cmap="grey")
        plt.axis("off")

        plt.subplot(i, 3, 3)

        diff = cv2.absdiff(r, f)
        # cv2.imshow("3", diff)
        diff_norm = diff / 255.0  # 归一化到 0-1

        x = []
        y = []
        c = []

        for _i in range(diff.shape[0]):
            for _j in range(diff.shape[1]):
                if diff[_i][_j] > 0.5:
                    x.append(_j)
                    y.append(_i)
                    c.append(diff_norm[_i][_j])

        # 颜色映射：相同区域白色，差异区域蓝色
        plt.scatter(x, y, c=c, s=1, cmap="Blues", marker=".", vmin=0, vmax=1)  # s 控制点的大小
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
        plt.xlim(0, 63)
        plt.ylim(63, 0)

        # 关闭坐标轴
        plt.axis("off")

        i += 1
        if i > 10:
            break
        # break

    # plt.sub
    plt.tight_layout()
    plt.show()


from PIL import Image
import numpy as np


def gray_to_blue_white(input_path, output_path):
    # 读取灰度图
    img = Image.open(input_path).convert("L")
    arr = np.array(img)

    # 创建RGB蓝色渐变映射
    blue_ratio = arr / 255.0  # 原图亮度作为蓝色强度
    white_ratio = 1.0 - blue_ratio  # 白色部分比例

    # RGB通道计算
    red = np.uint8(255 * white_ratio)
    green = np.uint8(255 * white_ratio)
    blue = np.uint8(255 * (white_ratio * 0.2 + blue_ratio))  # 加强蓝色纯度

    # 合并通道
    result = Image.fromarray(np.stack([red, green, blue], axis=2), "RGB")
    result.save(output_path)


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


def map_show():

    # 创建白到蓝的渐变映射
    colors = [(1, 1, 1), (0, 0, 1)]  # RGB格式，白色到蓝色
    cmap = LinearSegmentedColormap.from_list("white_blue", colors, N=256)

    # 创建数据（垂直方向渐变）
    data = np.linspace(0, 1, 100).reshape(-1, 1)  # 100行1列的渐变数据

    # 绘制图像
    fig, ax = plt.subplots(figsize=(2, 6))
    img = ax.imshow(data, cmap=cmap, aspect="auto", origin="lower")

    # 设置坐标轴
    ax.set_xticks([])  # 隐藏X轴
    ax.set_yticks([0, 99])  # 设置Y轴刻度位置
    ax.set_yticklabels(["0", "1"])  # 设置刻度标签
    ax.spines[:].set_visible(False)  # 隐藏边框

    plt.tight_layout()
    plt.show()


def save_color_diff_2(r, f, o):
    # 计算像素差
    import matplotlib.cm as cm

    diff = cv2.absdiff(r, f)
    X, Y = np.meshgrid(range(0, 64), range(0, 64))
    Z = diff
    cmap = cm.get_cmap("Blues")
    cmap.set_under("white")
    plt.pcolormesh(X, Y, Z, shading="nearest", cmap=cmap, vmin=1, vmax=255)
    plt.colorbar()
    plt.gca().invert_yaxis()
    plt.axis("off")
    plt.savefig(o)
    print(f"save at {o}")
    plt.clf()


if __name__ == "__main__":
    # visualize_pixel_difference_scatter(
    #     "./results/base/test_latest/images/testee3_layer19_real_B_vis.png",
    #     "./results/base/test_latest/images/testee3_layer19_fake_B_vis.png",
    #     "diff_output.png",
    # )
    gen_all2("./results/pp_aal_test1/test_latest/images")

    # gray_to_blue_white("./results/base/test_latest/images/testee5_layer26_error_map_grey_2.png", "test_result.png")
    # map_show()
    # test(
    #     "./results/base/test_latest/images/testee9_layer25_real_B_vis.png",
    #     "./results/base/test_latest/images/testee9_layer25_fake_B_vis.png",
    # )
