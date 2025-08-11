import cv2
import numpy as np
import glob
from numba import njit
import matplotlib.pyplot as plt

@njit
def compute_image_sharpness_fast(img: np.ndarray) -> float:
    """
    利用向量化计算图像清晰度评价值 F。

    参数：
        img: 2D numpy 数组, 灰度图像，shape = (M, N)

    返回：
        F: float, 图像的清晰度评价值
    """

    M, N = img.shape

    # 有效区域要求：
    # 为计算 Pave(x,y)= (f(x,y)+f(x+1,y)+f(x,y+1)+f(x-1,y)+f(x,y-1))/5
    # 和后续 G1st、G2nd，需要保证：
    #   f(x-1,y) 与 f(x,y-1) 存在 => x>=1, y>=1
    #   f(x+2,y), f(x,y+2), f(x+2,y+2) 存在 => x<=M-3, y<=N-3
    # 因此 x 的有效取值为 [1, M-3]，y 的有效取值为 [1, N-3]，
    # 对应的切片区域形状为 (M-3, N-3).

    # 取出有效区域对应的各个分量：
    # 中心像素 f(x,y)
    f_center  = img[1:M - 2, 1:N - 2]
    # f(x+1,y)
    f_xp = img[2:M - 1, 1:N - 2]
    # f(x,y+1)
    f_yp = img[1:M - 2, 2:N - 1]
    # f(x-1,y)
    f_xm = img[0:M - 3, 1:N - 2]
    # f(x,y-1)
    f_ym = img[1:M - 2, 0:N - 3]

    # 计算局部平均 Pave
    Pave = (f_center + f_xp + f_yp + f_xm + f_ym) / 5.0

    # 计算 G_1st:
    # f(x+1,y): f_xp   -> 已计算上面
    # f(x,y+1): f_yp
    # f(x+1,y+1): 对应切片为 img[2:M-1, 2:N-1]
    f_xp_yp = img[2:M - 1, 2:N - 1]
    G_1st = (np.abs(f_xp - Pave) +
             np.abs(f_yp - Pave) +
             np.abs(f_xp_yp - Pave)) ** 2

    # 计算 G_2nd:
    # f(x+2,y): img[3:M,   1:N-2]
    # f(x,y+2): img[1:M-2, 3:N]
    # f(x+2,y+2): img[3:M,   3:N]
    f_xp2 = img[3:M, 1:N - 2]
    f_yp2 = img[1:M - 2, 3:N]
    f_xp2_yp2 = img[3:M, 3:N]
    G_2nd = (np.abs(f_xp2 - Pave) +
             np.abs(f_yp2 - Pave) +
             np.abs(f_xp2_yp2 - Pave)) ** 2

    # 最终清晰度 F 为所有位置 G_1st * G_2nd 的和
    F = np.sum(G_1st * G_2nd)

    return F

@njit
def crop_center(image, scale=0.75):
    H, W = image.shape[:2]  # 获取原始图像尺寸
    H_new, W_new = int(H * scale), int(W * scale)  # 计算目标尺寸

    start_x = (W - W_new) // 2
    start_y = (H - H_new) // 2
    end_x = start_x + W_new
    end_y = start_y + H_new

    return image[start_y:end_y, start_x:end_x]  # 返回裁剪后的图像


@njit
def compute_image_sharpness_numba(img : np.ndarray) -> float:
    """ 使用 OpenCV 计算图像的清晰度 F """
    h, w = img.shape

    F = 0

    for x in range(1, w - 1, 2):
        for y in range(1, h - 1, 2):
            # 计算 Pave(x, y)
            pave = (img[y, x] + img[y, x + 1] + img[y, x - 1] + img[y + 1, x] + img[y - 1, x]) / 5

            # 计算 G1st(x, y)
            g1st = (abs(img[y, x + 1] - pave) + abs(img[y + 1, x] - pave) + abs(img[y + 1, x + 1] - pave)) ** 2

            # 计算 G2nd(x, y)
            g2nd = (abs(img[y, x + 2] - pave) + abs(img[y + 2, x] - pave) + abs(img[y + 2, x + 2] - pave)) ** 2

            # 计算清晰度 F
            F += g1st * g2nd
    F = F * 1e-6

    return F

@njit
def compute_image_sharpness_bre2d_rob(img: np.ndarray) -> float:
    """ 计算图像的清晰度 F_Bre2d,Rob，并避免数值溢出 """

    img = img.astype(np.float64)  # 转换为 float64 以避免溢出
    h, w = img.shape
    F = 0.0

    for x in range(0, w - 2):  # 修正索引范围，防止越界
        for y in range(0, h - 2):
            term1 = (img[y, x + 2] - img[y, x])
            term2 = (img[y + 2, x] - img[y, x])
            term3 = abs(img[y, x] - img[y + 1, x + 1]) + abs(img[y, x + 1] - img[y + 1, x])
            F += abs(term1 * term2 * term3)  # 计算累加

    return F


@njit
def compute_brenner_sharpness(img : np.ndarray) -> float:
    """ 使用 Brenner 梯度函数计算图像清晰度 """

    h, w = img.shape

    F_brenner = 0

    for x in range(w - 2):  # 避免越界
        for y in range(h):
            diff = img[y, x + 2] - img[y, x]  # 计算 f(x+2, y) - f(x, y)
            F_brenner += diff ** 2  # 累加平方值

    return F_brenner



def compute_image_sharpness(img : np.ndarray) -> float:
    """ 使用 OpenCV 计算图像的清晰度 F """
    img = crop_center(img)
    h, w = img.shape

    F = 0

    for x in range(2, w - 2, 2):
        for y in range(2, h - 2, 2):
            # 计算 Pave(x, y)
            pave = (img[y, x] + img[y, x + 1] + img[y, x - 1] + img[y + 1, x] + img[y - 1, x]) / 5

            # 计算 G1st(x, y)
            g1st = (abs(img[y, x + 1] - pave) + abs(img[y + 1, x] - pave) + abs(img[y + 1, x + 1] - pave)) ** 2

            # 计算 G2nd(x, y)
            g2nd = (abs(img[y, x + 2] - pave) + abs(img[y + 2, x] - pave) + abs(img[y + 2, x + 2] - pave)) ** 2

            # 计算清晰度 F
            F += g1st * g2nd

    return F



def img(img : np.ndarray) -> (np.ndarray, float):
    # 高斯模糊去噪 (参数可调)
    crop = None
    sharpness = 0
    # 自适应阈值二值化 (关键参数)
    _, thresh = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY_INV)
    # 形态学操作：闭运算填充小孔
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # 查找轮廓
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    # 设定瞳孔半径范围（根据实际情况调整）
    min_radius = 30
    max_radius = 250

    # 遍历轮廓，寻找最合适的瞳孔轮廓
    pupil_contour = None
    best_radius = 0

    for contour in contours:
        (x, y), radius = cv2.minEnclosingCircle(contour)
        radius = int(radius)
        if min_radius <= radius <= max_radius:
            if radius > best_radius:  # 选择最大合适的圆
                best_radius = radius
                pupil_contour = contour

    if pupil_contour is not None:
        (x, y), radius = cv2.minEnclosingCircle(pupil_contour)
        center = (int(x), int(y))
        radius = int(radius)

        radius = int(2 * radius)
        x = int(center[0] - radius)
        y = int(center[1] - radius)
        size = 2 * radius

        crop = img[y:y + size, x:x + size]

        sharpness = compute_image_sharpness_numba(crop)

        radius = int(radius / 2)
        centerx = int(size / 2)
        centery = int(size / 2)
        cv2.circle(crop, (centerx, centery), radius, (255, 0, 0), 3)

    return crop, sharpness


def img1():
    # 高斯模糊去噪 (参数可调)
    sharpness = 0
    crop = None
    sharp = []
    image_paths = glob.glob("zch_pupil/*.bmp", recursive=True)
    for image_path in image_paths:
        sharpness = 0
        crop = None
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)



        # 自适应阈值二值化 (关键参数)
        _, thresh = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY_INV)



        # 形态学操作：闭运算填充小孔
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # 查找轮廓
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


        # 设定瞳孔半径范围（根据实际情况调整）
        min_radius = 30
        max_radius = 250

        # 遍历轮廓，寻找最合适的瞳孔轮廓
        pupil_contour = None
        best_radius = 0

        for contour in contours:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            radius = int(radius)
            if min_radius <= radius <= max_radius:
                if radius > best_radius:  # 选择最大合适的圆
                    best_radius = radius
                    pupil_contour = contour


        if pupil_contour is not None:


            (x, y), radius = cv2.minEnclosingCircle(pupil_contour)
            center = (int(x), int(y))
            radius = int(radius)


            c = 2.5
            radius = int(c * radius)
            x = int(center[0] - radius)
            y = int(center[1] - radius)
            size = 2 * radius

            crop = img[y:y + size, x:x + size]

            # sharpness = compute_image_sharpness_bre2d_rob(crop)
            sharpness = compute_image_sharpness_numba(crop)
            # sharpness = compute_brenner_sharpness(crop)
            radius = int(radius / c)
            centerx = int(size / 2)
            centery = int(size / 2)
            cv2.circle(crop, (centerx, centery), radius, (255, 0, 0), 3)

        sharp.append(sharpness)
        print('sharpness:', sharpness, '/n')

        plt.figure(figsize=(6, 6))
        if crop is not None:
            plt.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title("Detected Pupil Contour")
        plt.axis("off")
        plt.show()

    plt.plot(sharp)
    plt.show()

    return

img1()
# output= img1()
# # 显示结果
# plt.figure(figsize=(6, 6))
# plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
# plt.title("Detected Pupil Contour")
# plt.axis("off")
# plt.show()
