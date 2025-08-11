import cv2
import numpy as np
from numba import njit
# 瞳孔检测函数：输入灰度图像，输出瞳孔中心和半径，以及瞳孔区域子图
def detect_pupil(gray_img):
    # 1. 预处理：高斯模糊去噪
    blurred = cv2.GaussianBlur(gray_img, (5, 5), 0)
    # 2. 自适应阈值分割瞳孔（瞳孔为黑色区域）
    thresh = cv2.adaptiveThreshold(blurred, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    # 3. 形态学闭运算，填充瞳孔内反光空洞
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    # 4. 查找所有轮廓
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pupil_center, pupil_radius = None, None
    if contours:
        # 遍历轮廓，选择最大且圆形的候选
        max_area = 0
        for contour in contours:
            # 拟合最小外接圆
            (x, y), radius = cv2.minEnclosingCircle(contour)
            area = cv2.contourArea(contour)
            # 圆形度判断：轮廓面积与拟合圆面积接近，且半径在合理范围
            if radius >= 20 and radius <= 300:  # 根据成像比例设置半径范围
                circle_area = np.pi * (radius ** 2)
                # 要求轮廓面积至少是拟合圆的50%
                if area > max_area and area > 0.5 * circle_area:
                    max_area = area
                    pupil_center = (int(x), int(y))
                    pupil_radius = int(radius)
                    print(pupil_center)
    return pupil_center, pupil_radius

# 图像清晰度评价函数：默认采用拉普拉斯方差+ Brenner梯度
# def compute_sharpness(img):
#     # 计算拉普拉斯方差
#     lap = cv2.Laplacian(img, cv2.CV_64F)
#     focus_measure1 = lap.var()  # 拉普拉斯变换的方差
#     # 计算 Brenner 梯度函数值
#     # （注意：直接Python循环计算大图像梯度开销大，这里简单实现）
#     h, w = img.shape
#     diff = img[:, 2:w] - img[:, 0:w-2]    # 计算每行像素与左右隔两个像素的差
#     focus_measure2 = np.sum(diff**2)     # Brenner梯度的和
#     # 可以根据需要对两个指标加权融合
#     F = focus_measure1 + focus_measure2 * 1e-6  # 注意尺度差异，适当缩放后相加
#     return float(F)



@njit
def compute_sharpness(img : np.ndarray) -> float:
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



# 自动对焦（爬山算法）函数：
# move_func: 电机移动函数，返回移动后新的图像
# init_image: 初始位置图像（灰度）
# step_mm: 每步移动的毫米数
# max_iter: 最大迭代步数（防止陷入无限循环）
def autofocus_hill_climbing(init_image, move_func, step_mm=0.5, max_iter=100):
    current_img = init_image
    current_F = compute_sharpness(current_img)
    best_F = current_F
    best_position = 0.0  # 相对初始位置偏移量记录
    # 首先尝试正方向
    next_img = move_func(+step_mm)
    if next_img is None:
        return best_position, best_F  # 无法获取图像则退出
    next_F = compute_sharpness(next_img)
    direction = 0
    if next_F > best_F:
        # 朝正方向清晰度提高
        best_F = next_F
        best_position += step_mm
        direction = +1  # 正方向
    else:
        # 恢复初始位置，再尝试负方向
        next_img = move_func(-step_mm)  # （假设move_func支持负值移动回去）
        if next_img is None:
            return best_position, best_F
        next_F = compute_sharpness(next_img)
        if next_F > best_F:
            best_F = next_F
            best_position -= step_mm
            direction = -1  # 负方向
        else:
            # 两个方向都无提升，认为当前已在最佳位置
            return best_position, best_F

    # 持续沿确定方向搜索
    steps = 1
    while steps < max_iter:
        # 按既定方向继续移动
        next_img = move_func(direction * step_mm)
        if next_img is None:
            break  # 达到边界或无法移动
        next_F = compute_sharpness(next_img)
        if next_F > best_F:
            # 清晰度继续提升
            best_F = next_F
            best_position += direction * step_mm
            steps += 1
        else:
            # 清晰度不再提升，停止搜索
            break
    return best_position, best_F

# ==== 以下为批量测试与电机模拟部分 ====

# 模拟电机控制类：从预先加载的图像列表中返回对应帧
class MotorSimulator:
    def __init__(self, images, start_index=0, step_mm=0.5, start_pos_mm=0.0):
        """
        images: 按焦距顺序的灰度图像列表
        start_index: 初始图像索引
        step_mm: 每步移动的毫米数
        start_pos_mm: 初始图像对应的电机位置（毫米）
        """
        self.images = images
        self.index = start_index
        self.step_mm = step_mm
        self.current_pos = start_pos_mm + start_index * step_mm

    def get_current_image(self):
        if 0 <= self.index < len(self.images):
            return self.images[self.index]
        else:
            return None

    def move(self, delta_mm):
        # 计算将移动的步数（取最接近的离散步）
        steps = int(round(delta_mm / self.step_mm))
        self.index += steps
        self.current_pos += steps * self.step_mm
        # 边界保护
        if self.index < 0:
            self.index = 0
        if self.index >= len(self.images):
            self.index = len(self.images) - 1
        # 返回移动后当前位置的图像
        return self.get_current_image()

# 批量测试函数：输入图像文件列表，模拟自动对焦过程，输出最佳清晰度对应位置
def test_autofocus_on_images(image_files, initial_index=0, step_mm=0.5):
    # 加载图像
    images = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in image_files]
    # 初始化电机模拟器
    sim = MotorSimulator(images, start_index=initial_index, step_mm=step_mm)
    init_img = sim.get_current_image()
    # 若提供的初始索引图像不存在瞳孔，则尝试检测最近的可用帧
    init_center, init_radius = detect_pupil(init_img)
    if init_center is None:
        # 如初始帧瞳孔检测失败，则直接使用图像中心区域计算清晰度
        init_crop = init_img[
            init_img.shape[0]//4: 3*init_img.shape[0]//4,
            init_img.shape[1]//4: 3*init_img.shape[1]//4
        ]
    else:
        # 截取瞳孔附近区域用于清晰度计算
        r = int(init_radius * 1.5)
        cx, cy = init_center
        init_crop = init_img[max(0, cy-r):cy+r, max(0, cx-r):cx+r]
    # 调用自动对焦算法
    best_offset, best_F = autofocus_hill_climbing(init_crop, sim.move, step_mm=step_mm)
    best_position_mm = sim.current_pos  # 获取模拟电机的当前位置（毫米）
    print(f"最佳清晰度={best_F:.2f} 对应电机位置≈{best_position_mm:.2f} mm")
    return best_position_mm, best_F

# ========= 示例测试 =========

# 假设用户提供了 "11.bmp"..."19.bmp" 样本，按焦距由低到高排序
image_files = [f"{i}.bmp" for i in range(11, 20)]
best_pos, best_focus = test_autofocus_on_images(image_files, initial_index=0, step_mm=0.5)
