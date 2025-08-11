import cv2
import numpy as np
from numba import njit
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List
import glob

@njit
def compute_image_sharpness_numba(img: np.ndarray) -> float:
    """使用 Numba 加速计算图像清晰度"""
    h, w = img.shape
    F = 0

    for x in range(1, w - 1, 2):
        for y in range(1, h - 1, 2):
            # 计算 Pave(x, y)
            #pave = (img[y, x] + img[y, x + 1] + img[y, x - 1] + img[y + 1, x] + img[y - 1, x]) / 5
            pave = img[y, x]

            # 计算 G1st(x, y)
            g1st = (abs(img[y, x + 1] - pave) + abs(img[y + 1, x] - pave) + abs(img[y + 1, x + 1] - pave)) ** 2

            # 计算 G2nd(x, y)
            g2nd = (abs(img[y, x + 2] - pave) + abs(img[y + 2, x] - pave) + abs(img[y + 2, x + 2] - pave)) ** 2

            # 计算清晰度 F
            F += g1st * g2nd
    F = F * 1e-6

    return F


def preprocess_image(img: np.ndarray) -> np.ndarray:
    """
    图像预处理：增强对比度和降噪
    """
    # 1. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(img)

    # 2. 高斯滤波降噪
    denoised = cv2.GaussianBlur(enhanced, (3, 3), 0)

    return denoised


def evaluate_contour_quality(contour: np.ndarray, center: Tuple[int, int], radius: int) -> float:
    """
    评估轮廓质量，返回得分
    """
    # 1. 圆形度评估
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return 0

    circularity = 4 * np.pi * area / (perimeter * perimeter)

    # 2. 紧凑性评估
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    if hull_area == 0:
        return 0

    solidity = area / hull_area

    # 3. 面积与半径一致性
    expected_area = np.pi * radius * radius
    area_ratio = min(area / expected_area, expected_area / area)

    # 综合得分
    score = circularity * 0.4 + solidity * 0.3 + area_ratio * 0.3

    return score


def evaluate_circle_quality(img: np.ndarray, circle: Tuple[int, int, int]) -> float:
    """
    评估圆的质量得分（基于灰度对比度）
    """
    x, y, r = circle
    h, w = img.shape

    # 边界检查
    if x - r < 0 or x + r >= w or y - r < 0 or y + r >= h:
        return 0.0

    # 创建mask
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (x, y), r, 255, -1)

    # 计算圆内平均灰度
    inner_mean = cv2.mean(img, mask)[0]

    # 创建环形区域
    outer_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(outer_mask, (x, y), min(int(r * 1.8), min(w, h) // 2), 255, -1)
    cv2.circle(outer_mask, (x, y), r, 0, -1)

    if cv2.countNonZero(outer_mask) == 0:
        return 0.0

    outer_mean = cv2.mean(img, outer_mask)[0]

    # 对比度得分（瞳孔应该比周围区域更暗）
    contrast = outer_mean - inner_mean

    return max(0, contrast)


def detect_pupil_contour(img: np.ndarray, debug: bool = False) -> Optional[Tuple[int, int, int]]:
    """
    使用轮廓检测瞳孔
    """
    # 预处理
    processed = preprocess_image(img)

    # 多种阈值方法
    thresholds = []

    # # 1. 自适应阈值
    # adaptive_thresh = cv2.adaptiveThreshold(
    #     processed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    # )
    # thresholds.append(adaptive_thresh)
    #
    # # 2. Otsu阈值
    # _, otsu_thresh = cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # thresholds.append(otsu_thresh)

    # 3. 固定阈值（多个值）
    for thresh_val in [30, 40, 50, 60]:
        _, fixed_thresh = cv2.threshold(processed, thresh_val, 255, cv2.THRESH_BINARY_INV)
        thresholds.append(fixed_thresh)

    best_circle = None
    best_score = 0

    for thresh in thresholds:
        # 形态学操作
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

        # 查找轮廓
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # 轮廓质量评估
            area = cv2.contourArea(contour)
            if area < 55000:  # 太小的轮廓跳过
                continue

            # 拟合圆
            (x, y), radius = cv2.minEnclosingCircle(contour)
            x, y, radius = int(x), int(y), int(radius)

            if radius < 150 or radius > 400:  # 半径范围检查
                continue

            # 计算轮廓质量得分
            contour_score = evaluate_contour_quality(contour, (x, y), radius)

            # 计算灰度对比度得分
            contrast_score = evaluate_circle_quality(processed, (x, y, radius))

            # 综合得分
            total_score = contour_score * 0.6 + contrast_score * 0.4

            if total_score > best_score:
                best_score = total_score
                best_circle = (x, y, radius)

    if debug and best_circle is not None:
        debug_img = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
        x, y, r = best_circle
        cv2.circle(debug_img, (x, y), r, (0, 255, 0), 2)
        cv2.circle(debug_img, (x, y), 2, (0, 0, 255), 3)
        plt.figure(figsize=(8, 6))
        plt.imshow(debug_img)
        plt.title(f"Contour Detection Result (Score: {best_score:.2f})")
        plt.axis('off')
        plt.show()

    return best_circle


def robust_pupil_detection(img: np.ndarray, debug: bool = False) -> Tuple[Optional[np.ndarray], float]:
    """
    鲁棒的瞳孔检测主函数
    """
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 使用轮廓检测
    detected_circle = detect_pupil_contour(img, debug=debug)

    if detected_circle is None:
        return None, 0.0

    # 提取瞳孔区域并计算清晰度
    x, y, radius = detected_circle
    print(f"瞳孔半径：{radius}像素")
    print(f"瞳孔中心位置:（{x} , {y}）")

    # 扩展区域用于清晰度计算
    expand_factor = 1.5
    expanded_radius = int(radius * expand_factor)

    # 边界检查
    h, w = img.shape
    x1 = max(0, x - expanded_radius)
    y1 = max(0, y - expanded_radius)
    x2 = min(w, x + expanded_radius)
    y2 = min(h, y + expanded_radius)

    crop = img[y1:y2, x1:x2]

    if crop.size == 0:
        return None, 0.0

    # 计算清晰度
    sharpness = compute_image_sharpness_numba(crop)


    # 在裁剪图像上标记瞳孔
    crop_display = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    center_x_crop = x - x1
    center_y_crop = y - y1
    cv2.circle(crop_display, (center_x_crop, center_y_crop), radius, (0, 255, 0), 2)
    cv2.circle(crop_display, (center_x_crop, center_y_crop), 2, (0, 0, 255), 3)


    '''
    ###################################
    '''

    # # 1. 计算外圆半径和边界框
    # expand_factor = 1.5
    # outer_radius = int(radius * expand_factor)
    #
    # # 获取图像的高度和宽度
    # h, w = img.shape[:2]
    #
    # # 计算包含外圆的正方形边界框 (Region of Interest, ROI)
    # # 确保边界不会超出图像范围
    # x1 = max(0, x - outer_radius)
    # y1 = max(0, y - outer_radius)
    # x2 = min(w, x + outer_radius)
    # y2 = min(h, y + outer_radius)
    #
    # # 从原图中裁剪出这个矩形ROI
    # roi = img[y1:y2, x1:x2]
    #
    # # 2. 创建掩码 (Mask)
    # # 创建一个和ROI一样大的全黑掩码
    # mask = np.zeros(roi.shape[:2], dtype="uint8")
    #
    # # 计算圆心在ROI内的相对坐标
    # center_x_roi = x - x1
    # center_y_roi = y - y1
    #
    # # 在掩码上画一个白色的实心大圆（外圆）
    # cv2.circle(mask, (center_x_roi, center_y_roi), outer_radius, 255, -1)
    #
    # # 在掩码上画一个黑色的实心小圆（内圆），从而形成一个白色的圆环
    # cv2.circle(mask, (center_x_roi, center_y_roi), radius, 0, -1)
    #
    # # 3. 应用掩码，得到最终的圆环图像
    # # 只有当掩码的像素是白色(255)时，才保留ROI中对应像素的值
    # ring_crop = cv2.bitwise_and(roi, roi, mask=mask)
    #
    # if ring_crop.size == 0:
    #     return None, 0.0
    #
    # # 计算清晰度
    # sharpness = compute_image_sharpness_numba(ring_crop)
    #
    #
    #
    # # 在裁剪图像上标记瞳孔
    # crop_display = cv2.cvtColor(ring_crop, cv2.COLOR_GRAY2BGR)
    # center_x_crop = x - x1
    # center_y_crop = y - y1
    # cv2.circle(crop_display, (center_x_crop, center_y_crop), radius, (0, 255, 0), 2)
    # cv2.circle(crop_display, (center_x_crop, center_y_crop), 2, (0, 0, 255), 3)

    # sharpness = compute_image_sharpness_numba(img)
    #
    # # 在裁剪图像上标记瞳孔
    # crop_display = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # center_x_crop = x - x1
    # center_y_crop = y - y1
    # cv2.circle(crop_display, (center_x_crop, center_y_crop), radius, (0, 255, 0), 2)
    # cv2.circle(crop_display, (center_x_crop, center_y_crop), 2, (0, 0, 255), 3)

    return crop_display, sharpness


def process_single_image(image_path: str, debug: bool = True) -> Tuple[float, bool]:
    """
    处理单张图像
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"无法读取图像: {image_path}")
        return 0.0, False

    crop_result, sharpness = robust_pupil_detection(img, debug=debug)

    if crop_result is not None:
        print(f"检测成功! 清晰度: {sharpness:.2f}")

        if debug:
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(img, cmap='gray')
            plt.title("Original image")
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.imshow(cv2.cvtColor(crop_result, cv2.COLOR_BGR2RGB))
            plt.title(f"Result (Sharpness: {sharpness:.2f})")
            plt.axis('off')
            plt.tight_layout()
            plt.show()

        return sharpness, True
    else:
        print("瞳孔检测失败")
        return 0.0, False


def batch_process_images(image_paths: List[str], debug: bool = False) -> List[float]:
    """
    批量处理图像
    """
    sharpness_values = []
    success_count = 0

    for i, image_path in enumerate(image_paths, 1):
        print(f"\n处理图像 {i}/{len(image_paths)}: {image_path}")
        sharpness, success = process_single_image(image_path, debug=debug)
        sharpness_values.append(sharpness)

        if success:
            success_count += 1

    print(f"\n总体统计:")
    print(f"成功检测: {success_count}/{len(image_paths)}")
    print(f"成功率: {success_count / len(image_paths) * 100:.1f}%")

    if sharpness_values:
        plt.figure(figsize=(12, 6))
        plt.plot()
        plt.plot(range(1, len(sharpness_values) + 1), sharpness_values, 'bo-')
        plt.xlabel('Image Count')
        plt.ylabel('Sharpness Value')
        plt.title('Sharpness Distribution')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return sharpness_values


# 使用示例
if __name__ == "__main__":
    # 单张图像处理示例
    # sharpness, success = process_single_image("S5233R02.jpg", debug=True)

    # 批量处理示例

    image_paths = glob.glob("yjx2/*.png", recursive=True)
    # image_paths = glob.glob("CASIA-IrisV4(JPG)/CASIA-Iris-Twins/**/*.jpg", recursive=True)
    # image_paths = [f"{i}.bmp" for i in range(1, 21)]
    sharpness_values = batch_process_images(image_paths, debug=False)

    pass

