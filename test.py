import cv2
import numpy as np
from numba import njit
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List
import glob


def extract_lower_pupil_roi(img, detected_circle):
    """
    方案一：提取瞳孔中心以下的区域，避开上方睫毛
    """
    if detected_circle is None:
        return None, None

    x, y, radius = detected_circle
    h, w = img.shape[:2]

    print(f"瞳孔半径：{radius}像素")
    print(f"瞳孔中心位置:（{x}, {y}）")

    # 只提取瞳孔中心以下的区域
    # 横向：瞳孔左右各扩展1.3倍半径
    # 纵向：从瞳孔中心开始，向下扩展1.5倍半径
    expand_factor_x = 1.3
    expand_factor_y = 1.8

    # 计算ROI边界
    x1 = max(0, int(x - radius * expand_factor_x))
    x2 = min(w, int(x + radius * expand_factor_x))
    y1 = max(0, y)  # 从瞳孔中心开始，不包括上半部分
    y2 = min(h, int(y + radius * expand_factor_y))

    # 提取ROI
    roi = img[y1:y2, x1:x2]

    # 返回ROI和位置信息（用于可视化）
    roi_info = {
        'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
        'center': (x, y), 'radius': radius
    }

    return roi, roi_info


def extract_lower_iris_roi(img, detected_circle):
    """
    方案二：提取瞳孔下方的虹膜区域，完全避开瞳孔和上方睫毛
    """
    if detected_circle is None:
        return None, None

    x, y, radius = detected_circle
    h, w = img.shape[:2]

    print(f"瞳孔半径：{radius}像素")
    print(f"瞳孔中心位置:（{x}, {y}）")

    # 虹膜区域通常是瞳孔半径的2-3倍
    # 我们提取瞳孔下方的一个矩形虹膜区域

    # 横向：瞳孔左右各扩展1.5倍半径（虹膜宽度）
    # 纵向：从瞳孔边缘下方开始，高度为0.8倍半径
    iris_width_factor = 1.25
    roi_height_factor = 0.15

    # 计算ROI边界
    x1 = max(0, int(x - radius * iris_width_factor))
    x2 = min(w, int(x + radius * iris_width_factor))
    y1 = max(0, int(y + radius * 1.05))  # 从瞳孔边缘稍下方开始
    y2 = min(h, int(y + radius * (1.05 + roi_height_factor)))

    # 确保ROI有效
    if y2 <= y1 or x2 <= x1:
        print("警告：ROI无效，使用备用方案")
        # 备用方案：使用瞳孔下方的固定大小区域
        y1 = max(0, int(y + radius))
        y2 = min(h, y1 + 100)  # 固定高度100像素
        x1 = max(0, x - 100)
        x2 = min(w, x + 100)

    # 提取ROI
    roi = img[y1:y2, x1:x2]

    # 返回ROI和位置信息
    roi_info = {
        'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
        'center': (x, y), 'radius': radius
    }

    return roi, roi_info


def extract_sector_roi(img, detected_circle, start_angle=45, end_angle=135):
    """
    方案三：提取扇形区域，更精确地避开睫毛
    start_angle, end_angle: 角度，0度为正右方，顺时针增加
    默认30-150度为下方扇形区域
    """
    if detected_circle is None:
        return None, None

    x, y, radius = detected_circle
    h, w = img.shape[:2]

    print(f"瞳孔半径：{radius}像素")
    print(f"瞳孔中心位置:（{x}, {y}）")

    # 创建mask
    mask = np.zeros((h, w), dtype=np.uint8)

    # 扇形的内外半径
    inner_radius = int(radius * 1)  # 避开瞳孔中心
    outer_radius = int(radius * 1.8)  # 包含虹膜区域

    # 绘制扇形mask
    # OpenCV的ellipse函数可以绘制扇形
    cv2.ellipse(mask, (x, y), (outer_radius, outer_radius),
                0, start_angle, end_angle, 255, -1)
    cv2.ellipse(mask, (x, y), (inner_radius, inner_radius),
                0, start_angle, end_angle, 0, -1)

    # 应用mask提取ROI
    masked_img = cv2.bitwise_and(img, img, mask=mask)

    # 找到mask的边界框，裁剪出最小矩形
    coords = np.column_stack(np.where(mask > 0))
    if len(coords) > 0:
        y1, x1 = coords.min(axis=0)
        y2, x2 = coords.max(axis=0)
        roi = masked_img[y1:y2 + 1, x1:x2 + 1]
        mask_roi = mask[y1:y2 + 1, x1:x2 + 1]
    else:
        roi = masked_img
        mask_roi = mask

    roi_info = {
        'mask': mask_roi,
        'center': (x, y),
        'radius': radius,
        'full_mask': mask
    }

    return roi, roi_info

@njit
def compute_image_sharpness_numba(img: np.ndarray) -> float:
    """使用 Numba 加速计算图像清晰度"""
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

    # 计算实际处理的像素数（近似值）
    count = ((w - 3) // 2) * ((h - 3) // 2)

    if count > 0:
        F = F * 0.1 / count
    else:
        F = 0

    return F


@njit
def compute_sharpness_laplacian(img: np.ndarray) -> float:
    """Laplacian方差法 - 对失焦特别敏感"""
    h, w = img.shape
    laplacian_sum = 0.0
    count = 0

    for y in range(1, h - 1):
        for x in range(1, w - 1):
            # 计算Laplacian
            laplacian = (img[y - 1, x] + img[y + 1, x] +
                         img[y, x - 1] + img[y, x + 1] -
                         4 * img[y, x])
            laplacian_sum += laplacian * laplacian
            count += 1

    return laplacian_sum / count

@njit
def variance_of_laplacian(img: np.ndarray) -> float:
    """
    Variance of Laplacian - One of the most reliable focus measures.
    Higher variance indicates sharper image.
    """
    h, w = img.shape
    laplacian = np.zeros_like(img, dtype=np.float64)

    # Apply Laplacian kernel
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            laplacian[y, x] = (
                    img[y - 1, x] + img[y + 1, x] + img[y, x - 1] + img[y, x + 1]
                    - 4 * img[y, x]
            )

    # Calculate variance
    mean_val = np.mean(laplacian)
    variance = np.mean((laplacian - mean_val) ** 2)

    return variance


@njit
def tenenbaum_gradient(img: np.ndarray) -> float:
    """
    Tenenbaum gradient (Tenengrad) - Uses Sobel operators.
    Sums the square of gradient magnitudes.
    """
    h, w = img.shape
    gradient_sum = 0.0

    for y in range(1, h - 1):
        for x in range(1, w - 1):
            # Sobel X
            gx = (img[y - 1, x + 1] + 2 * img[y, x + 1] + img[y + 1, x + 1] -
                  img[y - 1, x - 1] - 2 * img[y, x - 1] - img[y + 1, x - 1]) / 8.0

            # Sobel Y
            gy = (img[y + 1, x - 1] + 2 * img[y + 1, x] + img[y + 1, x + 1] -
                  img[y - 1, x - 1] - 2 * img[y - 1, x] - img[y - 1, x + 1]) / 8.0

            gradient_sum += gx * gx + gy * gy

    return gradient_sum / (h * w)


@njit
def brenner_gradient(img: np.ndarray) -> float:
    """
    Brenner's focus measure - Simple but effective.
    Uses squared differences between pixels separated by 2 positions.
    """
    h, w = img.shape
    focus_measure = 0.0
    count = 0

    # Horizontal differences
    for y in range(h):
        for x in range(w - 2):
            diff = float(img[y, x + 2]) - float(img[y, x])
            focus_measure += diff * diff
            count += 1

    # Vertical differences
    for y in range(h - 2):
        for x in range(w):
            diff = float(img[y + 2, x]) - float(img[y, x])
            focus_measure += diff * diff
            count += 1

    return focus_measure / count if count > 0 else 0.0


@njit
def normalized_variance(img: np.ndarray) -> float:
    """
    Normalized variance - Simple but often effective for focus detection.
    Less sensitive to illumination changes.
    """
    h, w = img.shape
    mean_val = np.mean(img)

    if mean_val == 0:
        return 0.0

    variance = 0.0
    for y in range(h):
        for x in range(w):
            diff = img[y, x] - mean_val
            variance += diff * diff

    variance = variance / (h * w)
    # Normalize by mean to reduce illumination dependency
    return variance / mean_val


@njit
def energy_of_gradient(img: np.ndarray) -> float:
    """
    Energy of gradient - Squared gradient method.
    Simple first-order derivative approach.
    """
    h, w = img.shape
    energy = 0.0

    for y in range(h - 1):
        for x in range(w - 1):
            dx = float(img[y, x + 1]) - float(img[y, x])
            dy = float(img[y + 1, x]) - float(img[y, x])
            energy += dx * dx + dy * dy

    return energy / ((h - 1) * (w - 1))


@njit
def modified_laplacian(img: np.ndarray) -> float:
    """
    Modified Laplacian - Uses absolute values instead of squares.
    More robust to noise than standard Laplacian.
    """
    h, w = img.shape
    ml_sum = 0.0

    for y in range(1, h - 1):
        for x in range(1, w - 1):
            ml = abs(2 * img[y, x] - img[y, x - 1] - img[y, x + 1]) + \
                 abs(2 * img[y, x] - img[y - 1, x] - img[y + 1, x])
            ml_sum += ml

    return ml_sum / ((h - 2) * (w - 2))


def frequency_domain_sharpness(img: np.ndarray) -> float:
    """
    Frequency domain analysis - High frequencies indicate sharpness.
    Note: This cannot be JIT compiled with numba due to FFT.
    """
    # Apply FFT
    f_transform = np.fft.fft2(img)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = np.abs(f_shift)

    h, w = img.shape
    center_y, center_x = h // 2, w // 2

    # Calculate total energy
    total_energy = np.sum(magnitude_spectrum)

    # Calculate high-frequency energy (outer region)
    # Create a mask for high frequencies
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)

    # Consider frequencies beyond 30% of the radius as high frequency
    radius_threshold = min(center_x, center_y) * 0.3
    high_freq_mask = dist_from_center > radius_threshold

    high_freq_energy = np.sum(magnitude_spectrum[high_freq_mask])

    # Return ratio of high frequency to total energy
    return high_freq_energy / total_energy if total_energy > 0 else 0.0


@njit
def diagonal_laplacian(img: np.ndarray) -> float:
    """
    Diagonal Laplacian - Includes diagonal neighbors.
    More comprehensive edge detection.
    """
    h, w = img.shape
    laplacian_sum = 0.0

    for y in range(1, h - 1):
        for x in range(1, w - 1):
            # 8-connected Laplacian
            lap = (img[y - 1, x - 1] + img[y - 1, x] + img[y - 1, x + 1] +
                   img[y, x - 1] - 8 * img[y, x] + img[y, x + 1] +
                   img[y + 1, x - 1] + img[y + 1, x] + img[y + 1, x + 1])
            laplacian_sum += abs(lap)

    return laplacian_sum / ((h - 2) * (w - 2))


@njit
def compute_sharpness_with_mask_numba(img: np.ndarray, mask: np.ndarray) -> float:
    """
    只在mask指定的区域内计算清晰度
    """
    h, w = img.shape
    F = 0
    count = 0

    for x in range(1, w - 1, 2):
        for y in range(1, h - 1, 2):
            # 只在mask内计算
            if mask[y, x] == 0:
                continue

            # 确保邻域也在mask内
            if (mask[y - 1, x] == 0 or mask[y + 1, x] == 0 or
                    mask[y, x - 1] == 0 or mask[y, x + 1] == 0):
                continue

            # 计算清晰度（原始算法）
            pave = (img[y, x] + img[y, x + 1] + img[y, x - 1] +
                    img[y + 1, x] + img[y - 1, x]) / 5

            g1st = (abs(img[y, x + 1] - pave) +
                    abs(img[y + 1, x] - pave) +
                    abs(img[y + 1, x + 1] - pave)) ** 2

            g2nd = (abs(img[y, x + 2] - pave) +
                    abs(img[y + 2, x] - pave) +
                    abs(img[y + 2, x + 2] - pave)) ** 2

            F += g1st * g2nd
            count += 1

    if count > 0:
        F = F * 0.1 / count
    else:
        F = 0

    return F


import pywt


def iris_wavelet_focus_measure(image, wavelet='db8', levels=4):
    """
    Multi-scale wavelet focus assessment for iris images
    Provides localized frequency analysis ideal for circular patterns
    """
    focus_measures = []

    for level in range(1, levels + 1):
        coeffs = pywt.wavedec2(image, wavelet, level=level)

        # Calculate focus measure for this scale
        if len(coeffs) > 1:
            detail_energy = 0
            approx_energy = np.sum(coeffs[0] ** 2)

            # Sum energy from all detail coefficients (LH, HL, HH)
            for detail in coeffs[1:]:
                if isinstance(detail, tuple):
                    detail_energy += sum(np.sum(d ** 2) for d in detail)

            focus_measure = detail_energy / (approx_energy + detail_energy + 1e-10)
            focus_measures.append(focus_measure)

    # Weight finer scales more heavily
    weights = np.exp(-np.arange(len(focus_measures)))
    return np.average(focus_measures, weights=weights)



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


# def detect_pupil_contour(img: np.ndarray, debug: bool = False) -> Optional[Tuple[int, int, int]]:
#     """
#     使用轮廓检测瞳孔
#     """
#     # 预处理
#     processed = preprocess_image(img)
#
#     # 多种阈值方法
#     thresholds = []
#
#     # # 1. 自适应阈值
#     # adaptive_thresh = cv2.adaptiveThreshold(
#     #     processed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
#     # )
#     # thresholds.append(adaptive_thresh)
#     #
#     # # 2. Otsu阈值
#     # _, otsu_thresh = cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
#     # thresholds.append(otsu_thresh)
#
#     # 3. 固定阈值（多个值）
#     for thresh_val in [30, 40, 50, 60]:
#         _, fixed_thresh = cv2.threshold(processed, thresh_val, 255, cv2.THRESH_BINARY_INV)
#         thresholds.append(fixed_thresh)
#
#     best_circle = None
#     best_score = 0
#
#     for thresh in thresholds:
#         # 形态学操作
#         kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
#         cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
#         cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
#
#         # 查找轮廓
#         contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#         for contour in contours:
#             # 轮廓质量评估
#             area = cv2.contourArea(contour)
#             if area < 55000:  # 太小的轮廓跳过
#                 continue
#
#             # 拟合圆
#             (x, y), radius = cv2.minEnclosingCircle(contour)
#             x, y, radius = int(x), int(y), int(radius)
#
#             if radius < 150 or radius > 400:  # 半径范围检查
#                 continue
#
#             # 计算轮廓质量得分
#             contour_score = evaluate_contour_quality(contour, (x, y), radius)
#
#             # 计算灰度对比度得分
#             contrast_score = evaluate_circle_quality(processed, (x, y, radius))
#
#             # 综合得分
#             total_score = contour_score * 0.6 + contrast_score * 0.4
#
#             if total_score > best_score:
#                 best_score = total_score
#                 best_circle = (x, y, radius)
#
#     # if debug and best_circle is not None:
#     #     debug_img = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
#     #     x, y, r = best_circle
#     #     cv2.circle(debug_img, (x, y), r, (0, 255, 0), 2)
#     #     cv2.circle(debug_img, (x, y), 2, (0, 0, 255), 3)
#     #     plt.figure(figsize=(8, 6))
#     #     plt.imshow(debug_img)
#     #     plt.title(f"Contour Detection Result (Score: {best_score:.2f})")
#     #     plt.axis('off')
#     #     plt.show()
#
#     return best_circle

def detect_pupil_contour(img: np.ndarray) -> Optional[Tuple[int, int, int]]:
    """使用轮廓检测瞳孔"""
    processed = preprocess_image(img)

    r_threshold = 150

    # 多种阈值方法
    thresholds = []

    # 固定阈值（多个值）

    # for thresh_val in [30, 40, 50, 60]:
    thresh_val = 30

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

        # 早期剔除：先按面积排序，只处理最大的N个轮廓
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        max_contours_to_process = 5  # 只处理前5个最大的轮廓

        for i, contour in enumerate(contours):
            if i >= max_contours_to_process:
                break

            area = cv2.contourArea(contour)
            if area < 55000:  # 太小的轮廓跳过
                continue

            # 拟合圆

            (cx, cy), radius = cv2.minEnclosingCircle(contour)
            cx, cy, radius = int(cx), int(cy), int(radius)


            if radius < r_threshold or radius > 2.5 * r_threshold:  # 半径范围检查
                continue

            # 计算轮廓质量得分
            contour_score = evaluate_contour_quality(contour, (cx, cy), radius)

            # 计算灰度对比度得分
            # contrast_score = evaluate_circle_quality(img, (cx, cy, radius))

            # 综合得分
            # total_score = contour_score * 0.6 + contrast_score * 0.4

            total_score = contour_score
            if total_score > best_score:
                best_score = total_score
                best_circle = (cx, cy, radius)

    return best_circle


# def robust_pupil_detection(img: np.ndarray, debug: bool = False) -> Tuple[Optional[np.ndarray], float]:
#     """
#     鲁棒的瞳孔检测主函数
#     """
#     if len(img.shape) == 3:
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#     # 使用轮廓检测
#     detected_circle = detect_pupil_contour(img, debug=debug)
#
#     if detected_circle is None:
#         return None, 0.0
#
#     # 提取瞳孔区域并计算清晰度
#     x, y, radius = detected_circle
#     print(f"瞳孔半径：{radius}像素")
#     print(f"瞳孔中心位置:（{x} , {y}）")
#
#     # 扩展区域用于清晰度计算
#     expand_factor = 1.3
#     expanded_radius = int(radius * expand_factor)
#
#     # 边界检查
#     h, w = img.shape
#     x1 = max(0, x - expanded_radius)
#     y1 = max(0, y - expanded_radius)
#     x2 = min(w, x + expanded_radius)
#     y2 = min(h, y + expanded_radius)
#
#     processed = preprocess_image(img)
#
#     crop = img[y1:y2, x1:x2]
#
#     if crop.size == 0:
#         return None, 0.0
#
#     # 计算清晰度
#     sharpness = compute_image_sharpness_numba(crop)
#
#     # 在裁剪图像上标记瞳孔
#     crop_display = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
#     center_x_crop = x - x1
#     center_y_crop = y - y1
#     cv2.circle(crop_display, (center_x_crop, center_y_crop), radius, (0, 255, 0), 2)
#     cv2.circle(crop_display, (center_x_crop, center_y_crop), 2, (0, 0, 255), 3)
#
#
#     '''
#     ###################################
#     '''
#
#     # # 1. 计算外圆半径和边界框
#     # expand_factor = 1.5
#     # outer_radius = int(radius * expand_factor)
#     #
#     # # 获取图像的高度和宽度
#     # h, w = img.shape[:2]
#     #
#     # # 计算包含外圆的正方形边界框 (Region of Interest, ROI)
#     # # 确保边界不会超出图像范围
#     # x1 = max(0, x - outer_radius)
#     # y1 = max(0, y - outer_radius)
#     # x2 = min(w, x + outer_radius)
#     # y2 = min(h, y + outer_radius)
#     #
#     # # 从原图中裁剪出这个矩形ROI
#     # roi = img[y1:y2, x1:x2]
#     #
#     # # 2. 创建掩码 (Mask)
#     # # 创建一个和ROI一样大的全黑掩码
#     # mask = np.zeros(roi.shape[:2], dtype="uint8")
#     #
#     # # 计算圆心在ROI内的相对坐标
#     # center_x_roi = x - x1
#     # center_y_roi = y - y1
#     #
#     # # 在掩码上画一个白色的实心大圆（外圆）
#     # cv2.circle(mask, (center_x_roi, center_y_roi), outer_radius, 255, -1)
#     #
#     # # 在掩码上画一个黑色的实心小圆（内圆），从而形成一个白色的圆环
#     # cv2.circle(mask, (center_x_roi, center_y_roi), radius, 0, -1)
#     #
#     # # 3. 应用掩码，得到最终的圆环图像
#     # # 只有当掩码的像素是白色(255)时，才保留ROI中对应像素的值
#     # ring_crop = cv2.bitwise_and(roi, roi, mask=mask)
#     #
#     # if ring_crop.size == 0:
#     #     return None, 0.0
#     #
#     # # 计算清晰度
#     # sharpness = compute_image_sharpness_numba(ring_crop)
#     #
#     #
#     #
#     # # 在裁剪图像上标记瞳孔
#     # crop_display = cv2.cvtColor(ring_crop, cv2.COLOR_GRAY2BGR)
#     # center_x_crop = x - x1
#     # center_y_crop = y - y1
#     # cv2.circle(crop_display, (center_x_crop, center_y_crop), radius, (0, 255, 0), 2)
#     # cv2.circle(crop_display, (center_x_crop, center_y_crop), 2, (0, 0, 255), 3)
#
#     # sharpness = compute_image_sharpness_numba(img)
#     #
#     # # 在裁剪图像上标记瞳孔
#     # crop_display = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
#     # center_x_crop = x - x1
#     # center_y_crop = y - y1
#     # cv2.circle(crop_display, (center_x_crop, center_y_crop), radius, (0, 255, 0), 2)
#     # cv2.circle(crop_display, (center_x_crop, center_y_crop), 2, (0, 0, 255), 3)
#
#     return crop_display, sharpness

def robust_pupil_detection(img: np.ndarray, debug: bool = False) -> Tuple[Optional[np.ndarray], float]:
    """
    改进的瞳孔检测主函数，使用优化的ROI选择
    """
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 使用轮廓检测
    detected_circle = detect_pupil_contour(img)

    if detected_circle is None:
        return None, 0.0

    # ========== 选择ROI提取方案 ==========

    # 方案一：瞳孔中心以下区域
    # roi, roi_info = extract_lower_pupil_roi(img, detected_circle)

    # 方案二：瞳孔下方虹膜区域
    roi, roi_info = extract_lower_iris_roi(img, detected_circle)

    # 方案三：扇形区域
    # roi, roi_info = extract_sector_roi(img, detected_circle, start_angle=30, end_angle=150)

    if roi is None or roi.size == 0:
        return None, 0.0

    # 计算清晰度（使用优化后的ROI）
    sharpness = compute_image_sharpness_numba(roi)

    # 扇形区域清晰度
    # sharpness = compute_sharpness_with_mask_numba(img, roi_info["full_mask"])
    # 如果需要调试，显示ROI区域
    if debug:
        display_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # 绘制瞳孔
        x, y, radius = detected_circle
        cv2.circle(display_img, (x, y), radius, (0, 255, 0), 2)
        cv2.circle(display_img, (x, y), 2, (0, 0, 255), 3)

        # 绘制ROI区域
        if 'x1' in roi_info:  # 矩形ROI
            cv2.rectangle(display_img,
                          (roi_info['x1'], roi_info['y1']),
                          (roi_info['x2'], roi_info['y2']),
                          (255, 0, 0), 2)
        elif 'full_mask' in roi_info:  # 扇形ROI
            contours, _ = cv2.findContours(roi_info['full_mask'],
                                           cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(display_img, contours, -1, (255, 0, 0), 2)

        # 显示
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(img, cmap='gray')
        plt.title("Original")

        plt.subplot(1, 3, 2)
        plt.imshow(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB))
        plt.title("Detection & ROI")

        plt.subplot(1, 3, 3)
        plt.imshow(roi, cmap='gray')
        plt.title(f"ROI (Sharpness: {sharpness:.2f})")

        plt.tight_layout()
        plt.show()

    return detected_circle, sharpness


def process_single_image(image_path: str, debug: bool = True) -> Tuple[float, bool]:
    """
    处理单张图像
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"无法读取图像: {image_path}")
        return 0.0, False

    crop_result, sharpness = robust_pupil_detection(img ,debug)

    if crop_result is not None:
        print(f"检测成功! 清晰度: {sharpness:.2f}")

        # if debug:
        #     plt.figure(figsize=(10, 5))
        #     plt.subplot(1, 2, 1)
        #     plt.imshow(img, cmap='gray')
        #     plt.title("Original image")
        #     plt.axis('off')
        #
        #     plt.subplot(1, 2, 2)
        #     # 检查crop_result是图像还是圆圈信息
        #     if isinstance(crop_result, tuple):
        #         # 如果是圆圈信息(x, y, radius)，在原图上绘制
        #         display_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        #         if len(crop_result) == 3:  # (x, y, radius)
        #             x, y, radius = crop_result
        #             cv2.circle(display_img, (x, y), radius, (0, 255, 0), 2)
        #             cv2.circle(display_img, (x, y), 2, (0, 0, 255), 3)
        #         plt.imshow(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB))
        #     else:
        #         # 如果是图像数组
        #         if crop_result.dtype != np.uint8:
        #             crop_result = crop_result.astype(np.uint8)
        #
        #         if len(crop_result.shape) == 2:
        #             plt.imshow(crop_result, cmap='gray')
        #         else:
        #             plt.imshow(cv2.cvtColor(crop_result, cv2.COLOR_BGR2RGB))
        #
        #     plt.title(f"Result (Sharpness: {sharpness:.2f})")
        #     plt.axis('off')
        #     plt.tight_layout()
        #     plt.show()

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

    image_paths = glob.glob("yjx/*.png", recursive=True)
    # image_paths = glob.glob("CASIA-IrisV4(JPG)/CASIA-Iris-Twins/**/*.jpg", recursive=True)
    # image_paths = [f"{i}.bmp" for i in range(1, 21)]
    sharpness_values = batch_process_images(image_paths, debug=False)

    pass

