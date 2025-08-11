import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import glob
import cpbd
from scipy.stats import spearmanr

# ===== 你原来的 robust_pupil_detection 修改版 =====
def robust_pupil_detection(img: np.ndarray, debug: bool = False) -> Tuple[Optional[np.ndarray], float]:
    from test import detect_pupil_contour, preprocess_image  # 如果这几个函数在同一文件，就直接调用
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    detected_circle = detect_pupil_contour(img, debug=debug)
    if detected_circle is None:
        return None, 0.0

    x, y, radius = detected_circle
    expand_factor = 1.3
    expanded_radius = int(radius * expand_factor)
    h, w = img.shape
    x1 = max(0, x - expanded_radius)
    y1 = max(0, y - expanded_radius)
    x2 = min(w, x + expanded_radius)
    y2 = min(h, y + expanded_radius)

    processed = preprocess_image(img)

    crop = img[y1:y2, x1:x2]

    if crop.size == 0:
        return None, 0.0

    return crop, 0.0  # 不在这里算清晰度了


# ===== 方法 1: CPBD =====
def cpbd_sharpness(img: np.ndarray) -> float:
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return float(cpbd.compute(img))


# ===== 方法 2: Polar 角向高频 =====
def polar_angular_highfreq(img: np.ndarray, polar_rows=60, polar_cols=360) -> float:
    h, w = img.shape
    center = (w // 2, h // 2)
    maxRadius = min(center[0], center[1], w - center[0] - 1, h - center[1] - 1)
    polar = cv2.warpPolar(img, (polar_cols, polar_rows), center, maxRadius, cv2.WARP_POLAR_LINEAR)
    ang_grad = np.diff(polar.astype(np.float32), axis=1)
    return float(np.mean(np.sum(ang_grad**2, axis=1)))


# ===== 方法 3: MLV =====
def mlv_sharpness(img: np.ndarray) -> float:
    imgf = img.astype(np.float32)
    p = np.pad(imgf, ((1,1),(1,1)), mode='edge')
    neighs = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            neighs.append(np.abs(imgf - p[1+dy:1+dy+imgf.shape[0], 1+dx:1+dx+imgf.shape[1]]))
    mlv = np.maximum.reduce(neighs)
    weight = mlv ** 1.5
    mean_mlv = np.mean(mlv)
    wstd = np.sqrt(np.mean(weight * (mlv - mean_mlv) ** 2))
    return float(wstd + 1e-12)


# ===== 方法 4: DCT 高频能量比 =====
def dct_high_freq_ratio(img: np.ndarray, hf_radius_ratio: float = 0.3) -> float:
    f = cv2.dct(img.astype(np.float32))
    mag = np.abs(f)
    h, w = mag.shape
    uu = np.arange(h).reshape(h, 1) / float(h)
    vv = np.arange(w).reshape(1, w) / float(w)
    radius = np.sqrt(uu ** 2 + vv ** 2)
    mask = radius > hf_radius_ratio
    hf = mag[mask].sum()
    total = mag.sum() + 1e-12
    return float(hf / total)


# ===== 方法 5: 多尺度 Tenengrad =====
def multiscale_tenengrad(img: np.ndarray, scales=(1, 2, 4)) -> float:
    total = 0.0
    for s in scales:
        if s == 1:
            small = img
        else:
            small = cv2.resize(img, (img.shape[1] // s, img.shape[0] // s), interpolation=cv2.INTER_LINEAR)
        gx = cv2.Sobel(small, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(small, cv2.CV_64F, 0, 1, ksize=3)
        e = np.mean(gx * gx + gy * gy)
        total += e
    return float(total)


import numpy as np

# ===== 方法列表 =====
sharpness_methods = {
    # "CPBD": cpbd_sharpness,
    # "polar" : polar_angular_highfreq,
    # "MLV": mlv_sharpness,
    # "DCT_HF": dct_high_freq_ratio,
    # "MultiScale_Tenengrad": multiscale_tenengrad,
}


# ===== 批量处理 =====
def batch_process_images(image_paths: List[str], debug=False):
    all_scores = {name: [] for name in sharpness_methods}
    for i, path in enumerate(image_paths, 1):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"无法读取图像: {path}")
            continue
        roi, _ = robust_pupil_detection(img, debug=debug)
        if roi is None:
            for name in sharpness_methods:
                all_scores[name].append(np.nan)
            continue
        for name, func in sharpness_methods.items():
            try:
                score = func(roi)
            except Exception as e:
                print(f"{name} 在 {path} 计算失败: {e}")
                score = np.nan
            all_scores[name].append(score)

    # 画曲线
    plt.figure(figsize=(12, 6))
    for name, scores in all_scores.items():
        plt.plot(range(1, len(scores) + 1), scores, marker='o', label=name)
    plt.xlabel("Image Index")
    plt.ylabel("Sharpness")
    plt.title("Sharpness Comparison")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    return all_scores


if __name__ == "__main__":
    image_paths = glob.glob("zch/*.png")
    scores = batch_process_images(image_paths, debug=False)
