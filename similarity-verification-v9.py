import cv2
import numpy as np
import fitz  # PyMuPDF
import os
from itertools import combinations
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from pathlib import Path
from datetime import datetime
import threading


class SealExtractor:
    def __init__(self, output_dir='output', seal_size=(256, 256), log_callback=None):
        """
        初始化公章提取器
        :param output_dir: 输出目录
        :param seal_size: 输出公章图片固定大小 (宽, 高)
        :param log_callback: 日志回调函数
        """
        self.output_dir = output_dir
        self.images_dir = os.path.join(output_dir, 'pages')
        self.seals_dir = os.path.join(output_dir, 'seals')
        self.analysis_dir = os.path.join(output_dir, 'analysis')
        self.seal_size = seal_size
        self.log_callback = log_callback

        # 创建输出目录
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.seals_dir, exist_ok=True)
        os.makedirs(self.analysis_dir, exist_ok=True)

        # 存储所有提取的印章信息
        self.all_seals = []

    def log(self, message):
        """输出日志"""
        if self.log_callback:
            self.log_callback(message)
        else:
            print(message)

    # ---------------- PDF 转图片 ----------------
    def pdf_to_images(self, pdf_path, zoom=2.0):
        self.log(f"正在转换PDF: {pdf_path}")
        pdf_document = fitz.open(pdf_path)
        mat = fitz.Matrix(zoom, zoom)
        image_paths = []

        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            pix = page.get_pixmap(matrix=mat)
            image_path = os.path.join(self.images_dir, f'page_{page_num + 1}.png')
            pix.save(image_path)
            image_paths.append(image_path)
            self.log(f"已保存第 {page_num + 1} 页: {image_path}")

        pdf_document.close()
        return image_paths

    # ---------------- 检测红色区域 ----------------
    def detect_red_regions(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        lower_red1 = np.array([0, 30, 30])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([150, 30, 30])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        return mask

    # ---------------- 霍夫圆检测 ----------------
    def detect_circles_hough(self, mask):
        circles = cv2.HoughCircles(
            mask,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=40,
            param1=50,
            param2=20,
            minRadius=25,
            maxRadius=250
        )
        if circles is not None:
            circles = np.uint16(np.around(circles))
            return circles[0]
        return []

    # ---------------- 检查公章结构 ----------------
    def check_seal_structure(self, image, mask, center, radius):
        x, y = center
        roi_mask = np.zeros_like(mask)
        cv2.circle(roi_mask, (x, y), radius, 255, -1)
        roi = cv2.bitwise_and(mask, roi_mask)

        edge_mask = np.zeros_like(mask)
        cv2.circle(edge_mask, (x, y), radius, 255, max(2, int(radius * 0.12)))
        cv2.circle(edge_mask, (x, y), int(radius * 0.80), 0, -1)
        edge_pixels = cv2.bitwise_and(mask, edge_mask)
        edge_ratio = np.sum(edge_pixels > 0) / max(np.sum(edge_mask > 0), 1)

        center_mask = np.zeros_like(mask)
        cv2.circle(center_mask, (x, y), int(radius * 0.35), 255, -1)
        center_pixels = cv2.bitwise_and(mask, center_mask)
        center_ratio = np.sum(center_pixels > 0) / max(np.sum(center_mask > 0), 1)

        ring_mask = np.zeros_like(mask)
        cv2.circle(ring_mask, (x, y), int(radius * 0.80), 255, -1)
        cv2.circle(ring_mask, (x, y), int(radius * 0.40), 0, -1)
        ring_pixels = cv2.bitwise_and(mask, ring_mask)
        ring_ratio = np.sum(ring_pixels > 0) / max(np.sum(ring_mask > 0), 1)

        total_ratio = np.sum(roi > 0) / max(np.sum(roi_mask > 0), 1)

        score = 0

        if edge_ratio > 0.2:
            score += 30 * min(edge_ratio / 0.4, 1.0)
        elif edge_ratio > 0.1:
            score += 15

        if 0.03 < center_ratio < 0.7:
            score += 25
        elif center_ratio > 0:
            score += 18

        if ring_ratio > 0.12:
            score += 30 * min(ring_ratio / 0.3, 1.0)
        elif ring_ratio > 0.08:
            score += 15

        if 0.12 < total_ratio < 0.55:
            score += 15
        elif total_ratio > 0.08:
            score += 10

        is_seal = score >= 60
        return is_seal, score

    # ---------------- 检测五角星中心 ----------------
    def detect_star_center(self, mask, center, radius):
        x, y = center
        roi_mask = np.zeros_like(mask)
        cv2.circle(roi_mask, (x, y), int(radius * 0.45), 255, -1)
        roi = cv2.bitwise_and(mask, roi_mask)

        contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        star_candidates = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 80 or area > 0.25 * np.pi * radius ** 2:
                continue

            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)

            if hull_area > 0:
                solidity = area / hull_area
                if not (0.4 < solidity < 0.85):
                    continue

            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

            if 6 <= len(approx) <= 18:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])

                    perimeter = cv2.arcLength(cnt, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        if 0.3 < circularity < 0.9:
                            star_candidates.append((cx, cy, area, solidity))

        if len(star_candidates) > 0:
            best = max(star_candidates, key=lambda t: t[2])
            return (best[0], best[1])
        return None

    # ---------------- 查找公章 ----------------
    def find_seals(self, image, mask):
        seals = []

        circles = self.detect_circles_hough(mask)
        for circle in circles:
            x, y, r = circle
            center = (int(x), int(y))
            radius = int(r)
            is_seal, score = self.check_seal_structure(image, mask, center, radius)
            if is_seal:
                seals.append({'center': center, 'radius': radius, 'score': score, 'method': 'hough'})

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 1500:
                continue

            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)

            if radius < 25 or radius > 250:
                continue

            is_duplicate = False
            for seal in seals:
                dist = np.sqrt((seal['center'][0] - center[0]) ** 2 +
                               (seal['center'][1] - center[1]) ** 2)
                if dist < max(radius, seal['radius']) * 0.3:
                    is_duplicate = True
                    break

            if is_duplicate:
                continue

            is_seal, score = self.check_seal_structure(image, mask, center, radius)
            if is_seal:
                seals.append({'center': center, 'radius': radius, 'score': score, 'method': 'contour'})

        seals.sort(key=lambda x: x['score'], reverse=True)

        filtered_seals = []
        for seal in seals:
            is_close = False
            for existing in filtered_seals:
                dist = np.sqrt((seal['center'][0] - existing['center'][0]) ** 2 +
                               (seal['center'][1] - existing['center'][1]) ** 2)
                if dist < (seal['radius'] + existing['radius']) * 0.4:
                    is_close = True
                    break
            if not is_close:
                filtered_seals.append(seal)

        return filtered_seals

    # ---------------- 提取公章 ----------------
    def extract_seal(self, image, mask, center, radius, padding=40):
        star_center = self.detect_star_center(mask, center, radius)
        if star_center is not None:
            cx, cy = star_center
        else:
            cx, cy = center

        r = radius + padding
        x1 = max(0, cx - r)
        y1 = max(0, cy - r)
        x2 = min(image.shape[1], cx + r)
        y2 = min(image.shape[0], cy + r)
        seal = image[y1:y2, x1:x2]

        h, w = seal.shape[:2]
        target_w, target_h = self.seal_size
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        seal_resized = cv2.resize(seal, (new_w, new_h), interpolation=cv2.INTER_AREA)

        background = np.ones((target_h, target_w, 3), dtype=np.uint8) * 255
        x_offset = (target_w - new_w) // 2
        y_offset = (target_h - new_h) // 2
        background[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = seal_resized

        return background

    # ---------------- 单页处理 ----------------
    def process_image(self, image_path, page_num, min_radius=0):
        self.log(f"\n处理页面 {page_num}: {image_path}")
        image = cv2.imread(image_path)
        if image is None:
            self.log(f"无法读取图片: {image_path}")
            return 0

        mask = self.detect_red_regions(image)
        seals = self.find_seals(image, mask)
        self.log(f"在第 {page_num} 页检测到 {len(seals)} 个公章")

        saved_count = 0
        filtered_count = 0

        for i, seal_info in enumerate(seals):
            if seal_info['radius'] < min_radius:
                filtered_count += 1
                self.log(f"  公章 #{i + 1}: 半径 {seal_info['radius']}px < {min_radius}px, 已过滤")
                continue

            seal_image = self.extract_seal(image, mask, seal_info['center'], seal_info['radius'])
            seal_filename = f'seal_page{page_num}_no{i + 1}.png'
            seal_path = os.path.join(self.seals_dir, seal_filename)
            cv2.imwrite(seal_path, seal_image)

            self.all_seals.append({
                'filename': seal_filename,
                'path': seal_path,
                'page': page_num,
                'index': i + 1,
                'image': seal_image,
                'method': seal_info['method'],
                'score': seal_info['score'],
                'radius': seal_info['radius']
            })

            saved_count += 1
            self.log(f"  保存公章 #{i + 1}: {seal_path}")
            self.log(f"    - 检测方法: {seal_info['method']}")
            self.log(f"    - 置信度分数: {seal_info['score']:.1f}")
            self.log(f"    - 半径: {seal_info['radius']}px")

        if filtered_count > 0:
            self.log(f"  (共过滤 {filtered_count} 个小尺寸公章)")

        return saved_count

    # ---------------- 计算两个印章的重叠度 ----------------
    def calculate_overlap(self, seal1_img, seal2_img):
        mask1 = self.detect_red_regions(seal1_img)
        mask2 = self.detect_red_regions(seal2_img)

        pixels1 = np.sum(mask1 > 0)
        pixels2 = np.sum(mask2 > 0)

        overlap_mask = cv2.bitwise_and(mask1, mask2)
        overlap_pixels = np.sum(overlap_mask > 0)

        min_pixels = min(pixels1, pixels2)
        if min_pixels == 0:
            overlap_ratio = 0
        else:
            overlap_ratio = overlap_pixels / min_pixels

        vis_img = np.ones((self.seal_size[1], self.seal_size[0] * 3 + 40, 3), dtype=np.uint8) * 255

        vis_img[:, 0:self.seal_size[0]] = seal1_img
        vis_img[:, self.seal_size[0] + 20:self.seal_size[0] * 2 + 20] = seal2_img

        overlay = seal1_img.copy()
        red_pixels2 = mask2 > 0
        overlay[red_pixels2] = [0, 0, 255]

        overlap_pixels_pos = overlap_mask > 0
        overlay[overlap_pixels_pos] = [0, 255, 0]

        vis_img[:, self.seal_size[0] * 2 + 40:] = overlay

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(vis_img, "Seal 1", (10, 30), font, 0.7, (0, 0, 0), 2)
        cv2.putText(vis_img, "Seal 2", (self.seal_size[0] + 30, 30), font, 0.7, (0, 0, 0), 2)
        cv2.putText(vis_img, "Overlay", (self.seal_size[0] * 2 + 50, 30), font, 0.7, (0, 0, 0), 2)

        overlap_text = f"Overlap: {overlap_ratio * 100:.1f}%"
        cv2.putText(vis_img, overlap_text, (self.seal_size[0] * 2 + 50, self.seal_size[1] - 20),
                    font, 0.6, (0, 0, 255) if overlap_ratio > 0.1 else (0, 255, 0), 2)

        return overlap_ratio, overlap_pixels, vis_img

    # ---------------- 分析所有印章的重叠情况 ----------------
    def analyze_seal_overlaps(self, overlap_threshold=0.1):
        self.log("\n" + "=" * 60)
        self.log("开始分析印章重叠情况")
        self.log("=" * 60)

        if len(self.all_seals) < 2:
            self.log("印章数量少于2个，无需进行重叠分析")
            return

        suspicious_pairs = []
        total_comparisons = 0

        for i, (seal1, seal2) in enumerate(combinations(self.all_seals, 2)):
            total_comparisons += 1

            overlap_ratio, overlap_pixels, vis_img = self.calculate_overlap(
                seal1['image'], seal2['image']
            )

            if overlap_ratio > overlap_threshold:
                self.log(
                    f"\n比较 {seal1['filename']} 与 {seal2['filename']} - 重叠率: {overlap_ratio * 100:.2f}% ⚠️ 可疑")

                compare_filename = f'suspicious_{seal1["filename"][:-4]}_vs_{seal2["filename"][:-4]}.png'
                compare_path = os.path.join(self.analysis_dir, compare_filename)
                cv2.imwrite(compare_path, vis_img)

                suspicious_pairs.append({
                    'seal1': seal1,
                    'seal2': seal2,
                    'overlap_ratio': overlap_ratio,
                    'overlap_pixels': overlap_pixels,
                    'compare_image': compare_path,
                    'compare_filename': compare_filename
                })
            else:
                self.log(f"\n比较 {seal1['filename']} 与 {seal2['filename']} - 重叠率: {overlap_ratio * 100:.2f}% ✓")

        self.log("\n" + "=" * 60)
        self.log("分析完成汇总")
        self.log("=" * 60)
        self.log(f"总印章数: {len(self.all_seals)}")
        self.log(f"总比较次数: {total_comparisons}")
        self.log(f"可疑重叠对数: {len(suspicious_pairs)}")

        if len(suspicious_pairs) > 0:
            report_path = os.path.join(self.analysis_dir, 'suspicious_seals_report.txt')
            with open(report_path, 'w', encoding='utf-8') as report_file:
                report_file.write("=" * 80 + "\n")
                report_file.write("可疑印章重叠检测报告\n")
                report_file.write("=" * 80 + "\n\n")
                report_file.write(f"检测时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                report_file.write(f"重叠阈值: {overlap_threshold * 100}%\n")
                report_file.write(f"总印章数: {len(self.all_seals)}\n")
                report_file.write(f"总比较次数: {total_comparisons}\n")
                report_file.write(f"可疑重叠对数: {len(suspicious_pairs)}\n\n")
                report_file.write("=" * 80 + "\n")
                report_file.write("可疑印章对详情\n")
                report_file.write("=" * 80 + "\n\n")

                self.log("\n⚠️  发现可疑印章对:")

                for idx, pair in enumerate(suspicious_pairs, 1):
                    self.log(f"\n  {idx}. {pair['seal1']['filename']} ↔ {pair['seal2']['filename']}")
                    self.log(f"     重叠率: {pair['overlap_ratio'] * 100:.2f}%")
                    self.log(f"     对比图: {pair['compare_filename']}")

                    report_file.write(f"【可疑对 #{idx}】\n")
                    report_file.write(f"  印章1: {pair['seal1']['filename']}\n")
                    report_file.write(f"    - 来源页面: 第 {pair['seal1']['page']} 页\n")
                    report_file.write(f"    - 页面编号: 第 {pair['seal1']['index']} 个印章\n")
                    report_file.write(f"    - 检测方法: {pair['seal1']['method']}\n")
                    report_file.write(f"    - 置信度分数: {pair['seal1']['score']:.1f}\n\n")

                    report_file.write(f"  印章2: {pair['seal2']['filename']}\n")
                    report_file.write(f"    - 来源页面: 第 {pair['seal2']['page']} 页\n")
                    report_file.write(f"    - 页面编号: 第 {pair['seal2']['index']} 个印章\n")
                    report_file.write(f"    - 检测方法: {pair['seal2']['method']}\n")
                    report_file.write(f"    - 置信度分数: {pair['seal2']['score']:.1f}\n\n")

                    report_file.write(f"  重叠分析:\n")
                    report_file.write(f"    - 重叠率: {pair['overlap_ratio'] * 100:.2f}%\n")
                    report_file.write(f"    - 重叠像素数: {pair['overlap_pixels']}\n")
                    report_file.write(f"    - 对比图文件: {pair['compare_filename']}\n")
                    report_file.write(
                        f"    - 风险等级: {'高' if pair['overlap_ratio'] > 0.3 else '中' if pair['overlap_ratio'] > 0.15 else '低'}\n")
                    report_file.write("\n" + "-" * 80 + "\n\n")

                report_file.write("=" * 80 + "\n")
                report_file.write("分析结论\n")
                report_file.write("=" * 80 + "\n")
                report_file.write("以上印章对存在明显重叠，建议进一步人工核查。\n")
                report_file.write("重叠可能的原因:\n")
                report_file.write("1. 印章为后期PS伪造（从同一原始印章复制）\n")
                report_file.write("2. 同一印章在不同位置重复盖章\n")
                report_file.write("3. 扫描/拍照导致的图像重影\n")
                report_file.write("\n建议: 对比原件进行验证\n")

            self.log(f"\n详细报告已保存至: {report_path}")
            self.log(f"可疑对比图片保存在: {self.analysis_dir}")
        else:
            self.log("\n✓ 未发现可疑重叠，所有印章均正常")
            self.log("  无需生成报告文件")

    # ---------------- 处理PDF ----------------
    def process_pdf(self, pdf_path, zoom=2.0, analyze_overlaps=True, overlap_threshold=0.1, min_radius=0):
        self.log("=" * 60)
        self.log("开始处理PDF文件")
        self.log("=" * 60)
        if min_radius > 0:
            self.log(f"半径过滤: 仅保存半径 >= {min_radius}px 的公章")
            self.log("=" * 60)

        self.all_seals = []

        image_paths = self.pdf_to_images(pdf_path, zoom)
        total_seals = 0
        for i, image_path in enumerate(image_paths):
            total_seals += self.process_image(image_path, i + 1, min_radius)

        self.log("\n" + "=" * 60)
        self.log(f"提取完成! 共提取 {total_seals} 个公章")
        self.log(f"页面图片保存在: {self.images_dir}")
        self.log(f"公章图片保存在: {self.seals_dir}")
        self.log("=" * 60)

        if analyze_overlaps and total_seals >= 2:
            self.analyze_seal_overlaps(overlap_threshold)


class SealDetectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("公章相似度工具")
        self.root.geometry("900x650")

        self.pdf_path = tk.StringVar()
        self.overlap_threshold = tk.DoubleVar(value=0.95)
        self.min_radius = tk.IntVar(value=100)
        self.output_path = tk.StringVar()

        self.create_widgets()

    def create_widgets(self):
        # 标题
        title_label = tk.Label(self.root, text="公章相似度工具",
                               font=("Arial", 16, "bold"))
        title_label.pack(pady=10)

        # 1. PDF文件选择
        pdf_frame = tk.Frame(self.root)
        pdf_frame.pack(pady=5, padx=20, fill=tk.X)

        tk.Label(pdf_frame, text="上传PDF文件:", width=14, anchor='w').pack(side=tk.LEFT)
        tk.Entry(pdf_frame, textvariable=self.pdf_path, width=100).pack(side=tk.LEFT, padx=5)
        tk.Button(pdf_frame, text="浏览...", command=self.select_pdf).pack(side=tk.LEFT)

        # 2. 重叠率阈值
        threshold_frame = tk.Frame(self.root)
        threshold_frame.pack(pady=5, padx=20, fill=tk.X)

        tk.Label(threshold_frame, text="重叠率阈值:", width=14, anchor='w').pack(side=tk.LEFT)
        self.threshold_scale = tk.Scale(threshold_frame, from_=0.1, to=0.999,
                                        orient=tk.HORIZONTAL, resolution=0.001,
                                        variable=self.overlap_threshold,
                                        command=self.update_threshold_label,
                                        length=700)
        self.threshold_scale.pack(side=tk.LEFT, padx=5)
        self.threshold_label = tk.Label(threshold_frame, text="95.0%", width=8)
        self.threshold_label.pack(side=tk.LEFT)

        # 3. 最小半径过滤
        radius_frame = tk.Frame(self.root)
        radius_frame.pack(pady=5, padx=20, fill=tk.X)

        tk.Label(radius_frame, text="最小半径过滤:", width=14, anchor='w').pack(side=tk.LEFT)
        self.radius_scale = tk.Scale(radius_frame, from_=10, to=1000,
                                     orient=tk.HORIZONTAL, resolution=10,
                                     variable=self.min_radius,
                                     command=self.update_radius_label,
                                     length=700)
        self.radius_scale.pack(side=tk.LEFT, padx=5)
        self.radius_label = tk.Label(radius_frame, text="100px", width=8)
        self.radius_label.pack(side=tk.LEFT)

        # 4. 分析结果保存路径
        output_frame = tk.Frame(self.root)
        output_frame.pack(pady=5, padx=20, fill=tk.X)

        tk.Label(output_frame, text="分析结果保存路径:", width=14, anchor='w').pack(side=tk.LEFT)
        tk.Entry(output_frame, textvariable=self.output_path, width=100).pack(side=tk.LEFT, padx=5)
        tk.Button(output_frame, text="浏览...", command=self.select_output).pack(side=tk.LEFT)

        # 开始分析按钮
        self.start_button = tk.Button(self.root, text="开始分析",
                                      command=self.start_analysis,
                                      bg="#4CAF50", fg="white",
                                      font=("Arial", 12, "bold"),
                                      padx=20, pady=5)
        self.start_button.pack(pady=15)

        # 进度显示
        self.progress_label = tk.Label(self.root, text="准备就绪", fg="blue")
        self.progress_label.pack(pady=5)

        # 日志显示区域
        log_frame = tk.Frame(self.root)
        log_frame.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)

        tk.Label(log_frame, text="处理日志:", anchor='w').pack(anchor='w')

        self.log_text = scrolledtext.ScrolledText(log_frame, width=100, height=20,
                                                  wrap=tk.WORD, font=("Consolas", 9))
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def update_threshold_label(self, value):
        """更新重叠率标签"""
        self.threshold_label.config(text=f"{float(value) * 100:.1f}%")

    def update_radius_label(self, value):
        """更新半径标签"""
        self.radius_label.config(text=f"{int(float(value))}px")

    def select_pdf(self):
        """选择PDF文件"""
        filename = filedialog.askopenfilename(
            title="选择PDF文件",
            filetypes=[("PDF文件", "*.pdf"), ("所有文件", "*.*")]
        )
        if filename:
            self.pdf_path.set(filename)

    def select_output(self):
        """选择输出目录"""
        dirname = filedialog.askdirectory(title="选择输出目录")
        if dirname:
            self.output_path.set(dirname)

    def log(self, message):
        """在日志区域显示消息"""
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.root.update()

    def start_analysis(self):
        """开始分析"""
        pdf_file = self.pdf_path.get()
        output_path = self.output_path.get()

        if not pdf_file:
            messagebox.showwarning("警告", "请选择PDF文件！")
            return

        if not os.path.exists(pdf_file):
            messagebox.showerror("错误", "PDF文件不存在！")
            return

        if not output_path:
            messagebox.showwarning("警告", "请选择结果存储路径！")
            return

        if not os.path.exists(output_path):
            messagebox.showerror("错误", "结果存储路径不存在！")
            return

            # 在新线程中执行，避免界面卡死
        thread = threading.Thread(target=self.process_pdf)
        thread.daemon = True
        thread.start()

    def process_pdf(self):
        """处理PDF文件"""
        self.start_button.config(state=tk.DISABLED)
        self.log_text.delete(1.0, tk.END)

        pdf_file = self.pdf_path.get()
        output_dir = self.output_path.get()
        overlap_threshold = self.overlap_threshold.get()
        min_radius = self.min_radius.get()

        self.progress_label.config(text=f"正在处理: {os.path.basename(pdf_file)}", fg="blue")

        try:
            # 创建输出目录
            Path(output_dir).mkdir(parents=True, exist_ok=True)

            # 创建提取器
            extractor = SealExtractor(
                output_dir=output_dir,
                seal_size=(1024, 1024),
                log_callback=self.log
            )

            # 处理PDF
            extractor.process_pdf(
                pdf_path=pdf_file,
                zoom=2.5,
                analyze_overlaps=True,
                overlap_threshold=overlap_threshold,
                min_radius=min_radius
            )

            # 完成
            self.progress_label.config(
                text=f"完成！共提取 {len(extractor.all_seals)} 个公章",
                fg="green"
            )
            self.start_button.config(state=tk.NORMAL)

            messagebox.showinfo("完成",
                                f"公章检测分析已完成！\n\n"
                                f"共提取公章: {len(extractor.all_seals)} 个\n"
                                f"结果已保存到: {output_dir}")

        except Exception as e:
            error_msg = f"处理失败: {str(e)}"
            self.log(f"\n❌ {error_msg}")
            self.progress_label.config(text="处理失败", fg="red")
            self.start_button.config(state=tk.NORMAL)
            messagebox.showerror("错误", error_msg)


def main():
    root = tk.Tk()
    app = SealDetectorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
