import cv2
import numpy as np
import fitz  # PyMuPDF
import os
from PIL import Image


class SealExtractor:
    def __init__(self, output_dir='output', seal_size=(256, 256)):
        """
        初始化公章提取器
        :param output_dir: 输出目录
        :param seal_size: 输出公章图片固定大小 (宽, 高)
        """
        self.output_dir = output_dir
        self.images_dir = os.path.join(output_dir, 'pages')
        self.seals_dir = os.path.join(output_dir, 'seals')
        self.seal_size = seal_size  # 统一印章输出大小

        # 创建输出目录
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.seals_dir, exist_ok=True)

    # ---------------- PDF 转图片 ----------------
    def pdf_to_images(self, pdf_path, zoom=2.0):
        print(f"正在转换PDF: {pdf_path}")
        pdf_document = fitz.open(pdf_path)
        mat = fitz.Matrix(zoom, zoom)
        image_paths = []

        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            pix = page.get_pixmap(matrix=mat)
            image_path = os.path.join(self.images_dir, f'page_{page_num + 1}.png')
            pix.save(image_path)
            image_paths.append(image_path)
            print(f"已保存第 {page_num + 1} 页: {image_path}")

        pdf_document.close()
        return image_paths

    # ---------------- 检测红色区域 ----------------
    def detect_red_regions(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 43, 46])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([156, 43, 46])
        upper_red2 = np.array([180, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        return mask

    # ---------------- 霍夫圆检测 ----------------
    def detect_circles_hough(self, mask):
        circles = cv2.HoughCircles(
            mask,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=50,
            param1=50,
            param2=30,
            minRadius=30,
            maxRadius=200
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
        cv2.circle(edge_mask, (x, y), radius, 255, max(2, int(radius * 0.1)))
        cv2.circle(edge_mask, (x, y), int(radius * 0.85), 0, -1)
        edge_pixels = cv2.bitwise_and(mask, edge_mask)
        edge_ratio = np.sum(edge_pixels > 0) / np.sum(edge_mask > 0)

        center_mask = np.zeros_like(mask)
        cv2.circle(center_mask, (x, y), int(radius * 0.3), 255, -1)
        center_pixels = cv2.bitwise_and(mask, center_mask)
        center_ratio = np.sum(center_pixels > 0) / np.sum(center_mask > 0)

        ring_mask = np.zeros_like(mask)
        cv2.circle(ring_mask, (x, y), int(radius * 0.85), 255, -1)
        cv2.circle(ring_mask, (x, y), int(radius * 0.35), 0, -1)
        ring_pixels = cv2.bitwise_and(mask, ring_mask)
        ring_ratio = np.sum(ring_pixels > 0) / np.sum(ring_mask > 0)

        total_ratio = np.sum(roi > 0) / np.sum(roi_mask > 0)
        score = 0

        if edge_ratio > 0.3:
            score += 30 * min(edge_ratio / 0.5, 1.0)
        if 0.05 < center_ratio < 0.6:
            score += 25
        elif center_ratio > 0:
            score += 15
        if ring_ratio > 0.15:
            score += 30 * min(ring_ratio / 0.35, 1.0)
        if 0.15 < total_ratio < 0.5:
            score += 15
        elif total_ratio > 0.1:
            score += 10

        is_seal = score >= 95
        return is_seal, score

    # ---------------- 检测五角星中心 ----------------
    def detect_star_center(self, mask, center, radius):
        """
        尝试检测五角星中心位置（返回坐标或None）
        """
        x, y = center
        roi_mask = np.zeros_like(mask)
        cv2.circle(roi_mask, (x, y), int(radius * 0.4), 255, -1)
        roi = cv2.bitwise_and(mask, roi_mask)

        contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        star_candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 100 or area > 0.2 * np.pi * radius ** 2:
                continue

            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
            if 8 <= len(approx) <= 15:  # 五角星的尖角比较多
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    star_candidates.append((cx, cy, area))

        if len(star_candidates) > 0:
            # 取面积最大的作为星中心
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
            if area < 2000:
                continue
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)
            if radius < 30 or radius > 200:
                continue
            is_duplicate = False
            for seal in seals:
                dist = np.sqrt((seal['center'][0] - center[0]) ** 2 + (seal['center'][1] - center[1]) ** 2)
                if dist < radius * 0.5:
                    is_duplicate = True
                    break
            if is_duplicate:
                continue
            is_seal, score = self.check_seal_structure(image, mask, center, radius)
            if is_seal:
                seals.append({'center': center, 'radius': radius, 'score': score, 'method': 'contour'})

        seals.sort(key=lambda x: x['score'], reverse=True)
        return seals

    # ---------------- 提取公章（以五角星为中心） ----------------
    def extract_seal(self, image, mask, center, radius, padding=30):
        """
        以五角星中心为截图中心，若未检测到则使用圆心
        """
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

        # 固定输出大小
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
    def process_image(self, image_path, page_num):
        print(f"\n处理页面 {page_num}: {image_path}")
        image = cv2.imread(image_path)
        if image is None:
            print(f"无法读取图片: {image_path}")
            return 0
        mask = self.detect_red_regions(image)
        seals = self.find_seals(image, mask)
        print(f"在第 {page_num} 页检测到 {len(seals)} 个公章")

        for i, seal_info in enumerate(seals):
            seal_image = self.extract_seal(image, mask, seal_info['center'], seal_info['radius'])
            seal_path = os.path.join(self.seals_dir, f'seal_page{page_num}_no{i + 1}.png')
            cv2.imwrite(seal_path, seal_image)
            print(f"  保存公章 #{i + 1}: {seal_path}")
            print(f"    - 检测方法: {seal_info['method']}")
            print(f"    - 置信度分数: {seal_info['score']:.1f}")
            print(f"    - 半径: {seal_info['radius']}px")
        return len(seals)

    # ---------------- 处理PDF ----------------
    def process_pdf(self, pdf_path, zoom=2.0):
        print("=" * 60)
        print("开始处理PDF文件")
        print("=" * 60)
        image_paths = self.pdf_to_images(pdf_path, zoom)
        total_seals = 0
        for i, image_path in enumerate(image_paths):
            total_seals += self.process_image(image_path, i + 1)
        print("\n" + "=" * 60)
        print(f"处理完成! 共提取 {total_seals} 个公章")
        print(f"页面图片保存在: {self.images_dir}")
        print(f"公章图片保存在: {self.seals_dir}")
        print("=" * 60)


def main():
    pdf_path = "test.pdf"
    extractor = SealExtractor(output_dir='output', seal_size=(256, 256))
    extractor.process_pdf(pdf_path, zoom=2.5)


if __name__ == "__main__":
    main()
