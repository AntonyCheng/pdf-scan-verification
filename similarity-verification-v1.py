import cv2
import numpy as np
import fitz  # PyMuPDF
import os
from PIL import Image


class SealExtractor:
    def __init__(self, output_dir='output'):
        """
        初始化公章提取器
        :param output_dir: 输出目录
        """
        self.output_dir = output_dir
        self.images_dir = os.path.join(output_dir, 'pages')
        self.seals_dir = os.path.join(output_dir, 'seals')

        # 创建输出目录
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.seals_dir, exist_ok=True)

    def pdf_to_images(self, pdf_path, zoom=2.0):
        """
        将PDF转换为图片
        :param pdf_path: PDF文件路径
        :param zoom: 缩放比例,越大图片越清晰(默认2.0相当于约150 DPI)
        :return: 保存的图片路径列表
        """
        print(f"正在转换PDF: {pdf_path}")

        # 打开PDF文件
        pdf_document = fitz.open(pdf_path)

        # 设置缩放矩阵
        mat = fitz.Matrix(zoom, zoom)

        image_paths = []
        for page_num in range(len(pdf_document)):
            # 获取页面
            page = pdf_document[page_num]

            # 将页面渲染为图片
            pix = page.get_pixmap(matrix=mat)

            # 保存图片
            image_path = os.path.join(self.images_dir, f'page_{page_num + 1}.png')
            pix.save(image_path)

            image_paths.append(image_path)
            print(f"已保存第 {page_num + 1} 页: {image_path}")

        pdf_document.close()
        return image_paths

    def detect_red_regions(self, image):
        """
        检测图片中的红色区域(公章通常是红色的)
        :param image: OpenCV图片对象
        :return: 红色区域的掩码
        """
        # 转换到HSV色彩空间
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # 定义红色的HSV范围(红色在HSV中分为两个区间)
        # 调整范围以更好地捕获公章的红色
        lower_red1 = np.array([0, 43, 46])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([156, 43, 46])
        upper_red2 = np.array([180, 255, 255])

        # 创建红色掩码
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)

        # 形态学操作,去除噪声
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        return mask

    def detect_circles_hough(self, mask):
        """
        使用霍夫圆变换检测圆形
        :param mask: 二值化掩码
        :return: 检测到的圆形列表
        """
        # 使用霍夫圆变换检测圆形
        circles = cv2.HoughCircles(
            mask,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=50,  # 圆心之间的最小距离
            param1=50,
            param2=30,
            minRadius=30,  # 最小半径(像素)
            maxRadius=200  # 最大半径(像素)
        )

        if circles is not None:
            circles = np.uint16(np.around(circles))
            return circles[0]
        return []

    def check_seal_structure(self, image, mask, center, radius):
        """
        检查是否符合公章结构特征:
        1. 圆形边框
        2. 中心区域有五角星(或其他图案)
        3. 环形文字区域

        :param image: 原始图片
        :param mask: 红色掩码
        :param center: 圆心坐标
        :param radius: 半径
        :return: (是否为公章, 置信度分数)
        """
        x, y = center

        # 提取圆形区域
        roi_mask = np.zeros_like(mask)
        cv2.circle(roi_mask, (x, y), radius, 255, -1)
        roi = cv2.bitwise_and(mask, roi_mask)

        # 1. 检查圆形边框 - 边缘应该有较多红色像素
        edge_mask = np.zeros_like(mask)
        cv2.circle(edge_mask, (x, y), radius, 255, max(2, int(radius * 0.1)))
        cv2.circle(edge_mask, (x, y), int(radius * 0.85), 0, -1)
        edge_pixels = cv2.bitwise_and(mask, edge_mask)
        edge_ratio = np.sum(edge_pixels > 0) / np.sum(edge_mask > 0)

        # 2. 检查中心区域 - 应该有五角星或其他图案
        center_mask = np.zeros_like(mask)
        cv2.circle(center_mask, (x, y), int(radius * 0.3), 255, -1)
        center_pixels = cv2.bitwise_and(mask, center_mask)
        center_ratio = np.sum(center_pixels > 0) / np.sum(center_mask > 0)

        # 3. 检查环形文字区域 - 中间环形应该有较多红色像素
        ring_mask = np.zeros_like(mask)
        cv2.circle(ring_mask, (x, y), int(radius * 0.85), 255, -1)
        cv2.circle(ring_mask, (x, y), int(radius * 0.35), 0, -1)
        ring_pixels = cv2.bitwise_and(mask, ring_mask)
        ring_ratio = np.sum(ring_pixels > 0) / np.sum(ring_mask > 0)

        # 4. 计算整体圆形区域的红色像素占比
        total_ratio = np.sum(roi > 0) / np.sum(roi_mask > 0)

        # 评分系统
        score = 0

        # 边框评分 (权重: 30%)
        if edge_ratio > 0.3:
            score += 30 * min(edge_ratio / 0.5, 1.0)

        # 中心评分 (权重: 25%) - 中心应该有内容但不能太满
        if 0.05 < center_ratio < 0.6:
            score += 25
        elif center_ratio > 0:
            score += 15

        # 环形文字评分 (权重: 30%)
        if ring_ratio > 0.15:
            score += 30 * min(ring_ratio / 0.35, 1.0)

        # 总体占比评分 (权重: 15%)
        if 0.15 < total_ratio < 0.5:
            score += 15
        elif total_ratio > 0.1:
            score += 10

        # 判断是否为公章 (阈值: 60分)
        is_seal = score >= 95

        return is_seal, score

    def find_seals(self, image, mask):
        """
        查找图片中的公章
        :param image: 原始图片
        :param mask: 红色区域掩码
        :return: 检测到的公章列表
        """
        seals = []

        # 方法1: 使用霍夫圆变换检测圆形
        circles = self.detect_circles_hough(mask)

        for circle in circles:
            x, y, r = circle
            center = (int(x), int(y))
            radius = int(r)

            # 检查是否符合公章结构
            is_seal, score = self.check_seal_structure(image, mask, center, radius)

            if is_seal:
                seals.append({
                    'center': center,
                    'radius': radius,
                    'score': score,
                    'method': 'hough'
                })

        # 方法2: 使用轮廓检测作为补充
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)

            # 过滤太小的区域
            if area < 2000:
                continue

            # 计算最小外接圆
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)

            # 过滤半径不合适的圆
            if radius < 30 or radius > 200:
                continue

            # 检查是否已经被霍夫变换检测到
            is_duplicate = False
            for seal in seals:
                dist = np.sqrt((seal['center'][0] - center[0]) ** 2 +
                               (seal['center'][1] - center[1]) ** 2)
                if dist < radius * 0.5:
                    is_duplicate = True
                    break

            if is_duplicate:
                continue

            # 检查是否符合公章结构
            is_seal, score = self.check_seal_structure(image, mask, center, radius)

            if is_seal:
                seals.append({
                    'center': center,
                    'radius': radius,
                    'score': score,
                    'method': 'contour'
                })

        # 按分数排序
        seals.sort(key=lambda x: x['score'], reverse=True)

        return seals

    def extract_seal(self, image, center, radius, padding=30):
        """
        提取单个公章
        :param image: 原始图片
        :param center: 圆心坐标
        :param radius: 半径
        :param padding: 额外的边距
        :return: 提取的公章图片
        """
        x, y = center
        r = radius + padding

        # 确保不超出图片边界
        x1 = max(0, x - r)
        y1 = max(0, y - r)
        x2 = min(image.shape[1], x + r)
        y2 = min(image.shape[0], y + r)

        # 裁剪公章区域
        seal = image[y1:y2, x1:x2]

        return seal

    def process_image(self, image_path, page_num):
        """
        处理单张图片,提取其中的公章
        :param image_path: 图片路径
        :param page_num: 页码
        :return: 提取的公章数量
        """
        print(f"\n处理页面 {page_num}: {image_path}")

        # 读取图片
        image = cv2.imread(image_path)
        if image is None:
            print(f"无法读取图片: {image_path}")
            return 0

        # 检测红色区域
        mask = self.detect_red_regions(image)

        # 查找公章
        seals = self.find_seals(image, mask)

        print(f"在第 {page_num} 页检测到 {len(seals)} 个公章")

        # 保存每个公章
        for i, seal_info in enumerate(seals):
            seal_image = self.extract_seal(
                image,
                seal_info['center'],
                seal_info['radius']
            )

            seal_path = os.path.join(
                self.seals_dir,
                f'seal_page{page_num}_no{i + 1}.png'
            )
            cv2.imwrite(seal_path, seal_image)
            print(f"  保存公章 #{i + 1}: {seal_path}")
            print(f"    - 检测方法: {seal_info['method']}")
            print(f"    - 置信度分数: {seal_info['score']:.1f}")
            print(f"    - 半径: {seal_info['radius']}px")

        return len(seals)

    def process_pdf(self, pdf_path, zoom=2.0):
        """
        处理整个PDF文件
        :param pdf_path: PDF文件路径
        :param zoom: 缩放比例(1.0=72dpi, 2.0=144dpi, 3.0=216dpi)
        """
        print("=" * 60)
        print("开始处理PDF文件")
        print("=" * 60)

        # 第一步: PDF转图片
        image_paths = self.pdf_to_images(pdf_path, zoom)

        # 第二步: 从每张图片中提取公章
        total_seals = 0
        for i, image_path in enumerate(image_paths):
            seals_count = self.process_image(image_path, i + 1)
            total_seals += seals_count

        print("\n" + "=" * 60)
        print(f"处理完成! 共提取 {total_seals} 个公章")
        print(f"页面图片保存在: {self.images_dir}")
        print(f"公章图片保存在: {self.seals_dir}")
        print("=" * 60)


def main():
    # 使用示例
    pdf_path = "test.pdf"  # 替换为你的PDF文件路径

    # 创建提取器
    extractor = SealExtractor(output_dir='output')

    # 处理PDF (zoom=2.0 相当于144 DPI, zoom=3.0更清晰)
    extractor.process_pdf(pdf_path, zoom=2.5)


if __name__ == "__main__":
    main()