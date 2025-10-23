import cv2
import numpy as np
import fitz  # PyMuPDF
import os
from PIL import Image
from itertools import combinations


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
        self.analysis_dir = os.path.join(output_dir, 'analysis')
        self.seal_size = seal_size

        # 创建输出目录
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.seals_dir, exist_ok=True)
        os.makedirs(self.analysis_dir, exist_ok=True)

        # 存储所有提取的印章信息
        self.all_seals = []

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

    # ---------------- 检测红色区域（改进版：扩大检测范围） ----------------
    def detect_red_regions(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # 扩大红色范围，降低饱和度和亮度的下限以检测暗淡的红色
        lower_red1 = np.array([0, 30, 30])  # 从43->30, 46->30
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([150, 30, 30])  # 从156->150, 43->30, 46->30
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)

        # 形态学处理，使用更大的核来连接断裂的公章
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        return mask

    # ---------------- 霍夫圆检测（改进版：降低阈值，扩大范围） ----------------
    def detect_circles_hough(self, mask):
        circles = cv2.HoughCircles(
            mask,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=40,  # 从50->40，允许更近的圆
            param1=50,
            param2=20,  # 从30->20，降低阈值提高检测率
            minRadius=25,  # 从30->25，检测更小的公章
            maxRadius=250  # 从200->250，检测更大的公章
        )
        if circles is not None:
            circles = np.uint16(np.around(circles))
            return circles[0]
        return []

    # ---------------- 检查公章结构（改进版：放宽评分标准） ----------------
    def check_seal_structure(self, image, mask, center, radius):
        x, y = center
        roi_mask = np.zeros_like(mask)
        cv2.circle(roi_mask, (x, y), radius, 255, -1)
        roi = cv2.bitwise_and(mask, roi_mask)

        # 边缘环检测（公章外圈）
        edge_mask = np.zeros_like(mask)
        cv2.circle(edge_mask, (x, y), radius, 255, max(2, int(radius * 0.12)))
        cv2.circle(edge_mask, (x, y), int(radius * 0.80), 0, -1)  # 从0.85->0.80
        edge_pixels = cv2.bitwise_and(mask, edge_mask)
        edge_ratio = np.sum(edge_pixels > 0) / max(np.sum(edge_mask > 0), 1)

        # 中心区域检测（五角星区域）
        center_mask = np.zeros_like(mask)
        cv2.circle(center_mask, (x, y), int(radius * 0.35), 255, -1)  # 从0.3->0.35
        center_pixels = cv2.bitwise_and(mask, center_mask)
        center_ratio = np.sum(center_pixels > 0) / max(np.sum(center_mask > 0), 1)

        # 文字环检测（单位名称区域）
        ring_mask = np.zeros_like(mask)
        cv2.circle(ring_mask, (x, y), int(radius * 0.80), 255, -1)  # 从0.85->0.80
        cv2.circle(ring_mask, (x, y), int(radius * 0.40), 0, -1)  # 从0.35->0.40
        ring_pixels = cv2.bitwise_and(mask, ring_mask)
        ring_ratio = np.sum(ring_pixels > 0) / max(np.sum(ring_mask > 0), 1)

        # 总体红色像素比例
        total_ratio = np.sum(roi > 0) / max(np.sum(roi_mask > 0), 1)

        # 改进的评分系统（降低阈值，更宽容）
        score = 0

        # 边缘环评分（权重30）
        if edge_ratio > 0.2:  # 从0.3->0.2
            score += 30 * min(edge_ratio / 0.4, 1.0)  # 从0.5->0.4
        elif edge_ratio > 0.1:
            score += 15

        # 中心区域评分（权重25）
        if 0.03 < center_ratio < 0.7:  # 从0.05->0.03, 0.6->0.7
            score += 25
        elif center_ratio > 0:
            score += 18  # 从15->18

        # 文字环评分（权重30）
        if ring_ratio > 0.12:  # 从0.15->0.12
            score += 30 * min(ring_ratio / 0.3, 1.0)  # 从0.35->0.3
        elif ring_ratio > 0.08:
            score += 15

        # 总体比例评分（权重15）
        if 0.12 < total_ratio < 0.55:  # 从0.15->0.12, 0.5->0.55
            score += 15
        elif total_ratio > 0.08:  # 从0.1->0.08
            score += 10

        # 降低判定阈值
        is_seal = score >= 98  # 从95->60
        return is_seal, score

    # ---------------- 检测五角星中心（改进版：更鲁棒的检测） ----------------
    def detect_star_center(self, mask, center, radius):
        x, y = center
        roi_mask = np.zeros_like(mask)
        cv2.circle(roi_mask, (x, y), int(radius * 0.45), 255, -1)  # 从0.4->0.45
        roi = cv2.bitwise_and(mask, roi_mask)

        contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        star_candidates = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 80 or area > 0.25 * np.pi * radius ** 2:  # 从100->80, 0.2->0.25
                continue

            # 计算凸包
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)

            # 凸包面积比（五角星的凸包面积应该比轮廓面积大）
            if hull_area > 0:
                solidity = area / hull_area
                # 五角星的solidity通常在0.5-0.75之间
                if not (0.4 < solidity < 0.85):  # 放宽范围
                    continue

            # 使用多边形近似
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)  # 从0.03->0.02，更精确

            # 五角星近似后通常有8-15个顶点（考虑角度偏移）
            if 6 <= len(approx) <= 18:  # 从8-15扩大到6-18
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])

                    # 计算形状因子（越接近1越圆，五角星应该偏离）
                    perimeter = cv2.arcLength(cnt, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        # 五角星的circularity通常在0.4-0.8之间
                        if 0.3 < circularity < 0.9:  # 放宽范围
                            star_candidates.append((cx, cy, area, solidity))

        if len(star_candidates) > 0:
            # 选择面积最大且凸度合理的候选
            best = max(star_candidates, key=lambda t: t[2])
            return (best[0], best[1])
        return None

    # ---------------- 查找公章（改进版：增加多尺度检测） ----------------
    def find_seals(self, image, mask):
        seals = []

        # 霍夫圆检测
        circles = self.detect_circles_hough(mask)
        for circle in circles:
            x, y, r = circle
            center = (int(x), int(y))
            radius = int(r)
            is_seal, score = self.check_seal_structure(image, mask, center, radius)
            if is_seal:
                seals.append({'center': center, 'radius': radius, 'score': score, 'method': 'hough'})

        # 轮廓检测
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 1500:  # 从2000->1500，检测更小的公章
                continue

            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)

            if radius < 25 or radius > 250:  # 从30-200扩大到25-250
                continue

            # 检查是否与已检测到的公章重复
            is_duplicate = False
            for seal in seals:
                dist = np.sqrt((seal['center'][0] - center[0]) ** 2 +
                               (seal['center'][1] - center[1]) ** 2)
                if dist < max(radius, seal['radius']) * 0.3:  # 从0.5->0.3，更严格去重
                    is_duplicate = True
                    break

            if is_duplicate:
                continue

            is_seal, score = self.check_seal_structure(image, mask, center, radius)
            if is_seal:
                seals.append({'center': center, 'radius': radius, 'score': score, 'method': 'contour'})

        # 按分数排序并去除低分重复
        seals.sort(key=lambda x: x['score'], reverse=True)

        # 二次去重：移除距离太近且分数较低的
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

    # ---------------- 提取公章（以五角星为中心） ----------------
    def extract_seal(self, image, mask, center, radius, padding=40):  # 从30->40
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
            seal_filename = f'seal_page{page_num}_no{i + 1}.png'
            seal_path = os.path.join(self.seals_dir, seal_filename)
            cv2.imwrite(seal_path, seal_image)

            # 记录印章信息
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

            print(f"  保存公章 #{i + 1}: {seal_path}")
            print(f"    - 检测方法: {seal_info['method']}")
            print(f"    - 置信度分数: {seal_info['score']:.1f}")
            print(f"    - 半径: {seal_info['radius']}px")

        return len(seals)

    # ---------------- 计算两个印章的重叠度 ----------------
    def calculate_overlap(self, seal1_img, seal2_img):
        """
        计算两个印章叠加后的重叠程度
        :param seal1_img: 第一个印章图片
        :param seal2_img: 第二个印章图片
        :return: (重叠率, 重叠像素数, 可视化图片)
        """
        # 提取红色区域
        mask1 = self.detect_red_regions(seal1_img)
        mask2 = self.detect_red_regions(seal2_img)

        # 计算红色像素
        pixels1 = np.sum(mask1 > 0)
        pixels2 = np.sum(mask2 > 0)

        # 计算重叠区域（两个印章都有红色的地方）
        overlap_mask = cv2.bitwise_and(mask1, mask2)
        overlap_pixels = np.sum(overlap_mask > 0)

        # 计算重叠率（相对于较小的印章）
        min_pixels = min(pixels1, pixels2)
        if min_pixels == 0:
            overlap_ratio = 0
        else:
            overlap_ratio = overlap_pixels / min_pixels

        # 创建可视化图片
        vis_img = np.ones((self.seal_size[1], self.seal_size[0] * 3 + 40, 3), dtype=np.uint8) * 255

        # 放置第一个印章
        vis_img[:, 0:self.seal_size[0]] = seal1_img

        # 放置第二个印章
        vis_img[:, self.seal_size[0] + 20:self.seal_size[0] * 2 + 20] = seal2_img

        # 创建叠加效果图
        overlay = seal1_img.copy()
        # 将第二个印章的红色部分叠加上去（用不同颜色显示重叠）
        red_pixels2 = mask2 > 0
        overlay[red_pixels2] = [0, 0, 255]  # 第二个印章用蓝色显示

        # 重叠部分用绿色标记
        overlap_pixels_pos = overlap_mask > 0
        overlay[overlap_pixels_pos] = [0, 255, 0]  # 重叠部分用绿色

        vis_img[:, self.seal_size[0] * 2 + 40:] = overlay

        # 添加文字说明
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(vis_img, "Seal 1", (10, 30), font, 0.7, (0, 0, 0), 2)
        cv2.putText(vis_img, "Seal 2", (self.seal_size[0] + 30, 30), font, 0.7, (0, 0, 0), 2)
        cv2.putText(vis_img, "Overlay", (self.seal_size[0] * 2 + 50, 30), font, 0.7, (0, 0, 0), 2)

        # 添加重叠率信息
        overlap_text = f"Overlap: {overlap_ratio * 100:.1f}%"
        cv2.putText(vis_img, overlap_text, (self.seal_size[0] * 2 + 50, self.seal_size[1] - 20),
                    font, 0.6, (0, 0, 255) if overlap_ratio > 0.1 else (0, 255, 0), 2)

        return overlap_ratio, overlap_pixels, vis_img

    # ---------------- 分析所有印章的重叠情况 ----------------
    def analyze_seal_overlaps(self, overlap_threshold=0.1):
        """
        分析所有印章之间的重叠情况
        :param overlap_threshold: 重叠率阈值，超过此值认为是可疑重叠
        """
        print("\n" + "=" * 60)
        print("开始分析印章重叠情况")
        print("=" * 60)

        if len(self.all_seals) < 2:
            print("印章数量少于2个，无需进行重叠分析")
            return

        suspicious_pairs = []
        total_comparisons = 0

        # 两两比较所有印章
        for i, (seal1, seal2) in enumerate(combinations(self.all_seals, 2)):
            total_comparisons += 1

            print(f"\n比较 {seal1['filename']} 与 {seal2['filename']}", end='')

            # 计算重叠度
            overlap_ratio, overlap_pixels, vis_img = self.calculate_overlap(
                seal1['image'], seal2['image']
            )

            # 判断是否可疑
            if overlap_ratio > overlap_threshold:
                print(f" - 重叠率: {overlap_ratio * 100:.2f}% ⚠️ 可疑")

                # 只保存可疑的对比图片
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
                print(f" - 重叠率: {overlap_ratio * 100:.2f}% ✓")

        # 总结报告
        print("\n" + "=" * 60)
        print("分析完成汇总")
        print("=" * 60)
        print(f"总印章数: {len(self.all_seals)}")
        print(f"总比较次数: {total_comparisons}")
        print(f"可疑重叠对数: {len(suspicious_pairs)}")

        # 只有存在可疑印章对时才生成报告
        if len(suspicious_pairs) > 0:
            # 生成可疑印章报告文件
            report_path = os.path.join(self.analysis_dir, 'suspicious_seals_report.txt')
            report_file = open(report_path, 'w', encoding='utf-8')
            report_file.write("=" * 80 + "\n")
            report_file.write("可疑印章重叠检测报告\n")
            report_file.write("=" * 80 + "\n\n")
            report_file.write(f"检测时间: {self._get_current_time()}\n")
            report_file.write(f"重叠阈值: {overlap_threshold * 100}%\n")
            report_file.write(f"总印章数: {len(self.all_seals)}\n")
            report_file.write(f"总比较次数: {total_comparisons}\n")
            report_file.write(f"可疑重叠对数: {len(suspicious_pairs)}\n\n")
            report_file.write("=" * 80 + "\n")
            report_file.write("可疑印章对详情\n")
            report_file.write("=" * 80 + "\n\n")

            print("\n⚠️  发现可疑印章对:")

            for idx, pair in enumerate(suspicious_pairs, 1):
                print(f"\n  {idx}. {pair['seal1']['filename']} ↔ {pair['seal2']['filename']}")
                print(f"     重叠率: {pair['overlap_ratio'] * 100:.2f}%")
                print(f"     对比图: {pair['compare_filename']}")

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

            report_file.close()
            print(f"\n详细报告已保存至: {report_path}")
            print(f"可疑对比图片保存在: {self.analysis_dir}")
        else:
            print("\n✓ 未发现可疑重叠，所有印章均正常")
            print("  无需生成报告文件")

    def _get_current_time(self):
        """获取当前时间字符串"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ---------------- 处理PDF ----------------
    def process_pdf(self, pdf_path, zoom=2.0, analyze_overlaps=True, overlap_threshold=0.1):
        """
        处理PDF并分析印章重叠
        :param pdf_path: PDF文件路径
        :param zoom: 缩放比例
        :param analyze_overlaps: 是否进行重叠分析
        :param overlap_threshold: 重叠率阈值（0-1），默认0.1即10%
        """
        print("=" * 60)
        print("开始处理PDF文件")
        print("=" * 60)

        # 清空之前的记录
        self.all_seals = []

        image_paths = self.pdf_to_images(pdf_path, zoom)
        total_seals = 0
        for i, image_path in enumerate(image_paths):
            total_seals += self.process_image(image_path, i + 1)

        print("\n" + "=" * 60)
        print(f"提取完成! 共提取 {total_seals} 个公章")
        print(f"页面图片保存在: {self.images_dir}")
        print(f"公章图片保存在: {self.seals_dir}")
        print("=" * 60)

        # 进行重叠分析
        if analyze_overlaps and total_seals >= 2:
            self.analyze_seal_overlaps(overlap_threshold)


def main():
    pdf_path = "test2.pdf"
    extractor = SealExtractor(output_dir='output', seal_size=(1024, 1024))

    # 处理PDF并分析重叠，重叠率超过10%认为可疑
    extractor.process_pdf(pdf_path, zoom=2.5, analyze_overlaps=True, overlap_threshold=0.95)


if __name__ == "__main__":
    main()