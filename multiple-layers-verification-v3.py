import fitz  # PyMuPDF
import os
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import tempfile
from pathlib import Path


class PDFSealDetector:
    def __init__(self):
        self.suspicious_pages = []
        # 印章检测参数
        self.seal_config = {
            'red_threshold': {
                'lower_hsv': np.array([0, 50, 50]),  # 红色下界
                'upper_hsv': np.array([10, 255, 255]),  # 红色上界
                'lower_hsv2': np.array([170, 50, 50]),  # 红色下界（另一范围）
                'upper_hsv2': np.array([180, 255, 255])  # 红色上界（另一范围）
            },
            'min_red_pixels': 100,  # 最小红色像素数
            'min_circularity': 0.5,  # 最小圆形度
            'min_seal_size': 30,  # 最小印章尺寸（像素）
            'max_seal_size': 800,  # 最大印章尺寸（像素）
            'min_red_ratio_in_contour': 0.3  # 轮廓内红色像素的最小比例
        }

    def analyze_pdf(self, pdf_path: str) -> Dict:
        """
        分析PDF文件，检查每页是否仅包含一张图片，如果有多个图层则检测每个图层是否包含印章
        返回分析结果
        """
        try:
            # 打开PDF文件
            pdf_document = fitz.open(pdf_path)
            total_pages = len(pdf_document)

            results = {
                'file_path': pdf_path,
                'total_pages': total_pages,
                'suspicious_pages': [],
                'pages_with_seals': [],  # 包含印章的页面
                'page_details': [],
                'is_suspicious': False,
                'summary': {
                    'pages_with_text': [],
                    'pages_with_multiple_images': [],
                    'pages_with_no_images': [],
                    'pages_with_drawings': [],
                    'clean_scan_pages': [],
                    'pages_with_possible_seals': [],  # 可能包含印章的页面
                    'suspicious_seal_layers': []  # 可疑的印章图层
                }
            }

            # 遍历每一页
            for page_num in range(total_pages):
                page = pdf_document[page_num]

                # 获取页面中的所有图片
                image_list = page.get_images()
                image_count = len(image_list)

                # 检查是否有文字
                text_content = page.get_text().strip()
                has_text = len(text_content) > 0

                # 检查是否有矢量图形（drawings）
                drawings = page.get_drawings()
                has_drawings = len(drawings) > 0

                # 分析每个图层
                image_info = []
                layer_seal_results = []
                page_has_seal = False
                suspicious_seal_found = False

                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    pix = fitz.Pixmap(pdf_document, xref)

                    # 将Pixmap转换为numpy数组进行分析
                    img_array = self._pixmap_to_numpy(pix)

                    # 检测当前图层中的印章
                    seal_result = self._detect_seal_in_layer(img_array, img_index)

                    img_data = {
                        'index': img_index,
                        'width': pix.width,
                        'height': pix.height,
                        'colorspace': pix.colorspace.name,
                        'alpha': pix.alpha,
                        'has_seal': seal_result['has_seal'],
                        'seal_info': seal_result if seal_result['has_seal'] else None,
                        'layer_type': 'transparent' if pix.alpha else 'opaque'
                    }

                    # 如果检测到印章
                    if seal_result['has_seal']:
                        page_has_seal = True
                        layer_seal_results.append({
                            'layer_index': img_index,
                            'seal_result': seal_result,
                            'is_transparent': pix.alpha > 0
                        })

                        # 如果是多图层页面中的印章，标记为可疑
                        if image_count > 1:
                            suspicious_seal_found = True
                            img_data['suspicious_seal'] = True

                    image_info.append(img_data)
                    pix = None  # 释放资源

                # 判断页面状态
                page_status = self._determine_page_status(
                    image_count, has_text, has_drawings
                )

                page_info = {
                    'page_number': page_num + 1,
                    'image_count': image_count,
                    'has_text': has_text,
                    'has_drawings': has_drawings,
                    'has_seal': page_has_seal,
                    'has_suspicious_seal': suspicious_seal_found,  # 多图层中的印章
                    'text_length': len(text_content),
                    'images': image_info,
                    'seal_detections': layer_seal_results,
                    'status': page_status,
                    'is_clean_scan': page_status == 'clean_scan'
                }

                results['page_details'].append(page_info)

                # 更新汇总信息
                if has_text:
                    results['summary']['pages_with_text'].append(page_num + 1)
                if image_count > 1:
                    results['summary']['pages_with_multiple_images'].append(page_num + 1)
                    if suspicious_seal_found:
                        results['summary']['suspicious_seal_layers'].append({
                            'page': page_num + 1,
                            'layers_with_seals': [r['layer_index'] for r in layer_seal_results]
                        })
                if image_count == 0:
                    results['summary']['pages_with_no_images'].append(page_num + 1)
                if has_drawings:
                    results['summary']['pages_with_drawings'].append(page_num + 1)
                if page_status == 'clean_scan':
                    results['summary']['clean_scan_pages'].append(page_num + 1)
                if page_has_seal:
                    results['summary']['pages_with_possible_seals'].append(page_num + 1)
                    results['pages_with_seals'].append(page_num + 1)

                # 判断是否可疑
                if page_status != 'clean_scan':
                    results['suspicious_pages'].append(page_num + 1)
                    results['is_suspicious'] = True

            pdf_document.close()
            return results

        except Exception as e:
            return {
                'error': str(e),
                'file_path': pdf_path
            }

    def _pixmap_to_numpy(self, pixmap: fitz.Pixmap) -> np.ndarray:
        """将PyMuPDF的Pixmap转换为numpy数组"""
        try:
            # 获取图像数据
            img_data = pixmap.samples
            img_array = np.frombuffer(img_data, dtype=np.uint8)

            # 根据颜色空间重塑数组
            if pixmap.colorspace is None:
                img_array = img_array.reshape(pixmap.height, pixmap.width)
                # 灰度图转BGR
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
            elif pixmap.n == 3:  # RGB
                img_array = img_array.reshape(pixmap.height, pixmap.width, 3)
                # RGB转BGR（OpenCV使用BGR）
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            elif pixmap.n == 4:  # RGBA
                img_array = img_array.reshape(pixmap.height, pixmap.width, 4)
                # RGBA转BGR
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
            else:
                # 尝试直接reshape
                channels = pixmap.n
                img_array = img_array.reshape(pixmap.height, pixmap.width, channels)
                if channels == 1:
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
                elif channels == 4:
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)

            return img_array

        except Exception as e:
            print(f"转换Pixmap时出错: {e}")
            # 返回空图像
            return np.zeros((pixmap.height, pixmap.width, 3), dtype=np.uint8)

    def _detect_seal_in_layer(self, image: np.ndarray, layer_index: int) -> Dict:
        """检测图层中是否包含印章（红色圆形）"""
        result = {
            'has_seal': False,
            'layer_index': layer_index,
            'seal_count': 0,
            'seals': [],
            'red_pixel_count': 0,
            'total_red_area': 0,
            'confidence': 0
        }

        try:
            if image is None or image.size == 0:
                return result

            # 转换到HSV颜色空间
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            # 创建红色掩码（红色在HSV中有两个范围）
            mask1 = cv2.inRange(hsv,
                                self.seal_config['red_threshold']['lower_hsv'],
                                self.seal_config['red_threshold']['upper_hsv'])
            mask2 = cv2.inRange(hsv,
                                self.seal_config['red_threshold']['lower_hsv2'],
                                self.seal_config['red_threshold']['upper_hsv2'])
            red_mask = cv2.bitwise_or(mask1, mask2)

            # 统计红色像素
            red_pixel_count = cv2.countNonZero(red_mask)
            result['red_pixel_count'] = red_pixel_count

            # 如果红色像素太少，直接返回
            if red_pixel_count < self.seal_config['min_red_pixels']:
                return result

            # 形态学操作，连接断开的部分
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
            red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)

            # 查找轮廓
            contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # 分析每个轮廓
            for contour in contours:
                area = cv2.contourArea(contour)

                # 获取边界框
                x, y, w, h = cv2.boundingRect(contour)

                # 过滤太小或太大的轮廓
                max_dimension = max(w, h)
                min_dimension = min(w, h)

                if (min_dimension < self.seal_config['min_seal_size'] or
                        max_dimension > self.seal_config['max_seal_size']):
                    continue

                # 计算圆形度
                perimeter = cv2.arcLength(contour, True)
                if perimeter == 0:
                    continue

                circularity = 4 * np.pi * area / (perimeter * perimeter)

                # 计算宽高比
                aspect_ratio = w / h if h > 0 else 0

                # 创建轮廓掩码，计算轮廓内红色像素的比例
                contour_mask = np.zeros(red_mask.shape, dtype=np.uint8)
                cv2.drawContours(contour_mask, [contour], -1, 255, -1)
                red_in_contour = cv2.bitwise_and(red_mask, contour_mask)
                red_pixels_in_contour = cv2.countNonZero(red_in_contour)
                contour_pixels = cv2.countNonZero(contour_mask)
                red_ratio_in_contour = red_pixels_in_contour / contour_pixels if contour_pixels > 0 else 0

                # 综合判断是否为印章
                is_circular = circularity >= self.seal_config['min_circularity']
                is_square_like = 0.7 <= aspect_ratio <= 1.3
                has_enough_red = red_ratio_in_contour >= self.seal_config['min_red_ratio_in_contour']

                if is_circular and is_square_like and has_enough_red:
                    # 计算置信度
                    confidence = (
                            circularity * 0.4 +  # 圆形度权重40%
                            (1 - abs(1 - aspect_ratio)) * 0.3 +  # 宽高比权重30%
                            red_ratio_in_contour * 0.3  # 红色比例权重30%
                    )

                    seal_info = {
                        'position': (x, y),
                        'size': (w, h),
                        'area': area,
                        'circularity': circularity,
                        'aspect_ratio': aspect_ratio,
                        'red_ratio': red_ratio_in_contour,
                        'confidence': confidence,
                        'center': (x + w // 2, y + h // 2),
                        'radius': max_dimension // 2
                    }
                    result['seals'].append(seal_info)
                    result['total_red_area'] += area

            # 判断是否检测到印章
            if result['seals']:
                result['has_seal'] = True
                result['seal_count'] = len(result['seals'])
                result['confidence'] = max(s['confidence'] for s in result['seals'])

            return result

        except Exception as e:
            result['error'] = str(e)
            return result

    def _determine_page_status(self, image_count: int, has_text: bool,
                               has_drawings: bool) -> str:
        """判断页面状态"""
        if image_count == 1 and not has_text and not has_drawings:
            return 'clean_scan'
        elif image_count == 0:
            return 'no_image'
        elif image_count > 1:
            return 'multiple_images'
        elif has_text:
            return 'contains_text'
        elif has_drawings:
            return 'contains_drawings'
        else:
            return 'unknown'

    def generate_report(self, analysis_result: Dict) -> str:
        """生成分析报告"""
        report = []
        report.append("=" * 70)
        report.append("PDF文件印章检测与完整性分析报告")
        report.append("=" * 70)
        report.append(f"文件路径: {analysis_result['file_path']}")
        report.append(f"总页数: {analysis_result['total_pages']}")
        report.append("-" * 70)

        if 'error' in analysis_result:
            report.append(f"错误: {analysis_result['error']}")
            return '\n'.join(report)

        # 汇总信息
        summary = analysis_result['summary']
        clean_pages = len(summary['clean_scan_pages'])
        total_pages = analysis_result['total_pages']

        # 印章检测汇总
        if summary['suspicious_seal_layers']:
            report.append("\n🚨 【印章检测警告】")
            report.append("=" * 70)
            report.append("检测到多图层页面中存在印章，可能为后期添加：")
            for item in summary['suspicious_seal_layers']:
                layers_str = ', '.join(map(str, item['layers_with_seals']))
                report.append(f"  ⚠️  第 {item['page']} 页 - 图层 {layers_str} 中发现印章")
            report.append("-" * 70)

        if analysis_result['is_suspicious']:
            report.append(f"\n⚠️  检测结果: 发现异常！")
            report.append(f"✓  正常扫描页: {clean_pages}/{total_pages} 页")

            if summary['pages_with_text']:
                report.append(f"⚠️  包含文字的页面: {', '.join(map(str, summary['pages_with_text']))}")
            if summary['pages_with_multiple_images']:
                report.append(f"⚠️  包含多张图片的页面: {', '.join(map(str, summary['pages_with_multiple_images']))}")
            if summary['pages_with_no_images']:
                report.append(f"⚠️  没有图片的页面: {', '.join(map(str, summary['pages_with_no_images']))}")
            if summary['pages_with_drawings']:
                report.append(f"⚠️  包含矢量图形的页面: {', '.join(map(str, summary['pages_with_drawings']))}")
        else:
            report.append("✓ 检测结果: 文档正常")
            report.append("✓ 所有页面均为纯扫描图片")

        # 详细页面信息
        report.append("\n" + "=" * 70)
        report.append("页面详细分析:")
        report.append("=" * 70)

        for page_detail in analysis_result['page_details']:
            page_num = page_detail['page_number']
            status = page_detail['status']

            # 根据状态使用不同的标记
            if status == 'clean_scan':
                marker = "✓"
                status_text = "正常扫描页"
            else:
                marker = "⚠️"
                status_text = self._get_status_description(status)

            report.append(f"\n{marker} 第 {page_num} 页: {status_text}")
            report.append(f"   - 图片数量: {page_detail['image_count']}")
            report.append(f"   - 包含文字: {'是' if page_detail['has_text'] else '否'}")
            report.append(f"   - 包含矢量图: {'是' if page_detail['has_drawings'] else '否'}")

            # 如果有多个图层，详细分析每个图层
            if page_detail['image_count'] > 1:
                report.append("   📋 图层分析:")
                for img in page_detail['images']:
                    layer_marker = "     •"
                    if img.get('has_seal'):
                        layer_marker = "     🔴"
                        report.append(f"{layer_marker} 图层 {img['index']}: "
                                      f"{img['width']}x{img['height']} ({img['layer_type']})")
                        report.append(f"        ⚠️  检测到印章！")
                        if img.get('seal_info'):
                            for i, seal in enumerate(img['seal_info']['seals'], 1):
                                report.append(f"        - 印章{i}: 置信度 {seal['confidence']:.1%}, "
                                              f"圆形度 {seal['circularity']:.2f}, "
                                              f"位置 {seal['position']}, "
                                              f"尺寸 {seal['size']}")
                    else:
                        report.append(f"{layer_marker} 图层 {img['index']}: "
                                      f"{img['width']}x{img['height']} ({img['layer_type']})")

            # 如果只有一个图层但检测到印章（正常情况）
            elif page_detail.get('has_seal') and page_detail['image_count'] == 1:
                report.append("   ✓ 检测到印章（单图层，正常扫描）")
                for detection in page_detail.get('seal_detections', []):
                    if detection['seal_result'].get('seals'):
                        for seal in detection['seal_result']['seals']:
                            report.append(f"      - 置信度 {seal['confidence']:.1%}, "
                                          f"圆形度 {seal['circularity']:.2f}")

        # 结论和建议
        report.append("\n" + "=" * 70)
        report.append("分析结论与建议:")
        report.append("=" * 70)

        if not analysis_result['is_suspicious']:
            report.append("✅ 该PDF文档符合纯扫描件标准")
            report.append("✅ 每页仅包含一张扫描图片，无额外元素")
            if analysis_result.get('pages_with_seals'):
                report.append("✅ 检测到的印章均在单一图层中，符合正常扫描特征")
        else:
            suspicious_ratio = len(analysis_result['suspicious_pages']) / total_pages

            # 重点分析印章问题
            if summary['suspicious_seal_layers']:
                report.append("\n🚨 【重要发现】")
                report.append("检测到多图层页面中存在印章，这通常表明：")
                report.append("  1. 印章可能是后期通过图像编辑软件添加的")
                report.append("  2. 文档可能经过篡改或伪造")
                report.append("  3. 建议进行人工审核或使用其他方式验证印章真实性")

            if summary['pages_with_multiple_images'] and not summary['suspicious_seal_layers']:
                report.append("\n⚠️  发现多图层页面但未检测到明显印章")
                report.append("  可能存在其他形式的文档修改")

            if summary['pages_with_text']:
                report.append("\n⚠️  发现可编辑文字，不是纯扫描件")

            if summary['pages_with_drawings']:
                report.append("\n⚠️  发现矢量图形，可能有后期添加的元素")

            report.append("\n建议采取以下措施：")
            report.append("  • 与原始文档进行对比")
            report.append("  • 检查文档来源的可靠性")
            report.append("  • 必要时进行专业的文档鉴定")

        return '\n'.join(report)

    def _get_status_description(self, status: str) -> str:
        """获取状态描述"""
        status_map = {
            'clean_scan': '正常扫描页',
            'no_image': '无图片',
            'multiple_images': '多张图片（可能存在图层叠加）',
            'contains_text': '包含文字元素',
            'contains_drawings': '包含矢量图形',
            'unknown': '未知状态'
        }
        return status_map.get(status, status)

    def quick_check(self, pdf_path: str) -> Tuple[bool, str]:
        """快速检查PDF是否为纯扫描件，重点关注印章"""
        result = self.analyze_pdf(pdf_path)

        if 'error' in result:
            return False, f"检查失败: {result['error']}"

        summary = result['summary']

        # 首先检查是否有可疑印章
        if summary['suspicious_seal_layers']:
            seal_pages = len(summary['suspicious_seal_layers'])
            return False, f"🚨 发现可疑印章！{seal_pages}个页面的多图层中检测到印章，可能为伪造"

        if not result['is_suspicious']:
            if result.get('pages_with_seals'):
                return True, "✓ 纯扫描件，无异常（含正常扫描印章）"
            else:
                return True, "✓ 纯扫描件，无异常"
        else:
            reasons = []

            if summary['pages_with_text']:
                reasons.append(f"包含文字({len(summary['pages_with_text'])}页)")
            if summary['pages_with_multiple_images']:
                reasons.append(f"多图层({len(summary['pages_with_multiple_images'])}页)")
            if summary['pages_with_no_images']:
                reasons.append(f"无图片({len(summary['pages_with_no_images'])}页)")
            if summary['pages_with_drawings']:
                reasons.append(f"包含矢量图({len(summary['pages_with_drawings'])}页)")

            return False, f"⚠️  异常: {', '.join(reasons)}"

    def visualize_seal_detection(self, pdf_path: str, page_num: int, output_dir: str = "seal_analysis"):
        """可视化印章检测结果"""
        os.makedirs(output_dir, exist_ok=True)

        try:
            pdf_document = fitz.open(pdf_path)

            if page_num > len(pdf_document):
                print(f"页码超出范围")
                return

            page = pdf_document[page_num - 1]
            image_list = page.get_images()

            for img_index, img in enumerate(image_list):
                xref = img[0]
                pix = fitz.Pixmap(pdf_document, xref)

                # 转换为numpy数组
                img_array = self._pixmap_to_numpy(pix)

                # 检测印章
                seal_result = self._detect_seal_in_layer(img_array, img_index)

                if seal_result['has_seal']:
                    # 在图像上标记检测到的印章
                    img_marked = img_array.copy()
                    for seal in seal_result['seals']:
                        x, y = seal['position']
                        w, h = seal['size']
                        # 画矩形框
                        cv2.rectangle(img_marked, (x, y), (x + w, y + h), (0, 0, 255), 2)
                        # 画圆形
                        center = seal['center']
                        radius = seal['radius']
                        cv2.circle(img_marked, center, radius, (0, 255, 0), 2)
                        # 添加置信度文本
                        text = f"Seal: {seal['confidence']:.1%}"
                        cv2.putText(img_marked, text, (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                    # 保存标记后的图像
                    output_path = os.path.join(output_dir,
                                               f"page{page_num}_layer{img_index}_marked.png")
                    cv2.imwrite(output_path, img_marked)
                    print(f"已保存标记图像: {output_path}")

                # 保存原始图层
                original_path = os.path.join(output_dir,
                                             f"page{page_num}_layer{img_index}_original.png")
                cv2.imwrite(original_path, img_array)

                pix = None

            pdf_document.close()

        except Exception as e:
            print(f"可视化时出错: {e}")


# 使用示例
def main():
    # 创建检测器实例
    detector = PDFSealDetector()

    # 单个文件分析
    pdf_path = "终验合格证书-fake.pdf"  # 替换为你的PDF文件路径

    # 快速检查
    print("=" * 70)
    print("快速检查结果:")
    print("=" * 70)
    is_clean, message = detector.quick_check(pdf_path)
    print(f"{message}\n")

    # 详细分析
    print("\n" + "=" * 70)
    print("开始详细分析...")
    print("=" * 70)
    result = detector.analyze_pdf(pdf_path)
    report = detector.generate_report(result)
    print(report)

    # 如果检测到可疑印章，进行可视化
    if result['summary'].get('suspicious_seal_layers'):
        print("\n" + "=" * 70)
        print("检测到可疑印章，正在生成可视化分析...")
        print("=" * 70)
        for item in result['summary']['suspicious_seal_layers']:
            page_num = item['page']
            print(f"正在分析第 {page_num} 页...")
            detector.visualize_seal_detection(pdf_path, page_num)
        print("可视化分析完成，请查看 seal_analysis 文件夹")

    # 保存报告到文件
    with open("pdf_seal_analysis_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n详细报告已保存到: pdf_seal_analysis_report.txt")


# 批量检查示例
def batch_check_example():
    detector = PDFSealDetector()

    # 批量检查多个PDF文件
    pdf_files = ["file1.pdf", "file2.pdf", "file3.pdf"]

    print("批量检查结果:")
    print("=" * 70)

    suspicious_files = []

    for pdf_file in pdf_files:
        if os.path.exists(pdf_file):
            is_clean, message = detector.quick_check(pdf_file)
            print(f"{pdf_file}: {message}")
            if not is_clean and "印章" in message:
                suspicious_files.append(pdf_file)

    if suspicious_files:
        print("\n" + "=" * 70)
        print("⚠️  需要重点关注的文件（可能包含伪造印章）:")
        for file in suspicious_files:
            print(f"  - {file}")

    print("=" * 70)


if __name__ == "__main__":
    main()
