import fitz  # PyMuPDF
import os
from typing import Dict, List, Tuple


class PDFSealDetector:
    def __init__(self):
        self.suspicious_pages = []

    def analyze_pdf(self, pdf_path: str) -> Dict:
        """
        分析PDF文件，检查每页是否仅包含一张图片
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
                'page_details': [],
                'is_suspicious': False,
                'summary': {
                    'pages_with_text': [],
                    'pages_with_multiple_images': [],
                    'pages_with_no_images': [],
                    'pages_with_drawings': [],
                    'clean_scan_pages': []
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

                # 获取图片详细信息
                image_info = []
                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    pix = fitz.Pixmap(pdf_document, xref)
                    image_info.append({
                        'index': img_index,
                        'width': pix.width,
                        'height': pix.height,
                        'colorspace': pix.colorspace.name,
                        'alpha': pix.alpha
                    })
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
                    'text_length': len(text_content),
                    'images': image_info,
                    'status': page_status,
                    'is_clean_scan': page_status == 'clean_scan'
                }

                results['page_details'].append(page_info)

                # 更新汇总信息
                if has_text:
                    results['summary']['pages_with_text'].append(page_num + 1)
                if image_count > 1:
                    results['summary']['pages_with_multiple_images'].append(page_num + 1)
                if image_count == 0:
                    results['summary']['pages_with_no_images'].append(page_num + 1)
                if has_drawings:
                    results['summary']['pages_with_drawings'].append(page_num + 1)
                if page_status == 'clean_scan':
                    results['summary']['clean_scan_pages'].append(page_num + 1)

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
        report.append("=" * 60)
        report.append("PDF文件完整性分析报告")
        report.append("=" * 60)
        report.append(f"文件路径: {analysis_result['file_path']}")
        report.append(f"总页数: {analysis_result['total_pages']}")
        report.append("-" * 60)

        if 'error' in analysis_result:
            report.append(f"错误: {analysis_result['error']}")
            return '\n'.join(report)

        # 汇总信息
        summary = analysis_result['summary']
        clean_pages = len(summary['clean_scan_pages'])
        total_pages = analysis_result['total_pages']

        if analysis_result['is_suspicious']:
            report.append(f"⚠️  检测结果: 发现异常！")
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
        report.append("\n" + "=" * 60)
        report.append("页面详细分析:")
        report.append("=" * 60)

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

            # 如果有异常，显示详细信息
            if status != 'clean_scan':
                if page_detail['has_text']:
                    report.append(f"   - 文字长度: {page_detail['text_length']} 字符")

                if page_detail['image_count'] > 1:
                    report.append("   - 图片详情:")
                    for img in page_detail['images']:
                        report.append(f"     • 图片{img['index'] + 1}: "
                                      f"{img['width']}x{img['height']}, "
                                      f"颜色: {img['colorspace']}, "
                                      f"透明: {'是' if img['alpha'] else '否'}")

        # 结论和建议
        report.append("\n" + "=" * 60)
        report.append("分析结论:")
        report.append("=" * 60)

        if not analysis_result['is_suspicious']:
            report.append("✓ 该PDF文档符合纯扫描件标准")
            report.append("✓ 每页仅包含一张扫描图片，无额外元素")
        else:
            suspicious_ratio = len(analysis_result['suspicious_pages']) / total_pages
            if suspicious_ratio > 0.5:
                report.append("⚠️  该文档大部分页面存在异常，可能不是纯扫描件")
            else:
                report.append("⚠️  该文档部分页面存在异常")

            # 根据具体问题给出建议
            if summary['pages_with_multiple_images']:
                report.append("⚠️  发现多图层页面，可能存在后期添加的公章或签名")
            if summary['pages_with_text']:
                report.append("⚠️  发现可编辑文字，不是纯扫描件")
            if summary['pages_with_drawings']:
                report.append("⚠️  发现矢量图形，可能有后期添加的元素")

        return '\n'.join(report)

    def _get_status_description(self, status: str) -> str:
        """获取状态描述"""
        status_map = {
            'clean_scan': '正常扫描页',
            'no_image': '无图片',
            'multiple_images': '多张图片（可能伪造）',
            'contains_text': '包含文字元素',
            'contains_drawings': '包含矢量图形',
            'unknown': '未知状态'
        }
        return status_map.get(status, status)

    def quick_check(self, pdf_path: str) -> Tuple[bool, str]:
        """快速检查PDF是否为纯扫描件"""
        result = self.analyze_pdf(pdf_path)

        if 'error' in result:
            return False, f"检查失败: {result['error']}"

        if not result['is_suspicious']:
            return True, "✓ 纯扫描件，无异常"
        else:
            reasons = []
            summary = result['summary']

            if summary['pages_with_text']:
                reasons.append(f"包含文字({len(summary['pages_with_text'])}页)")
            if summary['pages_with_multiple_images']:
                reasons.append(f"多图层({len(summary['pages_with_multiple_images'])}页)")
            if summary['pages_with_no_images']:
                reasons.append(f"无图片({len(summary['pages_with_no_images'])}页)")
            if summary['pages_with_drawings']:
                reasons.append(f"包含矢量图({len(summary['pages_with_drawings'])}页)")

            return False, f"⚠️  异常: {', '.join(reasons)}"


# 使用示例
def main():
    # 创建检测器实例
    detector = PDFSealDetector()

    # 单个文件分析
    pdf_path = "终验合格证书-fake.pdf"  # 替换为你的PDF文件路径

    # 快速检查
    print("快速检查结果:")
    is_clean, message = detector.quick_check(pdf_path)
    print(f"{message}\n")

    # 详细分析
    print("详细分析:")
    result = detector.analyze_pdf(pdf_path)
    report = detector.generate_report(result)
    print(report)

    # 保存报告到文件
    with open("pdf_analysis_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    print("\n报告已保存到: pdf_analysis_report.txt")


# 批量检查示例
def batch_check_example():
    detector = PDFSealDetector()

    # 批量检查多个PDF文件
    pdf_files = ["file1.pdf", "file2.pdf", "file3.pdf"]

    print("批量检查结果:")
    print("-" * 60)

    for pdf_file in pdf_files:
        if os.path.exists(pdf_file):
            is_clean, message = detector.quick_check(pdf_file)
            print(f"{pdf_file}: {message}")

    print("-" * 60)


if __name__ == "__main__":
    main()
