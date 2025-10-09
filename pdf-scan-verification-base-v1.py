import fitz  # PyMuPDF
import os
from typing import Dict, List, Tuple


class PDFSealDetector:
    def __init__(self):
        self.suspicious_pages = []

    def analyze_pdf(self, pdf_path: str) -> Dict:
        """
        分析PDF文件，检查每页的图片数量
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
                'is_suspicious': False
            }

            # 遍历每一页
            for page_num in range(total_pages):
                page = pdf_document[page_num]

                # 获取页面中的所有图片
                image_list = page.get_images()
                image_count = len(image_list)

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

                page_info = {
                    'page_number': page_num + 1,
                    'image_count': image_count,
                    'images': image_info
                }

                results['page_details'].append(page_info)

                # 判断是否可疑（超过1张图片）
                if image_count > 1:
                    results['suspicious_pages'].append(page_num + 1)
                    results['is_suspicious'] = True

            pdf_document.close()
            return results

        except Exception as e:
            return {
                'error': str(e),
                'file_path': pdf_path
            }

    def batch_analyze(self, pdf_files: List[str]) -> List[Dict]:
        """批量分析多个PDF文件"""
        results = []
        for pdf_file in pdf_files:
            if os.path.exists(pdf_file) and pdf_file.lower().endswith('.pdf'):
                result = self.analyze_pdf(pdf_file)
                results.append(result)
        return results

    def generate_report(self, analysis_result: Dict) -> str:
        """生成分析报告"""
        report = []
        report.append(f"PDF文件分析报告")
        report.append(f"文件路径: {analysis_result['file_path']}")
        report.append(f"总页数: {analysis_result['total_pages']}")
        report.append("-" * 50)

        if 'error' in analysis_result:
            report.append(f"错误: {analysis_result['error']}")
            return '\n'.join(report)

        if analysis_result['is_suspicious']:
            report.append(f"⚠️ 发现可疑页面！")
            report.append(f"可疑页面: {', '.join(map(str, analysis_result['suspicious_pages']))}")
        else:
            report.append("✓ 未发现可疑情况")

        report.append("\n页面详情:")
        for page_detail in analysis_result['page_details']:
            report.append(f"\n第 {page_detail['page_number']} 页:")
            report.append(f"  图片数量: {page_detail['image_count']}")

            if page_detail['image_count'] > 1:
                report.append("  ⚠️ 该页包含多张图片，可能存在伪造!")
                for img in page_detail['images']:
                    report.append(f"    - 图片{img['index'] + 1}: "
                                  f"{img['width']}x{img['height']}, "
                                  f"颜色空间: {img['colorspace']}, "
                                  f"透明通道: {'有' if img['alpha'] else '无'}")

        return '\n'.join(report)


# 使用示例
def main():
    # 创建检测器实例
    detector = PDFSealDetector()

    # 单个文件分析
    pdf_path = "XXXXXX.pdf"  # 替换为你的PDF文件路径
    result = detector.analyze_pdf(pdf_path)

    # 打印分析报告
    report = detector.generate_report(result)
    print(report)

    # 如果需要更详细的判断，可以添加额外的检测逻辑
    if result.get('is_suspicious'):
        print("\n建议进一步检查:")
        print("1. 对比各图片的尺寸，公章图片通常较小")
        print("2. 检查是否有透明通道（Alpha通道）")
        print("3. 分析图片的位置信息")


# 增强版检测函数 - 包含更多判断逻辑
def enhanced_detection(pdf_path: str) -> Dict:
    """
    增强版检测，包含更多判断维度
    """
    detector = PDFSealDetector()
    result = detector.analyze_pdf(pdf_path)

    if not result.get('is_suspicious'):
        return result

    # 对可疑页面进行深入分析
    enhanced_result = result.copy()
    enhanced_result['seal_probability'] = []

    for page_detail in result['page_details']:
        if page_detail['image_count'] > 1:
            # 分析图片特征
            images = page_detail['images']

            # 检查是否有小尺寸图片（可能是公章）
            small_images = [img for img in images
                            if img['width'] < 300 or img['height'] < 300]

            # 检查是否有透明通道的图片
            transparent_images = [img for img in images if img['alpha']]

            probability = 0
            reasons = []

            if len(small_images) > 0:
                probability += 40
                reasons.append("存在小尺寸图片")

            if len(transparent_images) > 0:
                probability += 30
                reasons.append("存在透明通道图片")

            if page_detail['image_count'] == 2:
                probability += 20
                reasons.append("恰好包含2张图片")

            enhanced_result['seal_probability'].append({
                'page': page_detail['page_number'],
                'probability': probability,
                'reasons': reasons
            })

    return enhanced_result


if __name__ == "__main__":
    main()
