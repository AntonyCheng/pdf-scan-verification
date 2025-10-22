import fitz  # PyMuPDF
import os
from pathlib import Path
from PIL import Image
import io


def extract_images_from_pdf(pdf_path, output_folder="extracted_images"):
    """
    从PDF文件中提取所有图片

    参数:
        pdf_path: PDF文件路径
        output_folder: 输出文件夹路径，默认为 "extracted_images"
    """
    # 创建输出文件夹
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    # 打开PDF文件
    pdf_document = fitz.open(pdf_path)

    # 获取PDF文件名（不含扩展名）
    pdf_name = Path(pdf_path).stem

    image_count = 0

    print(f"开始处理PDF文件: {pdf_path}")
    print(f"总页数: {len(pdf_document)}")

    # 遍历每一页
    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]

        # 获取页面中的图片列表
        image_list = page.get_images(full=True)

        print(f"\n第 {page_num + 1} 页，发现 {len(image_list)} 张图片")

        # 遍历页面中的每张图片
        for img_index, img in enumerate(image_list):
            xref = img[0]  # 图片的xref编号

            try:
                # 提取图片
                base_image = pdf_document.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]  # 图片扩展名

                # 生成图片文件名
                image_filename = f"{pdf_name}_page{page_num + 1}_img{img_index + 1}.{image_ext}"
                image_path = os.path.join(output_folder, image_filename)

                # 保存图片
                with open(image_path, "wb") as img_file:
                    img_file.write(image_bytes)

                image_count += 1
                print(f"  ✓ 保存: {image_filename}")

            except Exception as e:
                print(f"  ✗ 提取图片失败 (xref: {xref}): {str(e)}")

    pdf_document.close()

    print(f"\n提取完成！共保存 {image_count} 张图片到文件夹: {output_folder}")
    return image_count


def extract_images_advanced(pdf_path, output_folder="extracted_images", min_size=0):
    """
    高级版本：可以过滤小尺寸图片

    参数:
        pdf_path: PDF文件路径
        output_folder: 输出文件夹路径
        min_size: 最小图片尺寸（宽或高），单位像素，0表示不过滤
    """
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    pdf_document = fitz.open(pdf_path)
    pdf_name = Path(pdf_path).stem

    image_count = 0
    skipped_count = 0

    print(f"开始处理PDF文件: {pdf_path}")
    print(f"总页数: {len(pdf_document)}")
    if min_size > 0:
        print(f"最小尺寸过滤: {min_size}px")

    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]
        image_list = page.get_images(full=True)

        print(f"\n第 {page_num + 1} 页，发现 {len(image_list)} 张图片")

        for img_index, img in enumerate(image_list):
            xref = img[0]

            try:
                base_image = pdf_document.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]

                # 如果设置了尺寸过滤，检查图片尺寸
                if min_size > 0:
                    img_obj = Image.open(io.BytesIO(image_bytes))
                    width, height = img_obj.size

                    if width < min_size and height < min_size:
                        skipped_count += 1
                        print(f"  - 跳过小图片 ({width}x{height}px)")
                        continue

                image_filename = f"{pdf_name}_page{page_num + 1}_img{img_index + 1}.{image_ext}"
                image_path = os.path.join(output_folder, image_filename)

                with open(image_path, "wb") as img_file:
                    img_file.write(image_bytes)

                image_count += 1
                print(f"  ✓ 保存: {image_filename}")

            except Exception as e:
                print(f"  ✗ 提取图片失败 (xref: {xref}): {str(e)}")

    pdf_document.close()

    print(f"\n提取完成！")
    print(f"  - 保存图片: {image_count} 张")
    if skipped_count > 0:
        print(f"  - 跳过图片: {skipped_count} 张")
    print(f"  - 输出文件夹: {output_folder}")

    return image_count


# 使用示例
if __name__ == "__main__":
    # 基础用法：提取所有图片
    pdf_file = "哈尔滨灵彩电子科技有限公司.pdf"  # 替换为你的PDF文件路径
    extract_images_from_pdf(pdf_file)

    # 高级用法：过滤掉宽高都小于100px的图片
    # extract_images_advanced(pdf_file, output_folder="large_images", min_size=100)