import fitz  # PyMuPDF
import os
from pathlib import Path


def extract_images_from_pdf(pdf_path, output_folder="extracted_images"):
    """
    从PDF文件中提取图片（仅提取一页中有多张图片的页面）

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
    skipped_pages = 0
    no_image_pages = 0

    print(f"开始处理PDF文件: {pdf_path}")
    print(f"总页数: {len(pdf_document)}")

    # 遍历每一页
    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]

        # 获取页面中的图片列表
        image_list = page.get_images(full=True)

        print(f"\n第 {page_num + 1} 页，发现 {len(image_list)} 张图片")

        # 特别标注没有图片的页面
        if len(image_list) == 0:
            print(f"  ⚠️  【无图片页面】")
            no_image_pages += 1
            continue

        # 仅在一页有多张图片时才提取
        if len(image_list) == 1:
            print(f"  - 跳过（只有 1 张图片）")
            skipped_pages += 1
            continue

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

    print(f"\n提取完成！")
    print(f"  - 共保存 {image_count} 张图片")
    print(f"  - 跳过 {skipped_pages} 页（只有 1 张图片）")
    print(f"  - ⚠️  无图片页面: {no_image_pages} 页")
    print(f"  - 输出文件夹: {output_folder}")
    return image_count


# 使用示例
if __name__ == "__main__":
    # 提取一页中有多张图片的页面
    pdf_file = "终验合格证书-real.pdf"  # 替换为你的PDF文件路径
    extract_images_from_pdf(pdf_file)