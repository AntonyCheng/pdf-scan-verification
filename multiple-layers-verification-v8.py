import fitz  # PyMuPDF
import os
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from pathlib import Path
from datetime import datetime
import threading


class PDFImageExtractorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("多图层伪造公章工具")
        self.root.geometry("800x600")

        self.input_folder = tk.StringVar()
        self.output_folder = tk.StringVar()

        self.create_widgets()

    def create_widgets(self):
        # 标题
        title_label = tk.Label(self.root, text="多图层伪造公章工具",
                               font=("Arial", 16, "bold"))
        title_label.pack(pady=10)

        # 输入文件夹选择
        input_frame = tk.Frame(self.root)
        input_frame.pack(pady=5, padx=20, fill=tk.X)

        tk.Label(input_frame, text="PDF扫描件文件夹:", width=16, anchor='w').pack(side=tk.LEFT)
        tk.Entry(input_frame, textvariable=self.input_folder, width=80).pack(side=tk.LEFT, padx=5)
        tk.Button(input_frame, text="浏览...", command=self.select_input_folder).pack(side=tk.LEFT)

        # 输出文件夹选择
        output_frame = tk.Frame(self.root)
        output_frame.pack(pady=5, padx=20, fill=tk.X)

        tk.Label(output_frame, text="验证报告输出文件夹:", width=16, anchor='w').pack(side=tk.LEFT)
        tk.Entry(output_frame, textvariable=self.output_folder, width=80).pack(side=tk.LEFT, padx=5)
        tk.Button(output_frame, text="浏览...", command=self.select_output_folder).pack(side=tk.LEFT)

        # 开始按钮
        self.start_button = tk.Button(self.root, text="开始提取",
                                      command=self.start_extraction,
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

        self.log_text = scrolledtext.ScrolledText(log_frame, width=90, height=20,
                                                  wrap=tk.WORD, font=("Consolas", 9))
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def select_input_folder(self):
        folder = filedialog.askdirectory(title="选择包含PDF文件的文件夹")
        if folder:
            self.input_folder.set(folder)

    def select_output_folder(self):
        folder = filedialog.askdirectory(title="选择图片输出文件夹")
        if folder:
            self.output_folder.set(folder)

    def log(self, message):
        """在日志区域显示消息"""
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.root.update()

    def start_extraction(self):
        """开始提取图片"""
        input_dir = self.input_folder.get()
        output_dir = self.output_folder.get()

        if not input_dir or not output_dir:
            messagebox.showwarning("警告", "请选择输入和输出文件夹！")
            return

        if not os.path.exists(input_dir):
            messagebox.showerror("错误", "输入文件夹不存在！")
            return

        # 在新线程中执行，避免界面卡死
        thread = threading.Thread(target=self.process_pdfs,
                                  args=(input_dir, output_dir))
        thread.daemon = True
        thread.start()

    def process_pdfs(self, input_dir, output_dir):
        """处理所有PDF文件"""
        self.start_button.config(state=tk.DISABLED)
        self.log_text.delete(1.0, tk.END)

        # 创建输出目录
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # 查找所有PDF文件
        pdf_files = list(Path(input_dir).glob("*.pdf"))

        if not pdf_files:
            self.log("❌ 未找到任何PDF文件！")
            self.progress_label.config(text="未找到PDF文件", fg="red")
            self.start_button.config(state=tk.NORMAL)
            messagebox.showinfo("提示", "在选定文件夹中未找到PDF文件。")
            return

        self.log(f"📁 找到 {len(pdf_files)} 个PDF文件")
        self.log("=" * 80)

        # 准备报告
        report_lines = []
        report_lines.append("PDF扫描件伪造报告")
        report_lines.append("=" * 80)
        report_lines.append(f"处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"输入文件夹: {input_dir}")
        report_lines.append(f"输出文件夹: {output_dir}")
        report_lines.append(f"PDF文件总数: {len(pdf_files)}")
        report_lines.append("=" * 80)
        report_lines.append("")

        total_images = 0
        processed_files = 0

        # 处理每个PDF
        for idx, pdf_path in enumerate(pdf_files, 1):
            self.progress_label.config(
                text=f"正在处理: {idx}/{len(pdf_files)} - {pdf_path.name}",
                fg="blue"
            )

            self.log(f"\n[{idx}/{len(pdf_files)}] 处理文件: {pdf_path.name}")

            # 为每个PDF创建子文件夹
            pdf_name = pdf_path.stem
            pdf_output_dir = Path(output_dir) / pdf_name

            # 提取图片并收集日志
            image_count, file_log = self.extract_images_from_pdf(
                str(pdf_path),
                str(pdf_output_dir)
            )

            # 只有导出了图片的文件才添加到报告
            if image_count > 0:
                report_lines.append(f"文件: {pdf_path.name}")
                report_lines.append("-" * 80)
                report_lines.extend(file_log)
                report_lines.append("")

                total_images += image_count
                processed_files += 1

        # 添加总结
        report_lines.append("=" * 80)
        report_lines.append("处理总结")
        report_lines.append("-" * 80)
        report_lines.append(f"处理的PDF文件: {len(pdf_files)} 个")
        report_lines.append(f"存在问题的文件: {processed_files} 个")
        report_lines.append(f"共提取证据图片: {total_images} 张")
        report_lines.append("=" * 80)

        # 保存报告
        report_path = Path(output_dir) / f"PDF扫描件伪造报告_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(report_lines))

        self.log("\n" + "=" * 80)
        self.log(f"✅ 全部完成！")
        self.log(f"   处理文件: {len(pdf_files)} 个")
        self.log(f"   证据图片: {total_images} 张")
        self.log(f"   报告已保存: {report_path.name}")

        self.progress_label.config(
            text=f"完成！共提取 {total_images} 张图片",
            fg="green"
        )
        self.start_button.config(state=tk.NORMAL)

        messagebox.showinfo("完成",
                            f"处理完成！\n\n"
                            f"处理文件: {len(pdf_files)} 个\n"
                            f"证据图片: {total_images} 张\n"
                            f"报告已保存到输出文件夹")

    def extract_images_from_pdf(self, pdf_path, output_folder):
        """
        从PDF文件中提取图片（仅提取一页中有多张图片的页面）
        返回: (图片数量, 日志列表)
        """
        file_log = []

        try:
            # 打开PDF文件
            pdf_document = fitz.open(pdf_path)

            image_count = 0
            no_image_pages = 0
            pages_with_images = []

            file_log.append(f"总页数: {len(pdf_document)}")

            # 遍历每一页
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                image_list = page.get_images(full=True)

                # 记录没有图片的页面
                if len(image_list) == 0:
                    no_image_pages += 1
                    file_log.append(f"  第 {page_num + 1} 页: 【无图片页面】")
                    continue

                # 跳过只有1张图片的页面
                if len(image_list) == 1:
                    continue

                # 创建输出文件夹（只在需要时创建）
                if image_count == 0:
                    Path(output_folder).mkdir(parents=True, exist_ok=True)

                file_log.append(f"  第 {page_num + 1} 页: 共 {len(image_list)} 张证据图片")
                pages_with_images.append(page_num + 1)

                # 提取图片
                for img_index, img in enumerate(image_list):
                    xref = img[0]

                    try:
                        base_image = pdf_document.extract_image(xref)
                        image_bytes = base_image["image"]
                        image_ext = base_image["ext"]

                        image_filename = f"page{page_num + 1}_img{img_index + 1}.{image_ext}"
                        image_path = os.path.join(output_folder, image_filename)

                        with open(image_path, "wb") as img_file:
                            img_file.write(image_bytes)

                        image_count += 1
                        file_log.append(f"    ✓ {image_filename}")

                    except Exception as e:
                        file_log.append(f"    ✗ 提取失败: {str(e)}")

            pdf_document.close()

            # 添加统计信息
            if image_count > 0:
                file_log.append(f"提取结果: 共保存 {image_count} 张证据图片")
                if no_image_pages > 0:
                    file_log.append(f"无图片页面: {no_image_pages} 页")
                self.log(f"  ✓ 提取了 {image_count} 张证据图片")
            else:
                self.log(f"  - 未提取图片（符合PDF扫描件规则）")

            return image_count, file_log

        except Exception as e:
            error_msg = f"处理失败: {str(e)}"
            file_log.append(error_msg)
            self.log(f"  ✗ {error_msg}")
            return 0, file_log


def main():
    root = tk.Tk()
    app = PDFImageExtractorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()