# PDF扫描件伪造验证工具

```bash
# 创建环境
conda create -n pdf-scan-verification python=3.11

# 激活环境
conda activate pdf-scan-verification

# 安装依赖
pip install PyMuPDF pyinstaller

# 打包检测多图层伪造公章工具
pyinstaller --onefile --windowed --name "多图层伪造公章工具" ./multiple-layers-verification-v8.py
# 打包检测相同/相似公章工具
pyinstaller --onefile --windowed --name "公章相似度工具" ./similarity-verification-v9.py

```