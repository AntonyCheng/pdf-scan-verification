# PDF扫描件伪造验证工具

```bash
# 创建环境
conda create -n pdf-scan-verification python=3.11

# 激活环境
conda activate pdf-scan-verification

# 安装依赖
pip install PyMuPDF pyinstaller

# 打包EXE
pyinstaller --onefile --windowed --name "PDF扫描件伪造验证工具" ./multiple-layers-verification-v8.py

```