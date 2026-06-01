# 🌿 植物色素层析图像定量分析系统 (Pigment Analysis System)

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.20+-red.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.0+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

本系统是一个基于 **Python** 和 **Streamlit** 开发的自动化植物色素层析分析工具。它能够通过数字图像处理技术，自动识别滤纸层析实验中的色素带，并精准计算比移值（$R_f$ 值），极大简化了传统生物/化学实验中人工测量带来的误差和繁琐步骤。

---

## ✨ 核心特性

* **📸 自动化图像识别**：利用 OpenCV 和 SciPy 信号处理算法，自动定位色素带中心与边界。
* **📏 精准定量分析**：自动提取色带物理距离，并根据原点与溶剂前沿计算标准的 $R_f$ 值。
* **📈 吸收光谱可视化**：生成直观的像素浓度变化曲线（横向 T 型布局）。
* **🧪 化学舱深色 UI**：采用现代化、响应式的毛玻璃深色主题界面，提供极佳的沉浸式科研体验。
* **💾 一键导出功能**：支持将识别范围图、极简标注图以及实验定量数据（CSV）一键保存本地。

---

## 🛠️ 技术栈

* **前端交互**: [Streamlit](https://streamlit.io/)
* **图像处理**: [OpenCV](https://opencv.org/) (cv2)
* **数据分析**: [NumPy](https://numpy.org/), [Pandas](https://pandas.pydata.org/)
* **信号处理**: [SciPy](https://scipy.org/) (`find_peaks`, `peak_widths`)
* **图表绘制**: [Matplotlib](https://matplotlib.org/)

---

## 📖 实验算法说明

系统采用以下数学公式计算比移值：

$$R_f = \frac{D_p}{D_s}$$

其中：
- $D_p$ 为色素带**几何中心**至原点的距离。
- $D_s$ 为**溶剂前沿**至原点的距离。

> **核心改进**：系统不直接采用波峰点（Peak），而是通过信号包络算法获取色带上下边缘，取其算术平均值作为几何中心，有效解决了色带扩散不对称导致的测量偏差。

---

## 🚀 快速开始

### 1. 克隆项目
```bash
git clone [https://github.com/JiYu-Innovation/pigment-analysis.git](https://github.com/JiYu-Innovation/pigment-analysis.git)
cd pigment-analysis

2. 安装依赖
Bash
pip install -r requirements.txt
3. 部署注意 (GitHub Streamlit Cloud)
若在云端部署，请确保包含以下文件以支持中文显示：

packages.txt: 包含 fonts-noto-cjk。

app.py: 已包含动态字体加载逻辑。

📂 项目结构
Plaintext
pigment-analysis/
├── app.py              # 主程序代码
├── requirements.txt    # Python 依赖包
├── packages.txt        # 系统级字体依赖 (Linux)
└── README.md           # 项目说明文档
🤝 贡献与支持
本程序由 济宁师范学院 (Jining Normal University) 电子信息科学与技术专业团队开发。
如果您有任何改进建议，欢迎提交 Pull Request 或联系项目负责人。

Author: Wang Ning (JiYu-Innovation)

Date: 2026-06
