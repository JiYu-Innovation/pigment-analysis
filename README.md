# 🌿 植物色素层析图像定量分析系统 (Pigment Analysis)

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.20+-red.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.0+-green.svg)

本系统是一个基于 Python 和 Streamlit 开发的**自动化植物色素层析分析工具**。它能够通过数字图像处理技术，自动识别滤纸层析实验中的色素带，并精准计算比移值（Rf值），极大简化了传统生物/化学实验中人工测量带来的误差和繁琐步骤。

## ✨ 核心特性

- 📸 **自动化图像识别**：利用 OpenCV 和 SciPy 信号处理算法，自动定位色素带中心与边界。
- 📊 **精准定量分析**：自动提取色带物理距离，并根据原点与溶剂前沿计算标准的 Rf 值。
- 📈 **吸收光谱可视化**：生成直观的像素浓度变化曲线（横向 T 型布局）。
- 🧪 **化学舱深色 UI**：采用现代化、响应式的毛玻璃深色主题界面，提供极佳的沉浸式科研体验。
- 💾 **一键导出功能**：支持将识别范围图、极简标注图以及实验定量数据（CSV）一键保存本地。

## 🛠️ 技术栈

* **前端交互**：[Streamlit](https://streamlit.io/)
* **图像处理**：[OpenCV](https://opencv.org/) (cv2)
* **数据分析**：[NumPy](https://numpy.org/), [Pandas](https://pandas.pydata.org/)
* **信号处理**：[SciPy](https://scipy.org/) (`find_peaks`, `peak_widths`)
* **图表绘制**：[Matplotlib](https://matplotlib.org/)

## 🚀 快速开始

### 1. 克隆项目

```bash
https://github.com/JiYu-Innovation/pigment-analysis.git
```

