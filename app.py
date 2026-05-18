import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from PIL import Image

# 页面标题
st.set_page_config(page_title="色素层析定量识别系统", layout="wide")
st.title("🧪 植物色素层析图像定量分析系统")
st.markdown("通过计算机视觉技术自动寻找色素带中心，并计算相对于 1cm 原点的物理距离。")

# --- 侧边栏：参数调节 ---
st.sidebar.header("参数调节")
physical_h = st.sidebar.number_input("滤纸总高度 (cm)", value=10.0, step=0.5)
origin_offset = st.sidebar.number_input("原点距离底部高度 (cm)", value=1.0, step=0.1)

st.sidebar.subheader("算法灵敏度")
# 这里解决了你之前提到的识别不到浅色块的问题，让用户自己实时微调
prom = st.sidebar.slider("Prominence (灵敏度/突出程度)", 0.1, 5.0, 1.0, 0.1)
dist_cm = st.sidebar.slider("最小间距 (cm)", 0.05, 0.5, 0.12, 0.01)

# --- 文件上传 ---
uploaded_file = st.file_uploader("请上传裁剪好的单条滤纸照片...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # 读取图片
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h_px, w_px, _ = img.shape

    # 1. 比例尺计算
    cm_per_px = physical_h / h_px
    origin_y = h_px - int(origin_offset / cm_per_px)
    min_dist_px = int(dist_cm / cm_per_px)

    # 2. 强化信号处理 (Pigment Index)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1].astype(np.float32)
    v = hsv[:, :, 2].astype(np.float32)
    pigment_index = s + (255 - v)
    pigment_blur = cv2.GaussianBlur(pigment_index, (7, 7), 0)
    profile = np.mean(pigment_blur, axis=1)

    # 3. 寻找波峰
    peaks, _ = find_peaks(profile, distance=min_dist_px, prominence=prom)

    # 4. 结果记录
    vis_img = img_rgb.copy()
    results = []
    for cY in peaks:
        if cY <= origin_y:
            dist_cm_val = (origin_y - cY) * cm_per_px
            color = img_rgb[cY, w_px//2]
            results.append({"cY": cY, "dist": dist_cm_val, "color": color})
            # 绘制
            cv2.line(vis_img, (0, cY), (w_px, cY), (255, 0, 0), 2)
            cv2.putText(vis_img, f"{dist_cm_val:.2f}cm", (10, cY - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    results.sort(key=lambda x: x['dist'])

    # --- 网页展示 ---
    col1, col2, col3 = st.columns([1, 1, 1.5])

    with col1:
        st.subheader("识别结果图")
        # 绘制原点线
        cv2.line(vis_img, (0, origin_y), (w_px, origin_y), (0, 255, 0), 3)
        st.image(vis_img, use_column_width=True)

    with col2:
        st.subheader("浓度分布曲线")
        fig_curve, ax_c = plt.subplots()
        ax_c.plot(profile, range(len(profile)), color='purple')
        ax_c.plot(profile[peaks], peaks, "x", color='red')
        ax_c.axhline(y=origin_y, color='green', linestyle='--')
        ax_c.invert_yaxis()
        ax_c.set_xlabel("浓度强度")
        st.pyplot(fig_curve)

    with col3:
        st.subheader("定量数据表")
        if results:
            data = {"条带": [f"色素带 {i+1}" for i in range(len(results))],
                    "距离 (cm)": [f"{r['dist']:.2f}" for r in results]}
            st.table(data)
            
            st.subheader("提取出的颜色")
            # 展示色块
            c_cols = st.columns(len(results))
            for i, r in enumerate(results):
                with c_cols[i]:
                    # 创建一个小色块
                    color_block = np.zeros((50, 50, 3), dtype=np.uint8)
                    color_block[:] = r['color']
                    st.image(color_block, caption=f"{r['dist']:.2f}cm")
        else:
            st.warning("未检测到波峰，请尝试调低左侧 Prominence 灵敏度。")

else:
    st.info("👋 欢迎！请在上方上传一张滤纸层析图片开始分析。")