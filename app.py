import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths
import pandas as pd

# 1. 适配中文显示（Matplotlib）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'sans-serif'] 
plt.rcParams['axes.unicode_minus'] = False

# 页面基础配置
st.set_page_config(page_title="色素层析定量分析系统", layout="wide", initial_sidebar_state="expanded")

# --- 🧪 深度美化 CSS (化学实验舱/深色模式适配) ---
st.markdown("""
    <style>
    /* 全局容器：增加毛玻璃质感和微光边框 */
    .block-container { 
        max-width: 1050px !important; 
        padding: 2rem 2.5rem !important; 
        background: rgba(25, 29, 36, 0.4); 
        border: 1px solid rgba(76, 175, 80, 0.15);
        border-radius: 12px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        margin-top: 1.5rem;
    }
    
    /* 组件紧凑度调节 */
    .stElementContainer { margin-bottom: -2px !important; }
    
    /* 标题：叶绿素与溶剂的化学荧光渐变 */
    h1 { 
        font-size: 2.1rem !important; 
        margin-top: -1rem !important; 
        line-height: 1.3 !important; 
        padding-bottom: 1.5rem; 
        text-align: center;
        background: -webkit-linear-gradient(45deg, #4CAF50, #00BCD4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
    
    /* 小标题：增加虚线烧杯刻度感 */
    h3 { 
        font-size: 1.15rem !important; 
        margin-bottom: 0.8rem; 
        color: #E0E0E0 !important;
        border-bottom: 1px dashed rgba(76, 175, 80, 0.4);
        padding-bottom: 6px;
    }
    
    /* 侧边栏微调 */
    [data-testid="stSidebar"] { width: 260px !important; border-right: 1px solid rgba(0, 188, 212, 0.15); }
    
    /* 图片边框：模拟显微镜玻片质感 */
    .stImage > img { 
        border: 1px solid rgba(255,255,255,0.1); 
        border-radius: 6px; 
        padding: 4px; 
        background-color: rgba(0,0,0,0.3);
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        transition: transform 0.3s ease;
    }
    .stImage > img:hover { transform: scale(1.01); border-color: rgba(76, 175, 80, 0.5); }
    
    /* 下载按钮：荧光交互效果 */
    .stDownloadButton > button {
        border-radius: 6px;
        border: 1px solid rgba(76, 175, 80, 0.4);
        color: #81C784;
        background-color: transparent;
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
        width: 100%;
    }
    .stDownloadButton > button:hover {
        background-color: rgba(76, 175, 80, 0.15);
        border: 1px solid #4CAF50;
        color: #fff;
        box-shadow: 0 0 12px rgba(76, 175, 80, 0.5);
    }
    
    /* 隐藏DataFrame默认的索隐栏空白 */
    [data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; border: 1px solid rgba(255,255,255,0.05); }
    </style>
    """, unsafe_allow_html=True)

# --- 侧边栏：参数调节 ---
with st.sidebar:
    st.header("📐 物理参数 (Scale)")
    physical_h = st.number_input("滤纸总长 (cm)", value=10.0, step=0.1)
    origin_offset = st.number_input("点样原点高度 (cm)", value=1.0, step=0.1)
    solvent_offset = st.number_input("溶剂前沿高度 (cm)", value=9.0, step=0.1)
    
    st.header("🧬 算法微调 (Algorithm)")
    prom = st.slider("波峰灵敏度", 0.1, 5.0, 0.8, 0.1)
    dist_cm = st.slider("最小分辨间距(cm)", 0.05, 1.0, 0.15, 0.05)
    st.caption("💡 提示: 若漏掉浅色带，请适当调低灵敏度。")

st.title("🌿 植物色素层析图像定量分析系统 🔬")

# --- 核心逻辑函数 ---
def identify_pigment_name(hsv_color):
    h, s, v = hsv_color
    if s < 40: return "未知/浅色带"
    if h < 15 or h > 165: return "胡萝卜素"
    if 15 <= h < 35: return "叶黄素"
    if 35 <= h < 80: return "叶绿素 a"
    if 80 <= h < 100: return "叶绿素 b"
    return "其他成分"

# --- 文件上传 ---
uploaded_file = st.file_uploader("📤 上传滤纸层析照片...", type=["png", "jpg", "jpeg"], label_visibility="collapsed")

if uploaded_file is not None:
    # 图像解码
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h_px, w_px, _ = img.shape

    # 1. 物理坐标系换算
    cm_per_px = physical_h / h_px
    origin_y = h_px - int(origin_offset / cm_per_px)
    solvent_y = h_px - int(solvent_offset / cm_per_px)
    total_run_px = origin_y - solvent_y 

    # 2. 图像信号提取
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s_channel = hsv[:, :, 1].astype(np.float32)
    v_channel = hsv[:, :, 2].astype(np.float32)
    pigment_index = s_channel + (255 - v_channel)
    pigment_blur = cv2.GaussianBlur(pigment_index, (5, 5), 0)
    profile = np.mean(pigment_blur, axis=1)

    # 3. 寻找色素带中心
    min_dist_px = int(dist_cm / cm_per_px)
    peaks, _ = find_peaks(profile, distance=min_dist_px, prominence=prom)
    _, _, left_ips, right_ips = peak_widths(profile, peaks, rel_height=0.65)

    # 4. 结果计算与标注绘制
    vis_img_boxes = img_rgb.copy()
    vis_img_lines = img_rgb.copy()
    results = []
    
    box_colors = [(255, 87, 34), (76, 175, 80), (33, 150, 243), (255, 193, 7), (156, 39, 176), (0, 188, 212)]
    
    cv2.line(vis_img_boxes, (0, origin_y), (w_px, origin_y), (0, 255, 0), 2)
    cv2.line(vis_img_boxes, (0, solvent_y), (w_px, solvent_y), (0, 0, 255), 2)
    cv2.line(vis_img_lines, (0, origin_y), (w_px, origin_y), (0, 255, 0), 2)
    cv2.line(vis_img_lines, (0, solvent_y), (w_px, solvent_y), (0, 0, 255), 2)

    for i, p_y in enumerate(peaks):
        if solvent_y < p_y < origin_y:
            dist_px = origin_y - p_y
            dist_cm_val = dist_px * cm_per_px
            rf_value = dist_px / total_run_px if total_run_px > 0 else 0
            
            # --- 绘制图 A (彩框) ---
            y_top = max(0, int(left_ips[i]))
            y_bottom = min(h_px, int(right_ips[i]))
            box_color = box_colors[i % len(box_colors)]
            cv2.rectangle(vis_img_boxes, (0, y_top), (w_px, y_bottom), box_color, 2)
            cv2.line(vis_img_boxes, (0, p_y), (w_px, p_y), (255, 0, 0), 1) 
            cv2.putText(vis_img_boxes, f"{dist_cm_val:.2f}cm", (10, p_y - 8), 1, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.drawMarker(vis_img_boxes, (w_px//2, p_y), (255, 0, 0), cv2.MARKER_TILTED_CROSS, 15, 2)

            # --- 绘制图 B (红线极简) ---
            cv2.line(vis_img_lines, (0, p_y), (w_px, p_y), (255, 0, 0), 1) 
            cv2.putText(vis_img_lines, f"{dist_cm_val:.2f}cm", (10, p_y - 8), 1, 0.8, (0, 0, 0), 1, cv2.LINE_AA)

            results.append({
                "色素名称": identify_pigment_name(hsv[p_y, w_px//2]),
                "距离(cm)": round(dist_cm_val, 2),
                "Rf值": round(rf_value, 3),
                "RGB": img_rgb[p_y, w_px//2]
            })

    # --- 准备下载数据 ---
    _, buffer_boxes = cv2.imencode('.png', cv2.cvtColor(vis_img_boxes, cv2.COLOR_RGB2BGR))
    _, buffer_lines = cv2.imencode('.png', cv2.cvtColor(vis_img_lines, cv2.COLOR_RGB2BGR))
    byte_boxes = buffer_boxes.tobytes()
    byte_lines = buffer_lines.tobytes()

    # ================= 网页全新排版：左一右二 =================

    col_left, col_right = st.columns([1.2, 1.8], gap="large")

    # ===== 左侧模块 =====
    with col_left:
        st.subheader("🧪 色带分离图谱 (Chromatogram)")
        img_col1, img_col2 = st.columns(2)
        with img_col1:
            st.image(vis_img_boxes, use_container_width=True)
            st.download_button(label="📥 下载识别范围", data=byte_boxes, file_name="范围图.png", mime="image/png")
        with img_col2:
            st.image(vis_img_lines, use_container_width=True)
            st.download_button(label="📥 下载极简标注", data=byte_lines, file_name="极简图.png", mime="image/png")

    # ===== 右侧模块 =====
    with col_right:
        # 1. 浓度分布曲线 (透明背景美化版)
        st.subheader("📈 吸收光谱与浓度曲线 (Absorbance)")
        fig, ax = plt.subplots(figsize=(6, 2.6)) 
        
        # 图表背景透明化适配深色模式
        fig.patch.set_facecolor('none')
        ax.set_facecolor('none')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#666666')
        ax.spines['left'].set_color('#666666')
        ax.tick_params(axis='y', colors='#aaaaaa')
        
        ax.plot(profile, color='#00BCD4', linewidth=2, label='浓度指数') # 改用科技青色
        ax.plot(peaks, profile[peaks], "rx", markersize=8, label='波峰中心')
        ax.axvline(x=origin_y, color='#4CAF50', linestyle='--', alpha=0.8, label='点样原点')
        ax.axvline(x=solvent_y, color='#FF5722', linestyle='--', alpha=0.8, label='溶剂前沿')
        ax.set_xticks([]) 
        ax.legend(fontsize='small', loc="upper left", facecolor='#1E1E1E', edgecolor='none', labelcolor='white')
        st.pyplot(fig)

        # 2. 定量数据汇总
        st.subheader("📊 实验定量数据面板 (Data Report)")
        if results:
            results.sort(key=lambda x: x['Rf值'], reverse=True)
            df = pd.DataFrame(results).drop(columns=['RGB'])
            
            st.dataframe(df, use_container_width=True, height=180)
            
            # 底部工具栏
            tools_c1, tools_c2 = st.columns([1.2, 1.5])
            with tools_c1:
                csv_data = df.to_csv(index=False).encode('utf-8-sig')
                st.download_button("💾 导出原始数据 (.csv)", csv_data, "色素实验.csv", "text/csv")
            
            with tools_c2:
                c_grid = st.columns(min(len(results), 4))
                for i, r in enumerate(results[:4]):
                    with c_grid[i]:
                        block = np.zeros((15, 15, 3), dtype=np.uint8)
                        block[:] = r['RGB']
                        st.image(block, caption=r['色素名称'])
        else:
            st.warning("⚠️ 信号微弱：未能检测到有效色素带，请降低左侧‘灵敏度’。")

else:
    # 居中的空状态提示
    st.info("💡 请在上方上传拍摄清晰、裁剪整齐的单条滤纸图片，开始实验分析。")
