import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths
import pandas as pd
import platform

# 1. 自动识别环境设置字体
system_type = platform.system()

if system_type == "Linux":
    # 针对 GitHub Streamlit Cloud (Linux) 的设置
    plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'Noto Sans CJK SC', 'DejaVu Sans']
else:
    # 针对本地 Windows 或 Mac 的设置
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'sans-serif']

plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题

# 页面基础配置
st.set_page_config(page_title="植物色素层析图像定量分析系统", layout="wide", initial_sidebar_state="expanded")

# --- 🧪 深度美化 CSS (化学实验舱/深色模式适配) ---
st.markdown("""
    <style>
    /* 全局容器：增加毛玻璃质感和微光边框 */
    .block-container { 
        max-width: 1100px !important; 
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
    [data-testid="stSidebar"] { width: 280px !important; border-right: 1px solid rgba(0, 188, 212, 0.15); }
    
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
    prom = st.slider("波峰灵敏度 (Prominence)", 0.1, 5.0, 0.8, 0.1)
    dist_cm = st.slider("最小分辨间距 (cm)", 0.05, 1.0, 0.15, 0.05)
    rel_height = st.slider("色带宽测定比例 (Boundary)", 0.2, 0.95, 0.65, 0.05, 
                           help="控制色带边界(上下框)的宽度计算比例，调节此参数可直接改变色带宽及几何中点")
    
    st.caption("💡 提示: 若漏掉浅色带，请降低“波峰灵敏度”。调整“宽测定比例”可以完美控制色彩框上下边缘的包裹范围。")

st.title("🌿 植物色素层析图像定量分析系统 🔬")

# --- 核心辅助函数：物理顺序 + HSV 双通道色素高精度定性法 ---
def identify_pigments(results_list):
    """
    根据Rf值(从大到小)结合颜色对检测出的色素带进行多重检验分类。
    经典纸层析排序：胡萝卜素(最高) > 叶黄素 > 叶绿素a > 叶绿素b(最低)
    """
    # 按照 Rf 值降序排列 (即离原点从远到近)
    sorted_indices = sorted(range(len(results_list)), key=lambda i: results_list[i]['Rf值'], reverse=True)
    
    # 建立标准映射模板
    standard_names = ["胡萝卜素 (Orange-yellow)", "叶黄素 (Yellow)", "叶绿素 a (Blue-green)", "叶绿素 b (Yellow-green)"]
    
    for rank, idx in enumerate(sorted_indices):
        item = results_list[idx]
        hsv_color = item['_hsv']
        h, s, v = hsv_color
        
        # 1. 物理位置层析优先匹配
        if len(sorted_indices) <= 4:
            # 如果层析带数量正常(不超过4个)，根据Rf大小顺序进行直接定位
            guessed_name = standard_names[min(rank, len(standard_names)-1)]
        else:
            # 2. 如果检测出色差噪点带，使用色调与顺序混合校验
            if h < 18 or h > 165:
                guessed_name = "胡萝卜素"
            elif 18 <= h < 35:
                guessed_name = "叶黄素"
            elif 35 <= h < 82:
                guessed_name = "叶绿素 a"
            elif 82 <= h < 105:
                guessed_name = "叶绿素 b"
            else:
                guessed_name = "其他成分/杂质"
                
        results_list[idx]['色素名称'] = guessed_name
    return results_list

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

    # 2. 图像信号提取 (饱和度 + 亮度逆反增强)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s_channel = hsv[:, :, 1].astype(np.float32)
    v_channel = hsv[:, :, 2].astype(np.float32)
    pigment_index = s_channel + (255 - v_channel)
    pigment_blur = cv2.GaussianBlur(pigment_index, (5, 5), 0)
    profile = np.mean(pigment_blur, axis=1)

    # 3. 寻找色素波峰与对应的边界
    min_dist_px = int(dist_cm / cm_per_px)
    peaks, _ = find_peaks(profile, distance=min_dist_px, prominence=prom)
    # 计算波峰的包络宽度，这决定了彩色方框的 y_top 和 y_bottom
    _, _, left_ips, right_ips = peak_widths(profile, peaks, rel_height=rel_height)

    # 4. 结果计算与标注绘制
    vis_img_boxes = img_rgb.copy()
    vis_img_lines = img_rgb.copy()
    raw_results = []
    
    # 绘制原点 (绿线) 和溶剂前沿 (红线)
    cv2.line(vis_img_boxes, (0, origin_y), (w_px, origin_y), (0, 255, 0), 2)
    cv2.line(vis_img_boxes, (0, solvent_y), (w_px, solvent_y), (255, 0, 0), 2)
    cv2.line(vis_img_lines, (0, origin_y), (w_px, origin_y), (0, 255, 0), 2)
    cv2.line(vis_img_lines, (0, solvent_y), (w_px, solvent_y), (255, 0, 0), 2)

    box_colors = [(255, 87, 34), (76, 175, 80), (33, 150, 243), (255, 193, 7), (156, 39, 176), (0, 188, 212)]
    
    # 记录修改后的几何中点
    center_y_list = []

    for i, p_y in enumerate(peaks):
        if solvent_y < p_y < origin_y:
            # 提取色带上下最外围边缘
            y_top = max(0, int(round(left_ips[i])))
            y_bottom = min(h_px - 1, int(round(right_ips[i])))
            
            # ================= ✨ CRITICAL FIX (核心改进) ✨ =================
            # 计算色带上下边界的几何中点，取代原本不对称的峰值点 p_y 
            center_y = int(round((y_top + y_bottom) / 2.0))
            center_y_list.append(center_y)
            # ================================================================

            # 基于全新几何中点计算到原点的物理距离及 Rf
            dist_px = origin_y - center_y
            dist_cm_val = dist_px * cm_per_px
            rf_value = dist_px / total_run_px if total_run_px > 0 else 0
            
            # 挑选绘制颜色
            box_color = box_colors[i % len(box_colors)]
            
            # --- 绘制图 A (精确边界盒 + 几何对称中心标识) ---
            cv2.rectangle(vis_img_boxes, (0, y_top), (w_px, y_bottom), box_color, 2)
            # 绘制水平中心参考线 (现在完美位于框内中央)
            cv2.line(vis_img_boxes, (0, center_y), (w_px, center_y), (0, 191, 255), 1) 
            # 绘制对称中心十字标记
            cv2.drawMarker(vis_img_boxes, (w_px//2, center_y), (0, 191, 255), cv2.MARKER_TILTED_CROSS, 15, 2)
            # 标示中心对应的距离数值
            cv2.putText(vis_img_boxes, f"{dist_cm_val:.2f}cm", (12, center_y - 8), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

            # --- 绘制图 B (极简中心线视图) ---
            cv2.line(vis_img_lines, (0, center_y), (w_px, center_y), (255, 0, 0), 1) 
            cv2.putText(vis_img_lines, f"{dist_cm_val:.2f}cm", (12, center_y - 8), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

            raw_results.append({
                "色素名称": "检测中...", # 将由 identify_pigments 重新精确定位
                "距离(cm)": round(dist_cm_val, 2),
                "Rf值": round(rf_value, 3),
                "RGB": img_rgb[center_y, w_px//2],
                "_hsv": hsv[center_y, w_px//2],
                "_center_y": center_y,
                "_y_top": y_top,
                "_y_bottom": y_bottom
            })

    # 进行物理顺序纠正和色调合并定性
    results = identify_pigments(raw_results) if raw_results else []

    # --- 准备下载图像数据 ---
    _, buffer_boxes = cv2.imencode('.png', cv2.cvtColor(vis_img_boxes, cv2.COLOR_RGB2BGR))
    _, buffer_lines = cv2.imencode('.png', cv2.cvtColor(vis_img_lines, cv2.COLOR_RGB2BGR))
    byte_boxes = buffer_boxes.tobytes()
    byte_lines = buffer_lines.tobytes()

    # ================= 网页排版：左一右二 =================

    col_left, col_right = st.columns([1.2, 1.8], gap="large")

    # ===== 左侧模块 =====
    with col_left:
        st.subheader("🧪 色带分离图谱 (Chromatogram)")
        img_col1, img_col2 = st.columns(2)
        with img_col1:
            st.image(vis_img_boxes, use_container_width=True, caption="A. 彩框边界与几何中心标注")
            st.download_button(label="📥 下载识别范围", data=byte_boxes, file_name="范围图_几何中心版.png", mime="image/png")
        with img_col2:
            st.image(vis_img_lines, use_container_width=True, caption="B. 几何中心极简标注")
            st.download_button(label="📥 下载极简中心", data=byte_lines, file_name="极简图_几何中心版.png", mime="image/png")

    # ===== 右侧模块 =====
    with col_right:
        # 1. 浓度分布曲线 (透明背景美化版，叠加几何中心标记)
        st.subheader("📈 吸收光谱与浓度曲线 (Absorbance)")
        fig, ax = plt.subplots(figsize=(6, 2.7)) 
        
        # 图表背景透明化适配深色模式
        fig.patch.set_facecolor('none')
        ax.set_facecolor('none')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#666666')
        ax.spines['left'].set_color('#666666')
        ax.tick_params(axis='y', colors='#aaaaaa')
        
        # 绘制主光谱曲线
        ax.plot(profile, color='#00BCD4', linewidth=2, label='光谱浓度指数') 
        
        # 绘制检测得到的色带几何中心 (而非原始非对称波峰)
        if results:
            centers_y = [r['_center_y'] for r in results]
            ax.plot(centers_y, profile[centers_y], "rx", markersize=8, markeredgewidth=2, label='几何中心点')
            
            # 在图表上绘制半透明的边界范围填充
            for r in results:
                ax.axvspan(r['_y_top'], r['_y_bottom'], color='#FFC107', alpha=0.15)
                
        ax.axvline(x=origin_y, color='#4CAF50', linestyle='--', alpha=0.8, label='点样原点')
        ax.axvline(x=solvent_y, color='#FF5722', linestyle='--', alpha=0.8, label='溶剂前沿')
        ax.set_xticks([]) 
        ax.legend(fontsize='small', loc="upper left", facecolor='#161920', edgecolor='none', labelcolor='white')
        st.pyplot(fig)

        # 2. 定量数据汇总
        st.subheader("📊 实验定量数据面板 (Data Report)")
        if results:
            # 数据表降序展示
            results.sort(key=lambda x: x['Rf值'], reverse=True)
            df = pd.DataFrame(results).drop(columns=['RGB', '_hsv', '_center_y', '_y_top', '_y_bottom'])
            
            # 美化展示数据框
            st.dataframe(df, use_container_width=True, height=180)
            
            # 底部工具栏
            tools_c1, tools_c2 = st.columns([1.2, 1.8])
            with tools_c1:
                csv_data = df.to_csv(index=False).encode('utf-8-sig')
                st.download_button("💾 导出原始数据 (.csv)", csv_data, "色素几何中心定量报告.csv", "text/csv")
            
            with tools_c2:
                # 绘制色块提取验证
                st.markdown("**🎨 几何中心点色素原色提取**")
                c_grid = st.columns(min(len(results), 4))
                for i, r in enumerate(results[:4]):
                    with c_grid[i]:
                        block = np.zeros((20, 20, 3), dtype=np.uint8)
                        block[:] = r['RGB']
                        st.image(block, use_container_width=True)
                        st.caption(f"<div style='text-align:center;font-size:10px;'>{r['色素名称'].split(' ')[0]}</div>", unsafe_allow_html=True)
        else:
            st.warning("⚠️ 信号微弱：未能检测到有效色素带，请降低左侧‘波峰灵敏度’。")

else:
    # 居中的空状态提示
    st.info("💡 请在上方上传拍摄清晰、裁剪整齐的单条滤纸图片，开始高精度几何定量分析。")
