import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

# ==========================================
# 1. 配置区域 (样式调整)
# ==========================================
CONFIG = {
    # --- 画布设置 ---
    'figure_size': (15, 7),

    # --- 颜色设置 (4种) ---
    'colors_rgb': [
        (225, 172, 166), # 1. 桃粉
        (145, 180, 209), # 2. 天蓝
        (182, 204, 185), # 3. 豆绿
        (199, 180, 190), # 4. 藕荷
    ],

    # --- 字体与点大小 ---
    'font_family': 'Arial',
    'marker_size': 180,
    'label_fontsize': 10,

    # --- 背景渐变色设置 ---
    'bg_gradient_strength': 0.15, # 渐变强度 (0-1)，越小越淡
    # High D (右侧) 的倾向色 (R, G, B) - 设为淡粉红
    'color_high_d': (1.0, 0.85, 0.85),
    # High P (上方) 的倾向色 (R, G, B) - 设为淡青蓝
    'color_high_p': (0.85, 0.95, 1.0),
    # 原点 (左下) 的基础色 - 纯白
    'color_base': (1.0, 1.0, 1.0),

    # --- 边框与轴范围 ---
    'axis_limit_start': 48,
    'axis_limit_end': -3,
}

# ==========================================
# 2. 分子字典
# ==========================================
MOLECULE_MAP = {
    # --- 1. Ethers ---
    '乙二醇二甲醚（DME）': ('Ethers', 'DME'),
    '1,3-二氧戊环（DOL）': ('Ethers', 'DOL'),
    '四氢呋喃(THF)': ('Ethers', 'THF'),
    '甲缩醛（DMM)': ('Ethers', 'DMM'),
    '三聚甲醛（TO）': ('Ethers', 'TO'),
    '乙二醇二丁醚（EGBE）': ('Ethers', 'EGBE'),
    '乙二醇甲醚二氟乙醚（DMEE）': ('Ethers', 'DMEE'),
    '1，1，2，2-四氟乙基-2，2，2-三氟乙基醚（HFE）': ('Ethers', 'HFE'),
    '1,1,1-三氟-2-(2-甲氧基乙氧基)乙烷（TFEE）': ('Ethers', 'TFEE'),
    '六氟异丙基甲醚（HFME）': ('Ethers', 'HFME'),
    '2-甲基四氢呋喃(MeTHF)': ('Ethers', 'MeTHF'),
    'Ethane, methoxy-': ('Ethers', 'EME'),
    'Ethane, 1,2-dimethoxy-': ('Ethers', '1,2-DME'),
    'Ethene, ethoxy-': ('Ethers', 'EOE'),
    'Vinyl ether': ('Ethers', 'AVE'),
    'Furan, 2,3-dihydro-5-methyl-': ('Ethers', '2,3-DHF'),

    # --- 2. Carbonates ---
    '碳酸乙烯酯(EC)': ('Carbonates', 'EC'),
    '碳酸二甲酯（DMC）': ('Carbonates', 'DMC'),
    '碳酸甲乙酯(EMC)': ('Carbonates', 'EMC'),
    '碳酸二乙酯(DEC)': ('Carbonates', 'DEC'),
    '氟代碳酸乙烯酯（FEC）': ('Carbonates', 'FEC'),
    '双氟碳酸乙烯酯(DFEC)': ('Carbonates', 'DFEC'),
    '甲基三氟乙基碳酸酯(FEMC)': ('Carbonates', 'FEMC'),

    # --- 3. Esters ---
    '丙烯酸甲酯（MA）': ('Esters', 'MA'),
    '2-Propenoic acid, methyl ester': ('Esters', 'MA(en)'),
    '三氟乙酸乙酯(ETFA)': ('Esters', 'ETFA'),
    '二氟乙酸甲酯（MDFA）': ('Esters', 'MDFA'),
    '氟磺酰基二氟乙酸甲酯（MDFSA）': ('Esters', 'MDFSA'),
    '丙二醇甲醚醋酸酯(PMA）': ('Esters', 'PMA'),

    # --- 4. Nitriles ---
    '氟乙腈(FAN)': ('Nitriles', 'FAN'),
    'Succinonitrile': ('Nitriles', 'SN'),
    'Butanenitrile': ('Nitriles', 'BN'),
    '(Z)-2-Butenenitrile': ('Nitriles', '2-BN'),
    'Propanenitrile, 2-methyl-': ('Nitriles', 'I-BN'),

    # --- 5. Sulfides ---
    'Dimethyl sulfide': ('Sulfides', 'DMS'),
    'Divinyl sulfide': ('Sulfides', 'DVS'),
    'Ethyl propyl sulfide': ('Sulfides', 'EPS'),
    '3-Ethylthio-1-propene': ('Sulfides', 'ETP'),
    'Propane, 2-(ethylthio)-': ('Sulfides', '2-ETP'),

    # --- 6. Silanes ---
    '二甲基二乙氧基硅烷（DMS）': ('Silanes', 'DMDES'),
    '(2-氟乙氧基)三甲基硅烷（MFS）': ('Silanes', 'MFS'),

    # --- 7. Amides ---
    'N,N-二甲基三氟甲磺酰胺(DMTMSA)': ('Amides', 'DMTMSA'),
    'Methacrylamide': ('Amides', 'MAA'),

    # --- 8. Phosphates ---
    '三(2，2，2-三氟乙基)磷酸酯(TFEP)': ('Phosphates', 'TFEP'),

    # --- 9. Aromatics ---
    '六氟苯（HFB）': ('Aromatics', 'HFB'),
}

CATEGORY_ORDER = [
    'Ethers', 'Carbonates', 'Esters', 'Nitriles',
    'Sulfides', 'Silanes', 'Amides', 'Phosphates', 'Aromatics'
]

# ==========================================
# 3. 样式映射
# ==========================================
def rgb_to_mpl(rgb):
    return tuple([x/255.0 for x in rgb])

COLORS = [rgb_to_mpl(c) for c in CONFIG['colors_rgb']]
SHAPES = ['o', 's', '^', 'D', '*', 'ring']

STYLE_ASSIGNMENT = {
    'Ethers':      {'c_idx': 0, 'shape': 'o'},     # 粉色 圆圈
    'Carbonates':  {'c_idx': 1, 'shape': 's'},     # 蓝色 方块
    'Esters':      {'c_idx': 2, 'shape': '^'},     # 绿色 三角
    'Nitriles':    {'c_idx': 3, 'shape': 'D'},     # 紫色 菱形
    'Sulfides':    {'c_idx': 0, 'shape': 'o'},     # 粉色 五角星
    'Silanes':     {'c_idx': 1, 'shape': 'ring'},  # 蓝色 圆环
    'Amides':      {'c_idx': 2, 'shape': '*'},     # 绿色 圆圈
    'Phosphates':  {'c_idx': 3, 'shape': 's'},     # 紫色 方块
    'Aromatics':   {'c_idx': 0, 'shape': 'D'},     # 粉色 菱形
}

# ==========================================
# 4. 辅助函数
# ==========================================
def repel_labels(ax, x, y, labels, k=0.8, iter_count=80):
    tx = x + np.random.uniform(-1.5, 1.5, size=len(x))
    ty = y + np.random.uniform(-1.5, 1.5, size=len(y))

    for _ in range(iter_count):
        # 斥力
        for i in range(len(labels)):
            for j in range(len(labels)):
                if i == j: continue
                dx = tx[i] - tx[j]
                dy = ty[i] - ty[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < 4.0:
                    force = 1.0 / (dist + 0.1)
                    tx[i] += dx * force * 0.15
                    ty[i] += dy * force * 0.15
        # 弹力
        for i in range(len(labels)):
            dx = x[i] - tx[i]
            dy = y[i] - ty[i]
            dist = np.sqrt(dx*dx + dy*dy)
            target = 3.0
            if dist < target:
                 tx[i] -= dx * 0.05
                 ty[i] -= dy * 0.05
            else:
                 tx[i] += dx * 0.05
                 ty[i] += dy * 0.05

    texts = []
    for i in range(len(labels)):
        tx[i] = np.clip(tx[i], -2, 47)
        ty[i] = np.clip(ty[i], -2, 47)

        dist = np.sqrt((tx[i]-x[i])**2 + (ty[i]-y[i])**2)
        arrow_props = None
        if dist > 1.2:
            arrow_props = dict(arrowstyle='-', color='gray', alpha=0.4, lw=0.5)

        t = ax.annotate(labels[i],
                    xy=(x[i], y[i]),
                    xytext=(tx[i], ty[i]),
                    fontsize=CONFIG['label_fontsize'],
                    color='#333333',
                    alpha=0.9,
                    ha='center', va='center',
                    arrowprops=arrow_props)
        texts.append(t)

# ==========================================
# 5. 主程序
# ==========================================
try:
    df = pd.read_csv('prediction_ranking_result.csv')
except:
    print("未找到文件，请确保 'prediction_ranking_result(4).csv' 在当前目录下")
    exit()

def apply_map(name):
    return MOLECULE_MAP.get(name, ('Others', name[:4]))

df['Mapped'] = df['name'].apply(apply_map)
df['Category'] = df['Mapped'].apply(lambda x: x[0])
df['Label'] = df['Mapped'].apply(lambda x: x[1])

fig, ax = plt.subplots(figsize=CONFIG['figure_size'])

# 坐标轴设置 (翻转：48 -> -3)
start, end = CONFIG['axis_limit_start'], CONFIG['axis_limit_end']
ax.set_xlim(start, end)
ax.set_ylim(start, end)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.2)
ax.spines['bottom'].set_linewidth(1.2)

# --- 绘制背景 (Fixed Gradient Logic) ---
xx, yy = np.meshgrid(np.linspace(start, end, 200), np.linspace(start, end, 200))

# 计算归一化坐标 (0-1)
# X轴: Left(48) -> Right(-3).
# 我们希望 Right 是 High D (Norm=1). Left 是 Low D (Norm=0).
# Length = 48 - (-3) = 51.
# Norm_X = (48 - xx) / 51.0  => if xx=48, norm=0. if xx=-3, norm=1.
norm_x = (48 - xx) / 51.0
norm_y = (48 - yy) / 51.0
norm_x = np.clip(norm_x, 0, 1)
norm_y = np.clip(norm_y, 0, 1)

# 获取颜色配置
base = np.array(CONFIG['color_base'])
high_d = np.array(CONFIG['color_high_d'])
high_p = np.array(CONFIG['color_high_p'])

# 插值计算
# 这里的逻辑是：基础色 + X轴影响 + Y轴影响
# 权重根据 norm_x 和 norm_y
# 当靠近 High D (X=1, Y=0) -> 更多 High D 颜色
# 当靠近 High P (X=0, Y=1) -> 更多 High P 颜色
# 当在中间 (X=1, Y=1) -> 混合
# 当在左下 (X=0, Y=0) -> 基础色

# 简单的加权混合：
# result = base * (1-x)(1-y) + d * x(1-y) + p * (1-x)y + mix * xy ... 比较复杂
# 简化版：线性叠加
# RGB = Base + (TargetD - Base)*x + (TargetP - Base)*y
R = base[0] + (high_d[0] - base[0]) * norm_x + (high_p[0] - base[0]) * norm_y
G = base[1] + (high_d[1] - base[1]) * norm_x + (high_p[1] - base[1]) * norm_y
B = base[2] + (high_d[2] - base[2]) * norm_x + (high_p[2] - base[2]) * norm_y

bg = np.stack([R, G, B], axis=2)
bg = np.clip(bg, 0, 1) # 确保不超过1

ax.imshow(bg, extent=[start, end, start, end], origin='upper', aspect='auto', zorder=0)

# --- 绘制数据点 ---
all_x, all_y, all_labels = [], [], []

for cat in CATEGORY_ORDER:
    subset = df[df['Category'] == cat]
    if subset.empty: continue

    style = STYLE_ASSIGNMENT.get(cat, {'c_idx': 0, 'shape': 'o'})
    color = COLORS[style['c_idx']]
    shape = style['shape']

    x_vals = subset['Rank_D'].values
    y_vals = subset['Rank_P'].values

    if shape == 'ring':
        ax.scatter(x_vals, y_vals,
                   facecolors='none',
                   edgecolors=color,
                   linewidths=2.5,
                   marker='o',
                   s=CONFIG['marker_size'],
                   label=cat, zorder=10)
    else:
        ax.scatter(x_vals, y_vals,
                   c=[color],
                   marker=shape,
                   s=CONFIG['marker_size'],
                   edgecolors='none', # 无边框
                   label=cat, zorder=10)

    all_x.extend(x_vals)
    all_y.extend(y_vals)
    all_labels.extend(subset['Label'].tolist())

# --- 标签调整 ---
repel_labels(ax, np.array(all_x), np.array(all_y), all_labels)

# --- 装饰 ---
ax.set_xlabel('Rank D (Dipole Moment)', fontsize=12, fontweight='bold', labelpad=10)
ax.set_ylabel('Rank P (Polarizability)', fontsize=12, fontweight='bold', labelpad=10)
ax.set_title('Molecular Ranking Map', fontsize=16, fontweight='bold', pad=20)

legend = ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False,
                   fontsize=10, title="Category", labelspacing=1.2)
plt.setp(legend.get_title(), fontweight='bold')

plt.tight_layout()
plt.show()