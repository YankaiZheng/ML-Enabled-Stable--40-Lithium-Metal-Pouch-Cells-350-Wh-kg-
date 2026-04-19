import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from scipy.stats import spearmanr, pearsonr, rankdata
import joblib
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter

warnings.filterwarnings('ignore')

# ===== 配置参数 =====
RANDOM_SEED = 42
MODEL_FILE_D = 'D_finetune_model.joblib'
MODEL_FILE_P = 'P_finetune_model.joblib'

# ===== 分段规则配置 =====
SEGMENT_CONFIG = {
    'top_n': 10,           # 前多少名作为"顶部区域"
    'bottom_n': 10,        # 后多少名作为"底部区域"
    'top_space': 30,       # 顶部区域占用的空间百分比 (0-100)
    'middle_space': 40,    # 中间区域占用的空间百分比 (0-100)
    'bottom_space': 30,    # 底部区域占用的空间百分比 (0-100)
}

# ===== 误差边界线配置 =====
ERROR_BANDS_CONFIG = {
    'show_bands': False,  # 关闭误差线
    'global_bands': [10, 15],
    'band_colors': {
        10: '#3498db',
        15: '#95a5a6',
    },
    'band_linestyle': '-',
    'band_linewidth': 1.5,
}

# ===== Nature/Origin 风格配置 =====
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.major.width'] = 1.2
plt.rcParams['ytick.major.width'] = 1.2
plt.rcParams['xtick.major.size'] = 5
plt.rcParams['ytick.major.size'] = 5
plt.rcParams['axes.grid'] = False  # 关闭网格

# ===== 颜色配置 =====
def rgb_to_hex(rgb):
    """将RGB元组转换为十六进制颜色"""
    return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))

COLORS = {
    'best': rgb_to_hex((79, 119, 158)),      # 深蓝（最准）
    'good': rgb_to_hex((145, 180, 209)),     # 天蓝
    'okay': rgb_to_hex((182, 204, 185)),     # 豆绿
    'fair': rgb_to_hex((180, 194, 186)),     # 灰绿
    'poor': rgb_to_hex((199, 180, 190)),     # 藕荷
    'worst': rgb_to_hex((225, 172, 166)),    # 桃粉（最差）
}

# ==========================================
# 分段坐标轴 (Segmented Axis) - 局部放大图
# ==========================================
def segmented_transform(rank, n_samples, config=SEGMENT_CONFIG):
    """
    分段坐标变换：可自定义前N名、中间、后N名的空间分配

    参数:
        rank: 排名数组
        n_samples: 样本总数
        config: 分段配置字典
    """
    rank = np.array(rank)
    transformed = np.zeros_like(rank, dtype=float)

    top_n = config['top_n']
    bottom_n = config['bottom_n']
    top_space = config['top_space']
    middle_space = config['middle_space']
    bottom_space = config['bottom_space']

    # 计算边界位置
    boundary1 = top_space
    boundary2 = top_space + middle_space

    # 前N名：映射到0-boundary1
    mask1 = rank <= top_n
    if top_n > 1:
        transformed[mask1] = (rank[mask1] - 1) * (top_space / (top_n - 1))
    else:
        transformed[mask1] = 0

    # 中间部分：映射到boundary1-boundary2
    middle_count = n_samples - top_n - bottom_n
    mask2 = (rank > top_n) & (rank <= (n_samples - bottom_n))
    if middle_count > 0:
        transformed[mask2] = boundary1 + (rank[mask2] - top_n) * (middle_space / middle_count)

    # 后N名：映射到boundary2-100
    mask3 = rank > (n_samples - bottom_n)
    if bottom_n > 0:
        transformed[mask3] = boundary2 + (rank[mask3] - (n_samples - bottom_n)) * (bottom_space / bottom_n)

    return transformed

def inverse_segmented_transform(transformed, n_samples, config=SEGMENT_CONFIG):
    """反变换：从变换后的坐标恢复原始排名"""
    transformed = np.array(transformed)
    rank = np.zeros_like(transformed, dtype=float)

    top_n = config['top_n']
    bottom_n = config['bottom_n']
    top_space = config['top_space']
    middle_space = config['middle_space']
    bottom_space = config['bottom_space']

    boundary1 = top_space
    boundary2 = top_space + middle_space
    middle_count = n_samples - top_n - bottom_n

    # 前N名区域
    mask1 = transformed <= boundary1
    if top_n > 1:
        rank[mask1] = (transformed[mask1] * (top_n - 1) / top_space) + 1
    else:
        rank[mask1] = 1

    # 中间区域
    mask2 = (transformed > boundary1) & (transformed <= boundary2)
    if middle_count > 0:
        rank[mask2] = ((transformed[mask2] - boundary1) * middle_count / middle_space) + top_n

    # 后N名区域
    mask3 = transformed > boundary2
    if bottom_n > 0:
        rank[mask3] = ((transformed[mask3] - boundary2) * bottom_n / bottom_space) + (n_samples - bottom_n)

    return rank

def draw_error_bands(ax, n_samples, config=SEGMENT_CONFIG, error_config=ERROR_BANDS_CONFIG):
    """绘制全局误差边界线"""
    if not error_config['show_bands']:
        return

    top_n = config['top_n']
    bottom_n = config['bottom_n']

    for error in error_config['global_bands']:
        color = error_config['band_colors'].get(error, '#95a5a6')

        # 分三段绘制
        ranks_top = np.linspace(1, top_n, 100)
        x_top = segmented_transform(ranks_top, n_samples, config)
        y_upper_top = []
        y_lower_top = []
        for rank_true in ranks_top:
            rank_pred_upper = min(rank_true + error, n_samples)
            rank_pred_lower = max(rank_true - error, 1)
            y_u = segmented_transform([rank_pred_upper], n_samples, config)[0]
            y_l = segmented_transform([rank_pred_lower], n_samples, config)[0]
            y_upper_top.append(y_u)
            y_lower_top.append(y_l)

        if n_samples - bottom_n > top_n:
            ranks_mid = np.linspace(top_n + 1, n_samples - bottom_n, 100)
            x_mid = segmented_transform(ranks_mid, n_samples, config)
            y_upper_mid = []
            y_lower_mid = []
            for rank_true in ranks_mid:
                rank_pred_upper = min(rank_true + error, n_samples)
                rank_pred_lower = max(rank_true - error, 1)
                y_u = segmented_transform([rank_pred_upper], n_samples, config)[0]
                y_l = segmented_transform([rank_pred_lower], n_samples, config)[0]
                y_upper_mid.append(y_u)
                y_lower_mid.append(y_l)
        else:
            x_mid, y_upper_mid, y_lower_mid = [], [], []

        ranks_bottom = np.linspace(n_samples - bottom_n + 1, n_samples, 100)
        x_bottom = segmented_transform(ranks_bottom, n_samples, config)
        y_upper_bottom = []
        y_lower_bottom = []
        for rank_true in ranks_bottom:
            rank_pred_upper = min(rank_true + error, n_samples)
            rank_pred_lower = max(rank_true - error, 1)
            y_u = segmented_transform([rank_pred_upper], n_samples, config)[0]
            y_l = segmented_transform([rank_pred_lower], n_samples, config)[0]
            y_upper_bottom.append(y_u)
            y_lower_bottom.append(y_l)

        x_all = np.concatenate([x_top, x_mid, x_bottom])
        y_upper_all = np.concatenate([y_upper_top, y_upper_mid, y_upper_bottom])
        y_lower_all = np.concatenate([y_lower_top, y_lower_mid, y_lower_bottom])

        # 只画线，不填充
        ax.plot(x_all, y_upper_all, color=color,
               linestyle=error_config['band_linestyle'],
               linewidth=error_config['band_linewidth'], alpha=0.8, zorder=0.5,
               label=f'±{error} rank error')
        ax.plot(x_all, y_lower_all, color=color,
               linestyle=error_config['band_linestyle'],
               linewidth=error_config['band_linewidth'], alpha=0.8, zorder=0.5)

def plot_rank_scatter_segmented(y_true, y_pred, model_name,
                                figsize=(10, 8), region_linewidth=1.5,
                                config=SEGMENT_CONFIG,
                                error_config=ERROR_BANDS_CONFIG):
    """分段坐标轴散点图（局部放大图）"""
    # 计算排名
    true_ranks = rankdata(y_true)
    pred_ranks = rankdata(y_pred)
    rank_diff = np.abs(true_ranks - pred_ranks)

    # 计算相关系数
    rho, _ = spearmanr(y_true, y_pred)
    pcc, _ = pearsonr(y_true, y_pred)

    n_samples = len(y_true)

    # 提取配置参数
    top_n = config['top_n']
    bottom_n = config['bottom_n']
    top_space = config['top_space']
    middle_space = config['middle_space']
    bottom_space = config['bottom_space']

    boundary1 = top_space
    boundary2 = top_space + middle_space

    # 变换坐标
    true_ranks_trans = segmented_transform(true_ranks, n_samples, config)
    pred_ranks_trans = segmented_transform(pred_ranks, n_samples, config)

    # 根据排名差异设置颜色
    colors = []
    for diff in rank_diff:
        if diff <= 5:
            colors.append(COLORS['best'])
        elif diff <= 10:
            colors.append(COLORS['good'])
        elif diff <= 15:
            colors.append(COLORS['okay'])
        elif diff <= 20:
            colors.append(COLORS['fair'])
        elif diff <= 25:
            colors.append(COLORS['poor'])
        else:
            colors.append(COLORS['worst'])

    # 修复：正确统计准确性（包含临界值）
    # Top N: 真实排名 <= top_n 的样本中，预测排名 <= top_n 的数量
    top_mask = true_ranks <= top_n
    top_pred_correct = pred_ranks[top_mask] <= top_n
    top_accurate = np.sum(top_pred_correct)

    # Bottom N: 真实排名 > (n_samples - bottom_n) 的样本中，预测排名 > (n_samples - bottom_n) 的数量
    bottom_mask = true_ranks > (n_samples - bottom_n)
    bottom_pred_correct = pred_ranks[bottom_mask] > (n_samples - bottom_n)
    bottom_accurate = np.sum(bottom_pred_correct)

    # 创建图形
    fig, ax = plt.subplots(figsize=figsize)

    # 确保没有背景网格
    ax.grid(False)

    # 绘制误差边界线（在散点之前）
    draw_error_bands(ax, n_samples, config, error_config)

    # 绘制对角线
    diagonal_x = np.linspace(0, 100, 100)
    diagonal_y = diagonal_x
    ax.plot(diagonal_x, diagonal_y, 'k--', lw=2, alpha=0.6, zorder=2, label='Perfect Prediction')

    # 前N名区域边界
    ax.axvline(x=boundary1, ymin=0, ymax=boundary1/100, color='#27ae60',
               linewidth=region_linewidth, linestyle='-', zorder=1)
    ax.axhline(y=boundary1, xmin=0, xmax=boundary1/100, color='#27ae60',
               linewidth=region_linewidth, linestyle='-', zorder=1)
    ax.fill_between([0, boundary1], [0, 0], [boundary1, boundary1],
                    color='#d5f4e6', alpha=0.25, zorder=0)

    # 后N名区域边界
    ax.axvline(x=boundary2, ymin=boundary2/100, ymax=1, color='#c0392b',
               linewidth=region_linewidth, linestyle='-', zorder=1)
    ax.axhline(y=boundary2, xmin=boundary2/100, xmax=1, color='#c0392b',
               linewidth=region_linewidth, linestyle='-', zorder=1)
    ax.fill_between([boundary2, 100], [boundary2, boundary2], [100, 100],
                    color='#fadbd8', alpha=0.25, zorder=0)

    # 绘制散点
    scatter = ax.scatter(true_ranks_trans, pred_ranks_trans, c=colors, s=100, alpha=0.75,
                         edgecolors='white', linewidths=1.5, zorder=3)

    # 自定义坐标轴标签
    def format_axis(x, pos):
        if x <= boundary1:
            return f'{int(inverse_segmented_transform([x], n_samples, config)[0])}'
        elif x <= boundary2:
            val = inverse_segmented_transform([x], n_samples, config)[0]
            if abs(val - round(val)) < 0.1 or pos in [0, 1, 2, 3, 4]:
                return f'{int(round(val))}'
            return ''
        else:
            return f'{int(inverse_segmented_transform([x], n_samples, config)[0])}'

    ax.xaxis.set_major_formatter(FuncFormatter(format_axis))
    ax.yaxis.set_major_formatter(FuncFormatter(format_axis))

    # 设置刻度位置
    tick_positions = [0, boundary1/2, boundary1, (boundary1+boundary2)/2,
                     boundary2, (boundary2+100)/2, 100]
    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)

    # 标签和标题
    ax.set_xlabel('True Rank (Segmented Scale)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Predicted Rank (Segmented Scale)', fontsize=13, fontweight='bold')
    ax.set_title(f'Model {model_name}: Rank Prediction (Segmented Axis)\n' +
                 f'Spearman ρ = {rho:.3f}  |  Pearson r = {pcc:.3f}\n' +
                 f'Segments: Top{top_n}({top_space}%) - Middle({middle_space}%) - Bottom{bottom_n}({bottom_space}%)',
                 fontsize=14, fontweight='bold', pad=20)

    # 图例
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['best'],
                   markersize=10, label='Excellent (Δ ≤ 5)', markeredgewidth=1.5, markeredgecolor='white'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['good'],
                   markersize=10, label='Good (5 < Δ ≤ 10)', markeredgewidth=1.5, markeredgecolor='white'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['okay'],
                   markersize=10, label='Okay (10 < Δ ≤ 15)', markeredgewidth=1.5, markeredgecolor='white'),
        plt.Line2D([0], [0], color='k', linestyle='--', linewidth=2, label='Perfect Prediction'),
        plt.Line2D([0], [0], color='white', linewidth=0, label=''),
        mpatches.Patch(facecolor='#d5f4e6', edgecolor='#27ae60', linewidth=2,
                      label=f'Top {top_n}: {top_accurate}/{top_n} accurate'),
        mpatches.Patch(facecolor='#fadbd8', edgecolor='#c0392b', linewidth=2,
                      label=f'Bottom {bottom_n}: {bottom_accurate}/{bottom_n} accurate'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', frameon=False, fontsize=9.5, labelspacing=0.4)

    ax.set_xlim(-2, 102)
    ax.set_ylim(-2, 102)

    plt.tight_layout()

    # 保存时在文件名中体现配置
    config_str = f"T{top_n}_M_B{bottom_n}_{top_space}-{middle_space}-{bottom_space}"
    plt.savefig(f'{model_name}_Segmented_{config_str}.png', dpi=600, bbox_inches='tight')
    plt.savefig(f'{model_name}_Segmented_{config_str}.pdf', bbox_inches='tight')
    plt.show()

    print(f"✓ {model_name} 分段坐标轴图已保存")
    print(f"  - 配置: Top{top_n}({top_space}%) - Middle({middle_space}%) - Bottom{bottom_n}({bottom_space}%)")
    print(f"  - Top {top_n}准确率: {top_accurate}/{top_n} ({100*top_accurate/top_n:.1f}%)")
    print(f"  - Bottom {bottom_n}准确率: {bottom_accurate}/{bottom_n} ({100*bottom_accurate/bottom_n:.1f}%)")


# ==========================================
# 主程序
# ==========================================
def main():
    print("="*60)
    print("局部放大图生成器（可调参数版本）")
    print("="*60)

    # 打印当前分段配置
    print("\n当前分段配置:")
    print(f"  - 前 {SEGMENT_CONFIG['top_n']} 名占用 {SEGMENT_CONFIG['top_space']}% 空间")
    print(f"  - 中间部分占用 {SEGMENT_CONFIG['middle_space']}% 空间")
    print(f"  - 后 {SEGMENT_CONFIG['bottom_n']} 名占用 {SEGMENT_CONFIG['bottom_space']}% 空间")

    # 打印误差带配置
    if ERROR_BANDS_CONFIG['show_bands']:
        print("\n误差边界线配置:")
        print(f"  - 全局误差带: ±{ERROR_BANDS_CONFIG['global_bands']}")
    else:
        print("\n误差边界线: 已关闭")

    # 验证配置
    total_space = SEGMENT_CONFIG['top_space'] + SEGMENT_CONFIG['middle_space'] + SEGMENT_CONFIG['bottom_space']
    if total_space != 100:
        print(f"\n⚠️  警告: 空间分配总和为 {total_space}%，不等于100%！")
        print("请调整 SEGMENT_CONFIG 使总和为100%")
        return

    print("\n正在加载数据...")
    train_data = pd.read_excel('data2.xlsx')
    test_data = pd.read_excel('data4.xlsx')

    feature_cols = ['q+', 'q-', 'HOMO', 'LUMO']
    targets = ['D', 'P']

    X_train = train_data[feature_cols].values
    X_test = test_data[feature_cols].values

    model_files = {'D': MODEL_FILE_D, 'P': MODEL_FILE_P}

    # ===== 可调整参数 =====
    FIGSIZE = (10, 8)
    REGION_LINEWIDTH = 1.5

    for target in targets:
        print(f"\n{'='*60}")
        print(f"处理目标: {target}")
        print(f"{'='*60}")

        Y_train = train_data[target].values
        Y_test = test_data[target].values

        scaler = StandardScaler()
        Y_train_scaled = scaler.fit_transform(Y_train.reshape(-1, 1)).ravel()

        model_file = model_files[target]

        if joblib.os.path.exists(model_file):
            checkpoint = joblib.load(model_file)
            best_params = checkpoint['best_params']
            rf_params = {k.replace('regressor__', ''): v for k, v in best_params.items()}

            print("在全量训练集上重新训练...")
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('regressor', RandomForestRegressor(random_state=RANDOM_SEED, n_jobs=-1, **rf_params))
            ])
            pipeline.fit(X_train, Y_train_scaled)

            print("预测测试集...")
            Y_pred_scaled = pipeline.predict(X_test)
            Y_pred_real = scaler.inverse_transform(Y_pred_scaled.reshape(-1, 1)).ravel()

            # 计算排名
            true_ranks = rankdata(Y_test)
            pred_ranks = rankdata(Y_pred_real)

            # 计算并打印Top-K准确率（从3到20）
            print(f"\n{'='*70}")
            print(f"模型 {target} - Top-K 准确率统计 (K=3到20)")
            print(f"{'='*70}")
            print(f"{'K值':<10} {'正确数/总数':<20} {'准确率':<15}")
            print(f"{'-'*70}")

            for k in range(3, 21):
                if k > len(Y_test):
                    break
                top_mask = true_ranks <= k
                top_pred_correct = pred_ranks[top_mask] <= k
                top_accurate = np.sum(top_pred_correct)
                top_total = np.sum(top_mask)
                if top_total > 0:
                    acc = 100.0 * top_accurate / top_total
                    print(f"Top-{k:<5} {top_accurate}/{top_total:<15} {acc:>6.2f}%")

            print(f"{'='*70}\n")

            # 计算并打印倒数Top-K准确率
            print(f"\n{'='*70}")
            print(f"模型 {target} - 倒数Top-K 准确率统计 (K=3到20)")
            print(f"{'='*70}")
            print(f"{'K值':<10} {'正确数/总数':<20} {'准确率':<15}")
            print(f"{'-'*70}")

            for k in range(3, 21):
                if k > len(Y_test):
                    break
                bottom_mask = true_ranks > (len(Y_test) - k)
                bottom_pred_correct = pred_ranks[bottom_mask] > (len(Y_test) - k)
                bottom_accurate = np.sum(bottom_pred_correct)
                bottom_total = np.sum(bottom_mask)
                if bottom_total > 0:
                    acc = 100.0 * bottom_accurate / bottom_total
                    print(f"Bottom-{k:<2} {bottom_accurate}/{bottom_total:<15} {acc:>6.2f}%")

            print(f"{'='*70}\n")

            # 生成分段坐标轴图
            print("\n生成局部放大图（分段坐标轴）...")
            plot_rank_scatter_segmented(Y_test, Y_pred_real, target,
                                       figsize=FIGSIZE,
                                       region_linewidth=REGION_LINEWIDTH,
                                       config=SEGMENT_CONFIG,
                                       error_config=ERROR_BANDS_CONFIG)

        else:
            print(f"错误: 找不到模型文件 {model_file}")

    print("\n" + "="*60)
    print("✓ 所有图表生成完成！")
    print("="*60)

if __name__ == "__main__":
    main()