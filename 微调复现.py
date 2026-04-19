import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from scipy.stats import spearmanr, kendalltau
import joblib
import warnings
import multiprocessing
import os
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# ===== 配置参数 =====
RANDOM_SEED = 42
N_JOBS = -1  # 最大化CPU利用

# 模型文件名
MODEL_FILE_D = 'D_finetune_model.joblib'
MODEL_FILE_P = 'P_finetune_model.joblib'

# 绘图设置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']  # 防止中文乱码
plt.rcParams['axes.unicode_minus'] = False

np.random.seed(RANDOM_SEED)

print(f"CPU核心数: {multiprocessing.cpu_count()}")
print(f"使用并行数: {N_JOBS} (全部核心)\n")


# ===== 自动加载最优超参数 =====
print("=" * 60)
print("自动读取最优超参数...")
print("=" * 60)

if os.path.exists(MODEL_FILE_D):
    print(f"正在读取 {MODEL_FILE_D} ...")
    d_checkpoint = joblib.load(MODEL_FILE_D)
    BEST_PARAMS_D = d_checkpoint['best_params']
    print(">>> D 模型最优参数已加载")
else:
    print(f"错误: 找不到文件 {MODEL_FILE_D}")
    print("请先运行微调脚本生成模型文件，或检查文件名。")
    exit(1)

if os.path.exists(MODEL_FILE_P):
    print(f"正在读取 {MODEL_FILE_P} ...")
    p_checkpoint = joblib.load(MODEL_FILE_P)
    BEST_PARAMS_P = p_checkpoint['best_params']
    print(">>> P 模型最优参数已加载")
else:
    print(f"错误: 找不到文件 {MODEL_FILE_P}")
    exit(1)


# ===== 1. 加载数据 =====
print("\n" + "=" * 60)
print("加载数据...")
print("=" * 60)
train_data = pd.read_excel('data2.xlsx')
test_data = pd.read_excel('data4.xlsx')
print(f"训练集: {train_data.shape}")
print(f"测试集: {test_data.shape}")

if 'name' not in train_data.columns or 'name' not in test_data.columns:
    print("警告: 数据中没有'name'列，将使用索引作为标识")
    train_data['name'] = [f"Train_{i}" for i in range(len(train_data))]
    test_data['name'] = [f"Test_{i}" for i in range(len(test_data))]


# ===== 2. 特征和目标提取 =====
feature_cols = ['q+', 'q-', 'HOMO', 'LUMO']
target_cols = ['D', 'P']
print(f"\n特征列: {feature_cols}")
print(f"目标列: {target_cols}")

X_train = train_data[feature_cols].values
X_test = test_data[feature_cols].values

Y_train_D = train_data['D'].values
Y_train_P = train_data['P'].values
Y_test_D = test_data['D'].values
Y_test_P = test_data['P'].values


# ===== 3. 准备数据 =====
print("\n准备数据...")
# 定义标准化器 (后续绘图逆变换需要用到)
scaler_D = StandardScaler()
Y_train_D_scaled = scaler_D.fit_transform(Y_train_D.reshape(-1, 1)).ravel()
Y_test_D_scaled = scaler_D.transform(Y_test_D.reshape(-1, 1)).ravel()

scaler_P = StandardScaler()
Y_train_P_scaled = scaler_P.fit_transform(Y_train_P.reshape(-1, 1)).ravel()
Y_test_P_scaled = scaler_P.transform(Y_test_P.reshape(-1, 1)).ravel()

print("注意: 使用Pipeline在独立标准化特征，避免数据泄露")


# ===== 4. 训练 D 模型 =====
print("\n" + "=" * 60)
print("训练 D 模型")
print("=" * 60)


rf_params_D = {k.replace('regressor__', ''): v for k, v in BEST_PARAMS_D.items()}

# 在全部训练集上训练最终模型
print("\n在全部训练集上训练最终模型...")
pipeline_D_final = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', RandomForestRegressor(random_state=RANDOM_SEED, n_jobs=-1, **rf_params_D))
])
pipeline_D_final.fit(X_train, Y_train_D_scaled)

# 分别预测训练集和测试集
Y_pred_D_train = pipeline_D_final.predict(X_train)
Y_pred_D_test = pipeline_D_final.predict(X_test)

# 计算 Spearman 与 Kendall
sp_D_train = spearmanr(Y_train_D_scaled, Y_pred_D_train)[0]
sp_D_test = spearmanr(Y_test_D_scaled, Y_pred_D_test)[0]
kd_D_test = kendalltau(Y_test_D_scaled, Y_pred_D_test)[0]

print(f"训练集 Spearman (拟合): {sp_D_train:.4f}")
print(f"测试集 Spearman (预测): {sp_D_test:.4f}")
print(f"测试集 Kendall  (预测): {kd_D_test:.4f}")


# ===== 5. 训练 P 模型 =====
print("\n" + "=" * 60)
print("训练 P 模型")
print("=" * 60)


rf_params_P = {k.replace('regressor__', ''): v for k, v in BEST_PARAMS_P.items()}

print("\n在全部训练集上训练最终模型...")
pipeline_P_final = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', RandomForestRegressor(random_state=RANDOM_SEED, n_jobs=-1, **rf_params_P))
])
pipeline_P_final.fit(X_train, Y_train_P_scaled)

# 分别预测训练集和测试集
Y_pred_P_train = pipeline_P_final.predict(X_train)
Y_pred_P_test = pipeline_P_final.predict(X_test)

# 计算 Spearman 与 Kendall
sp_P_train = spearmanr(Y_train_P_scaled, Y_pred_P_train)[0]
sp_P_test = spearmanr(Y_test_P_scaled, Y_pred_P_test)[0]
kd_P_test = kendalltau(Y_test_P_scaled, Y_pred_P_test)[0]

print(f"训练集 Spearman (拟合): {sp_P_train:.4f}")
print(f"测试集 Spearman (预测): {sp_P_test:.4f}")
print(f"测试集 Kendall  (预测): {kd_P_test:.4f}")


# ===== 6. 绘图: 实际值 vs 预测值 (含训练集和测试集) =====
print("\n" + "=" * 60)
print("正在生成预测散点图 (Actual vs Predicted)...")
print("=" * 60)

sns.set_style("whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# --- 绘制 D 模型 ---
ax = axes[0]
y_train_d_real = scaler_D.inverse_transform(Y_train_D_scaled.reshape(-1, 1)).ravel()
y_pred_d_train_real = scaler_D.inverse_transform(Y_pred_D_train.reshape(-1, 1)).ravel()
y_test_d_real = scaler_D.inverse_transform(Y_test_D_scaled.reshape(-1, 1)).ravel()
y_pred_d_test_real = scaler_D.inverse_transform(Y_pred_D_test.reshape(-1, 1)).ravel()

ax.scatter(y_train_d_real, y_pred_d_train_real, color='royalblue', alpha=0.5, label=f'Train (Sp={sp_D_train:.3f})', s=30)
ax.scatter(y_test_d_real, y_pred_d_test_real, color='crimson', alpha=0.7, label=f'Test (Sp={sp_D_test:.3f})', s=50, marker='^')

all_min = min(y_train_d_real.min(), y_test_d_real.min())
all_max = max(y_train_d_real.max(), y_test_d_real.max())
ax.plot([all_min, all_max], [all_min, all_max], 'k--', lw=2, label='Perfect Fit')

ax.set_title('Model D: Actual vs Predicted', fontsize=14, fontweight='bold')
ax.set_xlabel('Actual D Value', fontsize=12)
ax.set_ylabel('Predicted D Value', fontsize=12)
ax.legend(fontsize=10)

# --- 绘制 P 模型 ---
ax = axes[1]
y_train_p_real = scaler_P.inverse_transform(Y_train_P_scaled.reshape(-1, 1)).ravel()
y_pred_p_train_real = scaler_P.inverse_transform(Y_pred_P_train.reshape(-1, 1)).ravel()
y_test_p_real = scaler_P.inverse_transform(Y_test_P_scaled.reshape(-1, 1)).ravel()
y_pred_p_test_real = scaler_P.inverse_transform(Y_pred_P_test.reshape(-1, 1)).ravel()

ax.scatter(y_train_p_real, y_pred_p_train_real, color='royalblue', alpha=0.5, label=f'Train (Sp={sp_P_train:.3f})', s=30)
ax.scatter(y_test_p_real, y_pred_p_test_real, color='crimson', alpha=0.7, label=f'Test (Sp={sp_P_test:.3f})', s=50, marker='^')

all_min_p = min(y_train_p_real.min(), y_test_p_real.min())
all_max_p = max(y_train_p_real.max(), y_test_p_real.max())
ax.plot([all_min_p, all_max_p], [all_min_p, all_max_p], 'k--', lw=2, label='Perfect Fit')

ax.set_title('Model P: Actual vs Predicted', fontsize=14, fontweight='bold')
ax.set_xlabel('Actual P Value', fontsize=12)
ax.set_ylabel('Predicted P Value', fontsize=12)
ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig('prediction_scatter_spearman.png', dpi=300)
print(">>> 图表已保存为: prediction_scatter_spearman.png")
plt.show()


# ===== 7. 输出结果 =====
print("\n" + "=" * 60)
print("最终结果汇总")
print("=" * 60)

print("\n【D 模型】")
print(f"训练集 Spearman: {sp_D_train:.4f}")
print(f"测试集 Spearman: {sp_D_test:.4f}")
print(f"测试集 Kendall  : {kd_D_test:.4f}")

print("\n【P 模型】")
print(f"训练集 Spearman: {sp_P_train:.4f}")
print(f"测试集 Spearman: {sp_P_test:.4f}")
print(f"测试集 Kendall  : {kd_P_test:.4f}")


# ===== 8. 保存模型 =====
print("\n" + "=" * 60)
print("保存模型...")
print("=" * 60)

joblib.dump({
    'pipeline': pipeline_D_final,
    'best_params': BEST_PARAMS_D,
    'target_scaler': scaler_D,
    'feature_names': feature_cols,
    'train_spearman': sp_D_train,
    'test_spearman': sp_D_test,
    'test_kendall': kd_D_test
}, 'D_reproduce_model.joblib')
print("已保存: D_reproduce_model.joblib")

joblib.dump({
    'pipeline': pipeline_P_final,
    'best_params': BEST_PARAMS_P,
    'target_scaler': scaler_P,
    'feature_names': feature_cols,
    'train_spearman': sp_P_train,
    'test_spearman': sp_P_test,
    'test_kendall': kd_P_test
}, 'P_reproduce_model.joblib')
print("已保存: P_reproduce_model.joblib")

print("\n" + "=" * 60)
print("完成！")