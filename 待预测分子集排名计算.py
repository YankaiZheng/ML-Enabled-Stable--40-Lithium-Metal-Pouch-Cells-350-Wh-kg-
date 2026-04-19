import pandas as pd
import numpy as np
import joblib
import os
import warnings

warnings.filterwarnings('ignore')

# ===== 配置 =====
# 待预测的文件名 (只要有 q+, q-, HOMO, LUMO 列即可)
INPUT_FILE = 'data3.csv'
OUTPUT_FILE = 'prediction_ranking_result.csv'
MODEL_FILE_D = 'D_finetune_model.joblib'
MODEL_FILE_P = 'P_finetune_model.joblib'

# ===== 1. 检查环境 =====
print("=" * 60)
print("启动预测程序...")
print("=" * 60)

if not os.path.exists(MODEL_FILE_D) or not os.path.exists(MODEL_FILE_P):
    print(f"错误: 找不到模型文件 {MODEL_FILE_D} 或 {MODEL_FILE_P}")
    exit(1)

# ===== 2. 加载数据 =====
if not os.path.exists(INPUT_FILE):
    print(f"提示: 未找到 {INPUT_FILE}，正在生成示例文件...")
    # 生成假数据用于测试
    df = pd.DataFrame({
        'name': [f'New_Mol_{i}' for i in range(1, 11)],
        'q+': np.random.rand(10),
        'q-': -np.random.rand(10),
        'HOMO': np.random.uniform(-0.3, -0.2, 10),
        'LUMO': np.random.uniform(-0.05, 0.05, 10)
    })
    df.to_csv(INPUT_FILE, index=False)
    predict_df = df
    print(f">>> 已生成 {INPUT_FILE}")
else:
    predict_df = pd.read_csv(INPUT_FILE)

print(f"加载了 {len(predict_df)} 个待预测分子。")

# 检查列
req_cols = ['q+', 'q-', 'HOMO', 'LUMO']
if not all(col in predict_df.columns for col in req_cols):
    print(f"错误: 输入文件必须包含这些列: {req_cols}")
    exit(1)

# ===== 3. 加载模型并预测 =====
print("正在加载模型...")
d_pkg = joblib.load(MODEL_FILE_D)
p_pkg = joblib.load(MODEL_FILE_P)

# D 模型预测
print("正在预测 D 值...")
pipeline_D = d_pkg['pipeline']
scaler_D = d_pkg['target_scaler']
pred_d_scaled = pipeline_D.predict(predict_df[req_cols].values)
pred_d_real = scaler_D.inverse_transform(pred_d_scaled.reshape(-1, 1)).ravel()

# P 模型预测
print("正在预测 P 值...")
pipeline_P = p_pkg['pipeline']
scaler_P = p_pkg['target_scaler']
pred_p_scaled = pipeline_P.predict(predict_df[req_cols].values)
pred_p_real = scaler_P.inverse_transform(pred_p_scaled.reshape(-1, 1)).ravel()

# ===== 4. 生成排名与结果 =====
print("正在计算排名...")
result_df = predict_df.copy()
result_df['Predicted_D'] = pred_d_real
result_df['Predicted_P'] = pred_p_real

# 计算排名 (Rank 1 = 数值最大)
# 如果你需要数值越小越好，请改 ascending=True
result_df['Rank_D'] = result_df['Predicted_D'].rank(ascending=False, method='min').astype(int)
result_df['Rank_P'] = result_df['Predicted_P'].rank(ascending=False, method='min').astype(int)

# 排序输出 (默认按 D 排名)
result_df = result_df.sort_values('Rank_D')

# 打印 Top 5
print("\n" + "=" * 60)
print(">>> 预测结果 Top 5 (按 D 值排序):")
print(result_df[['name', 'Predicted_D', 'Rank_D', 'Predicted_P', 'Rank_P']].head(5).to_string(index=False))

# 保存
result_df.to_csv(OUTPUT_FILE, index=False)
print("\n" + "=" * 60)
print(f"完整结果已保存至: {OUTPUT_FILE}")
print("=" * 60)