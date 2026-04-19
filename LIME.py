import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import shap
from lime import lime_tabular
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

# ===== 1. 环境配置与参数 =====
FEATURE_COLS = ['q+', 'q-', 'HOMO', 'LUMO']
# DMTMSA 的特征值 (请根据你的最新计算结果微调)
DMTMSA_VALS = {'q+': 0.405, 'q-': -0.371, 'HOMO': -0.29, 'LUMO': -0.013}
# 模型文件名
MODELS = {'D': 'D_finetune_model.joblib', 'P': 'P_finetune_model.joblib'}

plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

# ===== 2. 核心分析类 =====
class ChemExplainer:
    def __init__(self, train_csv):
        self.df_train = pd.read_excel(train_csv)
        self.X_train = self.df_train[FEATURE_COLS].values

    def load_pipeline(self, model_path, target):
        """加载模型并重新拟合以确保 Pipeline 完整性"""
        checkpoint = joblib.load(model_path)
        y_train = self.df_train[target].values

        # 处理目标缩放
        self.scaler_y = StandardScaler().fit(y_train.reshape(-1, 1))
        y_scaled = self.scaler_y.transform(y_train.reshape(-1, 1)).ravel()

        # 提取超参数并构建模型
        params = {k.replace('regressor__', ''): v for k, v in checkpoint['best_params'].items()}
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', RandomForestRegressor(random_state=42, **params))
        ])
        pipeline.fit(self.X_train, y_scaled)
        return pipeline

    def run_lime(self, pipeline, instance, target_name, pred_val):
        """生成并绘制 LIME 解释"""
        def predict_fn(X):
            return self.scaler_y.inverse_transform(pipeline.predict(X).reshape(-1, 1)).ravel()

        explainer = lime_tabular.LimeTabularExplainer(
            self.X_train, feature_names=FEATURE_COLS, mode='regression', random_state=42
        )
        exp = explainer.explain_instance(instance, predict_fn, num_features=4)

        # 绘图
        fig = plt.figure(figsize=(8, 5))
        features, values = zip(*exp.as_list())
        colors = ['#27ae60' if v > 0 else '#e74c3c' for v in values]
        plt.barh(features, values, color=colors, alpha=0.8)
        plt.title(f"LIME: {target_name} Prediction (Pred: {pred_val:.4f})", fontweight='bold')
        plt.xlabel("Local Contribution")
        plt.grid(axis='x', linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(f'LIME_{target_name}.png', dpi=300)
        return exp

    def run_shap(self, pipeline, instance, target_name, pred_val):
        """生成并绘制 SHAP 贡献图"""
        rf_model = pipeline.named_steps['regressor']
        scaler_x = pipeline.named_steps['scaler']

        # SHAP 计算
        explainer = shap.TreeExplainer(rf_model)
        instance_scaled = scaler_x.transform(instance.reshape(1, -1))
        shap_values = explainer.shap_values(instance_scaled)

        # 逆缩放 SHAP 值到原始单位
        y_std = np.sqrt(self.scaler_y.var_[0])
        rescaled_shap = shap_values[0] * y_std

        # 绘图
        fig = plt.figure(figsize=(8, 5))
        # 构造带数值的特征标签
        labels = [f"{FEATURE_COLS[i]} ({instance[i]:.3f})" for i in range(len(FEATURE_COLS))]
        colors = ['#ff0051' if v > 0 else '#008bfb' for v in rescaled_shap]

        # 按绝对值排序显示
        indices = np.argsort(np.abs(rescaled_shap))
        plt.barh(np.array(labels)[indices], rescaled_shap[indices], color=np.array(colors)[indices])

        plt.axvline(x=0, color='black', lw=0.8)
        plt.title(f"SHAP: {target_name} Prediction (Pred: {pred_val:.4f})", fontweight='bold')
        plt.xlabel("SHAP Value (Contribution to Prediction)")
        plt.tight_layout()
        plt.savefig(f'SHAP_{target_name}.png', dpi=300)
        return rescaled_shap

# ===== 3. 执行主程序 =====
if __name__ == "__main__":
    # 初始化分析器
    try:
        analyzer = ChemExplainer('data2.xlsx')
        instance = np.array([DMTMSA_VALS[c] for c in FEATURE_COLS])

        for target, model_path in MODELS.items():
            print(f"--- 正在分析目标: {target} ---")

            # 1. 加载与预测
            pipe = analyzer.load_pipeline(model_path, target)
            pred_scaled = pipe.predict(instance.reshape(1, -1))
            pred_raw = analyzer.scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))[0][0]

            # 2. LIME 分析
            analyzer.run_lime(pipe, instance, target, pred_raw)

            # 3. SHAP 分析
            analyzer.run_shap(pipe, instance, target, pred_raw)

        print("\n分析完成！图片已保存至当前目录。")

    except FileNotFoundError:
        print("错误：请确保 'data2.csv' 和模型文件在当前目录下。")