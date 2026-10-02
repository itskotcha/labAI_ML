import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager as fm

# =========================================================
# 1. การตั้งค่าฟอนต์ภาษาไทย และการเตรียมข้อมูล
# =========================================================
font_name = 'Tahoma' 
plt.rcParams['font.family'] = font_name
plt.rcParams['axes.unicode_minus'] = False

# ข้อมูลโมเดลจากตารางในใบงาน
models = ['K-Nearest Neighbors', 'Gradient Boosting', 'Multi-Layer Perceptron', 
          'CatBoost', 'Logistic Regression', 'SVM (RBF Kernel)', 
          'LightGBM', 'XGBoost', 'Random Forest']

# ดึงค่า Performance Metrics ทั้ง 4 ตัว
f1_scores = np.array([0.8972, 0.8955, 0.8938, 0.8933, 0.8916, 0.8916, 0.8905, 0.8900, 0.8895])
roc_auc = np.array([0.9904, 0.9913, 0.9901, 0.9905, 0.9907, 0.9867, 0.9903, 0.9900, 0.9906])
pr_auc = np.array([0.8370, 0.8687, 0.8110, 0.8462, 0.8503, 0.7571, 0.8537, 0.8489, 0.8409])
mcc = np.array([0.8908, 0.8895, 0.8883, 0.8872, 0.8861, 0.8861, 0.8840, 0.8829, 0.8803])


# =========================================================
# 2. การสร้างกราฟที่ 1: กราฟเส้น (Multiple Line Chart)
# =========================================================
plt.figure(1, figsize=(12, 7))

plt.plot(models, f1_scores, marker='o', linestyle='-', color='skyblue', linewidth=2, label='F1-Score')
plt.plot(models, roc_auc, marker='s', linestyle='--', color='salmon', linewidth=2, label='ROC-AUC')
plt.plot(models, pr_auc, marker='^', linestyle='-.', color='lightgreen', linewidth=2, label='PR-AUC')
plt.plot(models, mcc, marker='D', linestyle=':', color='mediumpurple', linewidth=2, label='MCC')

plt.title('Performance Comparison of Machine Learning Models (Line Chart)', fontsize=16)
plt.xlabel('Machine Learning Models', fontsize=12)
plt.ylabel('Scores', fontsize=12)
plt.xticks(rotation=30, ha='right')  # เอียงชื่อโมเดล 30 องศาเพื่อไม่ให้ทับกัน
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=10, loc='lower left')
plt.tight_layout()


# =========================================================
# 3. การสร้างกราฟที่ 2: กราฟแท่งกลุ่ม (Grouped Bar Chart)
# =========================================================
plt.figure(2, figsize=(14, 7))

x = np.arange(len(models))  # กำหนดตำแหน่งแกน X สำหรับกลุ่มของโมเดล
width = 0.2  # ความกว้างของแต่ละแท่งข้อมูล

# พล็อตแท่งข้อมูลขยับตำแหน่งตามความกว้างเพื่อไม่ให้ทับกัน
plt.bar(x - 1.5 * width, f1_scores, width, label='F1-Score', color='skyblue')
plt.bar(x - 0.5 * width, roc_auc, width, label='ROC-AUC', color='salmon')
plt.bar(x + 0.5 * width, pr_auc, width, label='PR-AUC', color='lightgreen')
plt.bar(x + 1.5 * width, mcc, width, label='MCC', color='mediumpurple')

plt.title('Performance Comparison of Machine Learning Models (Bar Chart)', fontsize=16)
plt.xlabel('Machine Learning Models', fontsize=12)
plt.ylabel('Scores', fontsize=12)
plt.xticks(x, models, rotation=30, ha='right')
plt.ylim(0.7, 1.05)  # ปรับ Range แกน Y ให้เห็นความแตกต่างของแท่งกราฟชัดขึ้น
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.legend(fontsize=11, loc='upper right')
plt.tight_layout()


# =========================================================
# 4. แสดงผลกราฟทั้งหมดพร้อมกัน
# =========================================================
plt.show()