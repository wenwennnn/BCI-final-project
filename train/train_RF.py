import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay

# === 1. 讀取 npz 特徵 ===
data = np.load("dreamer_psd_features.npz")
X = data["X"].reshape(len(data["X"]), -1)  # (N, 14*5)
Y = data["Y"]  # (N, 2), values: 0~4

# === 2. 標準化與切分資料 ===
X = StandardScaler().fit_transform(X)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# === 3. 模型訓練 ===
model = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)

# === 4. 評估準確率 ===
acc_arousal = accuracy_score(Y_test[:, 0], Y_pred[:, 0])
acc_valence = accuracy_score(Y_test[:, 1], Y_pred[:, 1])
print(f"Arousal Accuracy: {acc_arousal:.2%}")
print(f"Valence Accuracy: {acc_valence:.2%}")

# === 5. 視覺化 ===
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# 混淆矩陣：Arousal
ConfusionMatrixDisplay.from_predictions(
    Y_test[:, 0], Y_pred[:, 0],
    display_labels=[1, 2, 3, 4, 5],
    ax=axs[0], cmap="Blues", colorbar=False
)
axs[0].set_title(f"Arousal Confusion\nAcc: {acc_arousal:.2%}")

# 混淆矩陣：Valence
ConfusionMatrixDisplay.from_predictions(
    Y_test[:, 1], Y_pred[:, 1],
    display_labels=[1, 2, 3, 4, 5],
    ax=axs[1], cmap="Greens", colorbar=False
)
axs[1].set_title(f"Valence Confusion\nAcc: {acc_valence:.2%}")

plt.tight_layout()
plt.savefig("randomforest_result.jpg", dpi=300)
plt.show()
