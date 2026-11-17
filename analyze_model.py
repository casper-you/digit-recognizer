
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)


# ============================================================
# 1. 项目路径与运行设备
# ============================================================

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"

BATCH_SIZE = 256
RANDOM_STATE = 42
VAL_SIZE = 0.2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("CNN 模型错误分析")
print(f"项目目录：{ROOT}")
print(f"运行设备：{device}")
print("=" * 60)


# ============================================================
# 2. 自动查找训练数据和最佳模型
# ============================================================

train_candidates = [
    DATA_DIR / "train.csv",
    ROOT / "train.csv",
]

model_candidates = [
    MODEL_DIR / "best_digit_cnn.pth",
    ROOT / "best_digit_cnn.pth",
    ROOT / "notebooks" / "best_digit_cnn.pth",
]

train_path = next(
    (p for p in train_candidates if p.is_file()), None
)

model_path = next(
    (p for p in model_candidates if p.is_file()), None
)

if train_path is None:
    raise FileNotFoundError(
        "找不到 train.csv！请确认文件位于 data/train.csv "
        "或项目根目录下。"
    )

if model_path is None:
    # 在项目目录中搜索其他可能的保存位置
    matches = list(ROOT.glob("*.pth"))
    matches += list(MODEL_DIR.glob("*.pth")) if MODEL_DIR.exists() else []
    matches += list((ROOT / "notebooks").glob("*.pth"))

    if matches:
        raise FileNotFoundError(
            "找到了其他 .pth 文件，但没有找到 "
            "'best_digit_cnn.pth'。\n"
            f"找到的文件：{matches}\n"
            "请根据实际文件名修改 model_candidates。"
        )

    raise FileNotFoundError(
        "找不到 best_digit_cnn.pth！请确认训练时保存的模型文件位置。"
    )

print(f"\n训练数据：{train_path}")
print(f"模型权重：{model_path}")


# ============================================================
# 3. 定义 CNN
# 必须与训练时的模型结构及层名称一致
# ============================================================


class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # features.0
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # features.3
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        
        self.classifier = nn.Sequential(
            nn.Flatten(),                   # classifier.0
            nn.Linear(32 * 7 * 7, 128),    # classifier.1
            nn.ReLU(),                      # classifier.2
            nn.Linear(128, 10),             # classifier.3
        )


    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ============================================================
# 4. 读取数据并按照原来的设置划分验证集
# ============================================================

df = pd.read_csv(train_path)

if "label" not in df.columns:
    raise ValueError("train.csv 中没有找到 label 列。")

y = df["label"].to_numpy(dtype=np.int64)
X = df.drop(columns=["label"]).to_numpy(dtype=np.float32)

if X.shape[1] != 784:
    raise ValueError(
        f"预期每张图片有 784 个像素，实际为 {X.shape[1]}。"
    )

# 与训练时保持一致：按标签分层，固定随机种子
train_idx, val_idx = train_test_split(
    np.arange(len(y)),
    test_size=VAL_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)

X_val = X[val_idx] / 255.0
y_val = y[val_idx]

X_val = torch.from_numpy(
    X_val.reshape(-1, 1, 28, 28)
)
y_val = torch.from_numpy(y_val)

val_dataset = TensorDataset(X_val, y_val)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

print(f"\n训练集数量：{len(train_idx)}")
print(f"验证集数量：{len(val_idx)}")


# ============================================================
# 5. 加载最佳模型
# ============================================================

model = DigitCNN().to(device)

# 仅加载自己训练并信任的权重文件
state_dict = torch.load(
    model_path,
    map_location=device,
    weights_only=True,
)

# 如果保存的是包含 state_dict 的字典，也兼容处理
if isinstance(state_dict, dict) and "state_dict" in state_dict:
    state_dict = state_dict["state_dict"]

model.load_state_dict(state_dict)
model.eval()

print("最佳模型加载成功！")


# ============================================================
# 6. 在验证集上进行预测
# ============================================================

all_labels = []
all_preds = []
wrong_images = []
wrong_labels = []
wrong_preds = []

with torch.inference_mode():
    for images, labels in val_loader:
        images = images.to(device)

        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu()

        all_labels.extend(labels.numpy().tolist())
        all_preds.extend(preds.numpy().tolist())

        # 收集预测错误的样本
        wrong_mask = preds != labels

        for img, label, pred in zip(
            images.cpu()[wrong_mask],
            labels[wrong_mask],
            preds[wrong_mask],
        ):
            wrong_images.append(img.squeeze().numpy())
            wrong_labels.append(int(label))
            wrong_preds.append(int(pred))

all_labels = np.array(all_labels)
all_preds = np.array(all_preds)

accuracy = np.mean(all_labels == all_preds)

print("\n" + "=" * 60)
print(f"验证集准确率：{accuracy:.4%}")
print(f"验证集总样本数：{len(all_labels)}")
print(f"预测错误数量：{len(wrong_labels)}")
print("=" * 60)


# ============================================================
# 7. 输出每个数字的分类指标
# ============================================================

print("\n分类报告：")

print(
    classification_report(
        all_labels,
        all_preds,
        labels=list(range(10)),
        digits=4,
        zero_division=0,
    )
)


# ============================================================
# 8. 绘制混淆矩阵
# 行 = 真实数字，列 = 预测数字
# ============================================================

cm = confusion_matrix(
    all_labels,
    all_preds,
    labels=list(range(10)),
)

fig, ax = plt.subplots(figsize=(10, 8))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=list(range(10)),
)

disp.plot(
    ax=ax,
    cmap="Blues",
    values_format="d",
    colorbar=False,
)

ax.set_title("CNN Confusion Matrix")
plt.tight_layout()

cm_path = ROOT / "confusion_matrix.png"
plt.savefig(cm_path, dpi=200, bbox_inches="tight")

print(f"\n混淆矩阵已保存：{cm_path}")
plt.show()


# ============================================================
# 9. 找出最容易混淆的数字组合
# ============================================================

cm_off_diagonal = cm.copy()
np.fill_diagonal(cm_off_diagonal, 0)

pairs = []

for true_digit in range(10):
    for pred_digit in range(10):
        count = cm_off_diagonal[true_digit, pred_digit]
        if count > 0:
            pairs.append(
                (true_digit, pred_digit, int(count))
            )

pairs.sort(key=lambda item: item[2], reverse=True)

print("\n最常见的错误类型（真实数字 -> 预测数字）：")

for true_digit, pred_digit, count in pairs[:10]:
    print(f"{true_digit} -> {pred_digit}: {count} 次")


# ============================================================
# 10. 展示预测错误的图片
# ============================================================

num_show = min(25, len(wrong_labels))

if num_show > 0:
    cols = 5
    rows = (num_show + cols - 1) // cols

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(12, 2.5 * rows),
    )

    axes = np.asarray(axes).reshape(-1)

    for i in range(num_show):
        axes[i].imshow(wrong_images[i], cmap="gray")
        axes[i].set_title(
            f"True: {wrong_labels[i]} | Pred: {wrong_preds[i]}"
        )
        axes[i].axis("off")

    for i in range(num_show, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()

    wrong_path = ROOT / "misclassified_images.png"
    plt.savefig(wrong_path, dpi=200, bbox_inches="tight")

    print(f"\n错误图片已保存：{wrong_path}")
    plt.show()

else:
    print("\n恭喜！当前验证集没有预测错误的图片。")


print("\n模型错误分析完成！")


# ============================================================
# 11. 生成 Kaggle submission.csv
# ============================================================

test_candidates = [
    ROOT / "data" / "test.csv",
    ROOT / "test.csv",
]

test_path = next(
    (p for p in test_candidates if p.is_file()), None
)

if test_path is None:
    print("\n未找到 test.csv，暂时跳过 Kaggle 提交文件生成。")
else:
    test_df = pd.read_csv(test_path)

    if test_df.shape[1] != 784:
        raise ValueError(
            f"测试集预期有784个像素列，实际为{test_df.shape[1]}。"
        )

    X_test = test_df.to_numpy(dtype=np.float32) / 255.0
    X_test = torch.from_numpy(
        X_test.reshape(-1, 1, 28, 28)
    )

    test_loader = DataLoader(
        TensorDataset(X_test),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    predictions = []

    model.eval()
    with torch.inference_mode():
        for (images,) in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            predictions.extend(preds.cpu().tolist())

    submission = pd.DataFrame({
        "ImageId": range(1, len(predictions) + 1),
        "Label": predictions,
    })

    submission_dir = ROOT / "submissions"
    submission_dir.mkdir(parents=True, exist_ok=True)

    submission_path = submission_dir / "submission.csv"
    submission.to_csv(submission_path, index=False)

    print(f"\nKaggle预测图片数量：{len(predictions)}")
    print(f"提交文件已保存：{submission_path}")
    print(submission.head())

