import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from torchvision import transforms
import matplotlib.pyplot as plt

from Network_SUBM import Net
from Preprocessing_SUBM import FacialKeypointsDataset, Normalize, ToTensor, Resize

# ----------------------------
# 1. デバイスの設定（CUDAが使える場合はGPU、なければCPU）
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ----------------------------
# 2. データセットの前処理と読み込み
# ----------------------------
data_transform = transforms.Compose([
    Resize(224),  # 入力サイズ 224×224 にリサイズ
    Normalize(),  # 正規化とグレースケール変換
    ToTensor()    # Tensor への変換
])

# npzファイルからデータセットを読み込み
full_dataset = FacialKeypointsDataset(
    npz_file=r"C:\Users\showm\Face-Alignment\face_alignment\training_images.npz",
    transform=data_transform
)

# ----------------------------
# 3. K-fold Cross Validation の設定
# ----------------------------
k_folds = 5
batch_size = 10
num_epochs = 35

kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)

def euclid_dist(pred_points, actual_points):
    dist = torch.sqrt(torch.sum((pred_points - actual_points) ** 2, dim=2))
    return dist

fold_results = []

# ----------------------------
# 4. K-fold Cross Validation の実施（CUDA対応版）
# ----------------------------
for fold, (train_ids, val_ids) in enumerate(kfold.split(full_dataset)):
    print(f"============================")
    print(f"Fold {fold + 1}/{k_folds}")
    print(f"----------------------------")

    # トレーニングとバリデーション用のサブセットを作成
    train_subset = Subset(full_dataset, train_ids)
    val_subset = Subset(full_dataset, val_ids)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0)

    # 各Fold毎に新しいモデルを初期化し、デバイスに移動
    model = Net(1, 20, 88).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())

    for epoch in range(num_epochs):
        # --- トレーニングフェーズ ---
        model.train()
        train_loss = 0.0
        train_euclid = 0.0

        for data in train_loader:
            # 入力データをデバイスに転送し、float型に変換
            images = data['image'].float().to(device)
            key_points = data['keypoints'].view(data['keypoints'].size(0), -1).float().to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, key_points)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # RMSE計算のために形状を整形しデバイス上で計算
            output_points_2ded = outputs.view(outputs.size(0), -1, 2)
            key_points_2ded = key_points.view(key_points.size(0), -1, 2)
            train_euclid += euclid_dist(output_points_2ded, key_points_2ded).mean().item()

        avg_train_loss = train_loss / len(train_loader)
        avg_train_euc = train_euclid / len(train_loader)

        # --- バリデーションフェーズ ---
        model.eval()
        val_loss = 0.0
        val_euclid = 0.0

        with torch.no_grad():
            for data in val_loader:
                images = data['image'].float().to(device)
                key_points = data['keypoints'].view(data['keypoints'].size(0), -1).float().to(device)

                outputs = model(images)
                loss = criterion(outputs, key_points)
                val_loss += loss.item()

                output_points_2ded = outputs.view(outputs.size(0), -1, 2)
                key_points_2ded = key_points.view(key_points.size(0), -1, 2)
                val_euclid += euclid_dist(output_points_2ded, key_points_2ded).mean().item()

        avg_val_loss = val_loss / len(val_loader)
        avg_val_euc = val_euclid / len(val_loader)

        print(f"Fold {fold + 1}, Epoch {epoch + 1}: Train Loss {avg_train_loss:.4f}, Train RMSE {avg_train_euc:.4f} | "
              f"Val Loss {avg_val_loss:.4f}, Val RMSE {avg_val_euc:.4f}")

    # 各Foldのバリデーション損失を記録
    fold_results.append(avg_val_loss)

    # 各Foldごとにモデルを保存（必要に応じて確認用）
    save_path = os.path.join(r'C:\Users\showm\Face-Alignment\face_alignment', f"keypoints_model_fold_{fold + 1}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model for fold {fold + 1} saved to {save_path}\n")

avg_performance = sum(fold_results) / k_folds
print(f"\nK-fold Cross Validation 結果: Average Validation Loss: {avg_performance:.4f}")

plt.figure(figsize=(8, 4))
plt.plot(range(1, k_folds + 1), fold_results, marker='o', linestyle='--')
plt.xlabel('Fold')
plt.ylabel('Validation Loss')
plt.title('各Foldにおける Validation Loss')
plt.grid(True)
plt.show()

# ==============================
# 5. 最適なハイパーパラメータが決定したら、全データを用いた再学習（Final Training）を実施
# ==============================
print("===> Full Dataset Training for Final Model <===")
final_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
final_model = Net(1, 20, 88).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(final_model.parameters())

for epoch in range(num_epochs):
    final_model.train()
    train_loss = 0.0
    train_euclid = 0.0

    for data in final_loader:
        images = data['image'].float().to(device)
        key_points = data['keypoints'].view(data['keypoints'].size(0), -1).float().to(device)

        optimizer.zero_grad()
        outputs = final_model(images)
        loss = criterion(outputs, key_points)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        output_points_2ded = outputs.view(outputs.size(0), -1, 2)
        key_points_2ded = key_points.view(key_points.size(0), -1, 2)
        train_euclid += euclid_dist(output_points_2ded, key_points_2ded).mean().item()

    avg_train_loss = train_loss / len(final_loader)
    avg_train_euc = train_euclid / len(final_loader)
    print(f"Final Model, Epoch {epoch + 1}: Loss {avg_train_loss:.4f}, RMSE {avg_train_euc:.4f}")

# ----------------------------
# 6. 最終モデルの保存
# ----------------------------
final_save_path = os.path.join(r'C:\Users\showm\Face-Alignment\face_alignment', "final_keypoints_model.pth")
torch.save(final_model.state_dict(), final_save_path)
print(f"Final model saved to {final_save_path}")
