import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import nni
# 动态获取参数
import os

def is_nni_running():
    """判断是否在NNI环境中运行"""
    return "NNI_PLATFORM" in os.environ

if is_nni_running():
    params = nni.get_next_parameter()

    layer1_neurons = params.get("layer1_neurons")
    # layer2_neurons = params.get("layer2_neurons", min(layer1_neurons - 16, 32))  # 确保 layer2_neurons < layer1_neurons
    layer2_neurons = params.get("layer2_neurons") 
    layer3_neurons = params.get("layer3_neurons")
else: layer1_neurons,layer2_neurons,layer3_neurons  = 19,79,75
# # 额外约束检查（如果需要更加严格）
# layer2_neurons = max(16, min(layer2_neurons, layer1_neurons - 16))  # 不小于16且小于layer1_neurons

# # 打印约束结果，便于调试
# print(f"Layer1 neurons: {layer1_neurons}, Layer2 neurons: {layer2_neurons}")

# Load and preprocess data
qns = pd.read_csv(r'data\data_cleaned.csv')  # Load dataset
X = qns.iloc[:, 2:].values  # Extract features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset
X_train_raw, X_temp = train_test_split(X, test_size=0.4)
X_val_raw, X_test_raw = train_test_split(X_temp, test_size=0.5)
X_train = scaler.fit_transform(X_train_raw)
X_val = scaler.transform(X_val_raw)
X_test = scaler.transform(X_test_raw)

# Define PyTorch dataset
class QuestionnaireDataset(Dataset):
    def __init__(self, data):
        self.data = torch.tensor(data, dtype=torch.float32)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.data[idx]

train_dataset = QuestionnaireDataset(X_train)
val_dataset = QuestionnaireDataset(X_val)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
# Define PyTorch dataset
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, sparsity_weight=1e-4, sparsity_target=0.05):
        super(Autoencoder, self).__init__()
        
        self.sparsity_weight = sparsity_weight  # 稀疏性正则化权重
        self.sparsity_target = sparsity_target  # 稀疏性目标值
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, layer1_neurons),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(layer1_neurons, layer2_neurons),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(layer2_neurons, layer3_neurons),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(layer3_neurons, latent_dim)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, layer3_neurons),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(layer3_neurons, layer2_neurons),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(layer2_neurons, layer1_neurons),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(layer1_neurons, input_dim),
        )
        
    def forward(self, x):
        # Forward pass
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

    def kl_divergence(self, p, p_hat):
        """计算稀疏性约束的KL散度"""
        p_hat = torch.clamp(p_hat, 1e-10, 1 - 1e-10)  # 避免 log(0)
        return p * torch.log(p / p_hat) + (1 - p) * torch.log((1 - p) / (1 - p_hat))


# Model initialization
input_dim = X_train.shape[1]
latent_dim = 5
model = Autoencoder(input_dim, latent_dim)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

# Train Autoencoder
best_val_loss = float('inf')
patience = 10
epochs_without_improvement = 0
train_losses, val_losses = [], []

for epoch in range(2000):
    model.train()
    train_loss = 0
    for batch_features, _ in train_loader:
        optimizer.zero_grad()
        encoded, outputs = model(batch_features)
        average_activation = encoded.mean(dim=0)  # 隐含层的平均激活
        kl_div_loss = model.kl_divergence(model.sparsity_target, average_activation).mean()
        loss = criterion(outputs, batch_features)+model.sparsity_weight * kl_div_loss
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * batch_features.size(0)
    train_losses.append(train_loss / len(train_loader.dataset))

    # Validation step
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_features, _ in val_loader:
            encoded, outputs = model(batch_features)
            loss = criterion(outputs, batch_features)
            val_loss += loss.item() * batch_features.size(0)
    val_losses.append(val_loss / len(val_loader.dataset))
    scheduler.step(val_loss)
    if is_nni_running():
    # 报告训练损失和验证损失
        nni.report_intermediate_result({
            "train_loss": train_loss / len(train_loader.dataset),
            "val_loss": val_loss / len(val_loader.dataset)
        })

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

# # Plot loss curves
# plt.plot(train_losses, label='Train Loss')
# plt.plot(val_losses, label='Validation Loss')
# plt.legend()
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Loss Curves')
# plt.show()

# Extract latent features
model.eval()
with torch.no_grad():
    encoded, reconstructed = model(torch.tensor(X, dtype=torch.float32))
    X_latent = model.encoder(torch.tensor(X_scaled, dtype=torch.float32)).numpy()
    reconstruction_errors = torch.mean((torch.tensor(X, dtype=torch.float32) - reconstructed) ** 2, dim=1).numpy()

total_variance = np.var(X, axis=0).sum()
latent_df = pd.DataFrame(X_latent, columns=[f'Factor_{i+1}' for i in range(latent_dim)])
# 计算每个因子贡献的方差
# 通过重建数据的方差贡献，计算解释率
reconstruction_variance = np.var(reconstructed.numpy(), axis=0).sum()

# 计算方差解释率
explained_variance_ratio = reconstruction_variance / total_variance
print(f"Explained variance ratio: {explained_variance_ratio:.8f}")
nni.report_final_result(explained_variance_ratio)
