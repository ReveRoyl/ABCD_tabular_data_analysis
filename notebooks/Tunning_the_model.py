import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from factor_analyzer import FactorAnalyzer
from models import Autoencoder
from sklearn.decomposition import NMF
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from utils import translate_text
from sklearn.model_selection import KFold
import shap
import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import optuna
from sklearn.model_selection import KFold
import torch.nn.functional as F
import io
from PIL import Image
from utils import get_cbcl_details

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

code_dir = Path(os.getcwd())
data_path = code_dir.parent / "data"
assert os.path.exists(
    data_path
), "Data directory not found. Make sure you're running this code from the root directory of the project."

with open(data_path / "cbcl_data_remove_unrelated.csv", "r", encoding="utf-8") as f:
    qns = pd.read_csv(f)

X = qns.iloc[:, 1:].values

# Standardize the data
# scaler = StandardScaler()
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split into training and validation sets
X_train_raw, X_temp = train_test_split(X, test_size=0.4)
X_val_raw, X_test_raw = train_test_split(X_temp, test_size=0.5)


X_train = scaler.fit_transform(X_train_raw)
X_val = scaler.transform(X_val_raw)
X_test = scaler.transform(X_test_raw)

class QuestionnaireDataset(Dataset):
    def __init__(self, data):
        self.data = torch.tensor(data, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.data[idx]  # Input and target are the same


# the autoencoder architecture
class AutoencoderModel(nn.Module):
    def __init__(
        self, input_dim, latent_dim, layer1_neurons, layer2_neurons, layer3_neurons, layer4_neurons
    ):
        super(AutoencoderModel, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, layer1_neurons),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(layer1_neurons, layer2_neurons),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(layer2_neurons, layer3_neurons),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(layer3_neurons, layer4_neurons),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(layer4_neurons, latent_dim),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, layer4_neurons),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(layer4_neurons, layer3_neurons),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(layer3_neurons, layer2_neurons),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(layer2_neurons, layer1_neurons),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(layer1_neurons, input_dim),
        )

    def forward(self, x):
        # forward
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded



def is_nni_running():
    """If NNI is running, return True; otherwise, return False."""
    return "NNI_PLATFORM" in os.environ


def decorrelation_loss(latent_repr):
    """
    计算潜在表示的去相关正则化损失。
    latent_repr: (batch_size, latent_dim)
    """
    batch_size, latent_dim = latent_repr.shape
    # 计算协方差矩阵
    latent_repr = latent_repr - latent_repr.mean(dim=0, keepdim=True)  # 先中心化
    cov_matrix = (latent_repr.T @ latent_repr) / batch_size  # 计算协方差
    mask = torch.eye(latent_dim, device=latent_repr.device)  # 生成单位矩阵
    loss = torch.sum((cov_matrix * (1 - mask))**2)  # 只计算非对角元素
    return loss

class Autoencoder:
    def __init__(
        self,
        X_train,
        X_val,
        encoding_dim,
        layer1_neurons=0,
        layer2_neurons=0,
        layer3_neurons=0,
        layer4_neurons=0,
    ):

        train_dataset = QuestionnaireDataset(X_train)
        val_dataset = QuestionnaireDataset(X_val)

        self.train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=32)

        # Model initialization
        input_dim = X_train.shape[1]
        latent_dim = encoding_dim
        self.model = AutoencoderModel(
            input_dim, latent_dim, layer1_neurons, layer2_neurons, layer3_neurons, layer4_neurons
        ).to(device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.1, patience=5
        )
        self.explained_variance_ratio_total_value = None
    
    def get_model(self):
        return self.model
    
    def train(self, show_plot=False):
        best_val_loss = float("inf")
        patience = 20
        epochs_without_improvement = 0
        train_losses, val_losses = [], []

        # Modify the training loop to include decorrelation loss
        for epoch in range(2000):
            self.model.train()
            train_loss = 0
            for batch_features, _ in self.train_loader:
                batch_features = batch_features.to(device)
                self.optimizer.zero_grad()
                outputs = self.model(batch_features)
                reconstruction_loss = self.criterion(outputs, batch_features)
                
                # Calculate decorrelation loss
                latent_repr = self.model.encoder(batch_features)
                decorrelation_loss_value = decorrelation_loss(latent_repr)
                
                # Combine losses
                combined_loss = reconstruction_loss + decorrelation_loss_value
                
                combined_loss.backward()
                self.optimizer.step()
                train_loss += combined_loss.item() * batch_features.size(0)
            train_losses.append(train_loss / len(self.train_loader.dataset))

            # Validation step
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_features, _ in self.val_loader:
                    batch_features = batch_features.to(device)
                    outputs = self.model(batch_features)
                    reconstruction_loss = self.criterion(outputs, batch_features)
                    
                    # Calculate decorrelation loss
                    latent_repr = self.model.encoder(batch_features)
                    decorrelation_loss_value = decorrelation_loss(latent_repr)
                    
                    # Combine losses
                    combined_loss = reconstruction_loss + decorrelation_loss_value
                    
                    val_loss += combined_loss.item() * batch_features.size(0)
            val_losses.append(val_loss / len(self.val_loader.dataset))
            self.scheduler.step(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        if show_plot:
            # Plot loss curves
            plt.plot(train_losses, label="Train Loss")
            plt.plot(val_losses, label="Validation Loss")
            plt.legend()
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Loss Curves")
            plt.show()

    def evaluate_on_data(self, X):
        self.model.eval()
        with torch.no_grad():
            # Forward pass to get reconstructed data
            if not isinstance(X, np.ndarray):
                X = np.array(X)
            X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

            reconstructed = self.model(X_tensor)

            # Get the output of the encoder, i.e., latent factors
            latent_factors = self.model.encoder(
                X_tensor
            ).cpu().numpy()

            # Calculate the variance of each latent factor
            latent_variances = np.var(latent_factors, axis=0)

            # Calculate the total variance of the original data
            total_variance = np.var(X, axis=0).sum()

            # Calculate the explained variance ratio for each latent factor
            explained_variance_ratios = latent_variances / total_variance

            # Print the explained variance ratio for each latent factor
            # for i, ratio in enumerate(explained_variance_ratios):
            #     print(f"Explained variance ratio of latent factor {i+1}: {ratio:.8f}")

            # Calculate reconstruction errors
            reconstruction_errors = torch.mean(
                (X_tensor - reconstructed) ** 2,
                dim=1,
            ).cpu().numpy()

            # Calculate the variance contribution of the reconstructed data
            reconstruction_variance = np.var(reconstructed.cpu().numpy(), axis=0).sum()

            # Calculate the total explained variance ratio
            explained_variance_ratio_total = reconstruction_variance / total_variance
            # print(
            #     f"Total explained variance ratio (by all factors): {explained_variance_ratio_total:.8f}"
            # )

        return (
            latent_factors,
            reconstruction_errors,
            explained_variance_ratios,
            explained_variance_ratio_total,
        )
def objective(trial):
    # 读取数据
    code_dir = Path(os.getcwd())
    data_path = code_dir.parent / "data"
    assert os.path.exists(data_path), "Cannot find data folder, please run the code in root directory of the project."
    
    qns = pd.read_csv(data_path / "cbcl_data_remove_unrelated.csv", encoding="utf-8")
    X = qns.iloc[:, 2:].values
    # Optuna 搜索的参数范围
    layer1_neurons = trial.suggest_int("layer1_neurons", 50, 200)
    layer2_neurons = trial.suggest_int("layer2_neurons", 50, 200)
    layer3_neurons = trial.suggest_int("layer3_neurons", 50, 200)
    layer4_neurons = trial.suggest_int("layer4_neurons", 50, 200)

    # 外层 CV
    outer_cv = KFold(n_splits=2, shuffle=True)
    outer_scores = []

    for outer_fold, (outer_train_idx, outer_test_idx) in enumerate(outer_cv.split(X), start=1):
        print(f"====== Outer Fold {outer_fold} ======")
        X_outer_train_raw, X_outer_test_raw = X[outer_train_idx], X[outer_test_idx]
        
        scaler_fold = MinMaxScaler()
        X_outer_train = scaler_fold.fit_transform(X_outer_train_raw)
        X_outer_test = scaler_fold.transform(X_outer_test_raw)
        
        # 内层 CV
        inner_cv = KFold(n_splits=2, shuffle=True)
        inner_scores = []

        for inner_fold, (inner_train_idx, inner_val_idx) in enumerate(inner_cv.split(X_outer_train), start=1):
            print(f"  -- Inner Fold {inner_fold}")
            X_inner_train = X_outer_train[inner_train_idx]
            X_inner_val = X_outer_train[inner_val_idx]
            
            autoencoder_inner = Autoencoder(
                X_inner_train,
                X_inner_val,
                encoding_dim=5,
                layer1_neurons=layer1_neurons,
                layer2_neurons=layer2_neurons,
                layer3_neurons=layer3_neurons,
                layer4_neurons=layer4_neurons,
            )
            
            autoencoder_inner.train()
            _, inner_score, _, inner_variance_explained = autoencoder_inner.evaluate_on_data(X_inner_val)
            inner_score = inner_score.mean()
            print(f"Inner fold reconstruction error: {inner_score}")
            print(f"Inner fold explained variance ratio: {inner_variance_explained}")
            inner_scores.append(inner_score)
        
        avg_inner_score = np.mean(inner_scores)
        print(f"Average inner CV score for outer fold {outer_fold}: {avg_inner_score}")
        
        # 用整个 outer train 再训练一次，在 outer test 上评估
        autoencoder_outer = Autoencoder(
            X_outer_train,
            X_outer_test,
            encoding_dim=5,
            layer1_neurons=layer1_neurons,
            layer2_neurons=layer2_neurons,
            layer3_neurons=layer3_neurons,
            layer4_neurons=layer4_neurons,
        )
        autoencoder_outer.train()
        _, outer_score, _, outer_variance_explained = autoencoder_outer.evaluate_on_data(X_outer_test)
        outer_score = outer_score.mean()
        print(f"Outer fold {outer_fold} test reconstruction error: {outer_score}")
        print(f"Outer fold {outer_fold} explained variance ratio: {outer_variance_explained}")
        outer_scores.append(outer_score)

    final_avg_score = np.mean(outer_scores)
    print("Final average test reconstruction error:", final_avg_score)

    return final_avg_score

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=15)

    print("Best trial:")
    print(study.best_trial)
    print("Best hyperparameters:", study.best_trial.params)