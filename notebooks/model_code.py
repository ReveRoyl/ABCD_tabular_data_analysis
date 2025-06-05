# model.py

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import scipy.stats as stats

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def is_nni_running():
    return "NNI_PLATFORM" in os.environ


class QuestionnaireDataset(Dataset):
    def __init__(self, data):
        self.data = torch.tensor(data, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.data[idx]


# ===== AE =====

def decorrelation_loss(latent_repr: torch.Tensor) -> torch.Tensor:
    batch_size, latent_dim = latent_repr.shape
    centered = latent_repr - latent_repr.mean(dim=0, keepdim=True)
    cov = (centered.T @ centered) / batch_size
    eye = torch.eye(latent_dim, device=latent_repr.device)
    off_diag = cov * (1 - eye)
    return torch.sum(off_diag**2)


class AEModel(nn.Module):
    def __init__(self, input_dim, latent_dim, h1, h2, h3):
        super(AEModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.LeakyReLU(0.01),
            nn.Linear(h1, h2),
            nn.LeakyReLU(0.01),
            nn.Linear(h2, h3),
            nn.LeakyReLU(0.01),
            nn.Linear(h3, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, h3),
            nn.LeakyReLU(0.01),
            nn.Linear(h3, h2),
            nn.LeakyReLU(0.01),
            nn.Linear(h2, h1),
            nn.LeakyReLU(0.01),
            nn.Linear(h1, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


class AE:
    """
    简化版示例：Autoencoder 的外层封装
    __init__(X_train, X_val, encoding_dim, h1, h2, h3)
    train(max_epochs=..., patience=..., show_plot=...)
    evaluate_on_data(X) -> (latent, rec_errors, evr, total_evr, recon)
    """
    def __init__(self, X_train, X_val, encoding_dim, h1=0, h2=0, h3=0):
        train_ds = QuestionnaireDataset(X_train)
        val_ds = QuestionnaireDataset(X_val)
        self.train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        self.val_loader = DataLoader(val_ds, batch_size=32)

        input_dim = X_train.shape[1]
        self.model = AEModel(input_dim, encoding_dim, h1, h2, h3).to(device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.1, patience=5
        )

    def train(self, max_epochs=2000, patience=20, show_plot=False):
        best_val_loss = float("inf")
        epochs_no_improve = 0
        train_losses, val_losses = [], []
        for epoch in range(max_epochs):
            self.model.train()
            total_train_loss = 0
            for batch_x, _ in self.train_loader:
                batch_x = batch_x.to(device)
                self.optimizer.zero_grad()
                reconstructed = self.model(batch_x)
                rec_loss = self.criterion(reconstructed, batch_x)
                latent = self.model.encoder(batch_x)
                dec_loss = decorrelation_loss(latent)
                loss = rec_loss + dec_loss
                loss.backward()
                self.optimizer.step()
                total_train_loss += loss.item() * batch_x.size(0)

            train_avg = total_train_loss / len(self.train_loader.dataset)
            train_losses.append(train_avg)

            # 验证
            self.model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for batch_x, _ in self.val_loader:
                    batch_x = batch_x.to(device)
                    reconstructed = self.model(batch_x)
                    rec_loss = self.criterion(reconstructed, batch_x)
                    latent = self.model.encoder(batch_x)
                    dec_loss = decorrelation_loss(latent)
                    total_val_loss += (rec_loss + dec_loss).item() * batch_x.size(0)

            val_avg = total_val_loss / len(self.val_loader.dataset)
            val_losses.append(val_avg)
            self.scheduler.step(val_avg)

            if val_avg < best_val_loss:
                best_val_loss = val_avg
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        if show_plot:
            plt.plot(train_losses, label="Train Loss")
            plt.plot(val_losses, label="Validation Loss")
            plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("AE Loss Curves")
            plt.legend(); plt.show()

    def evaluate_on_data(self, X):
        self.model.eval()
        X_np = np.asarray(X)
        with torch.no_grad():
            X_t = torch.tensor(X_np, dtype=torch.float32).to(device)
            recon = self.model(X_t)
            latent = self.model.encoder(X_t).cpu().numpy()
            latent_vars = np.var(latent, axis=0)
            total_var = np.var(X_np, axis=0).sum()
            evr = latent_vars / total_var
            recon_np = recon.cpu().numpy()
            rec_errors = np.mean((X_np - recon_np)**2, axis=1)
            recon_var = np.var(recon_np, axis=0).sum()
            total_evr = recon_var / total_var
        return latent, rec_errors, evr, total_evr, recon_np


# ===== SparseAE =====

class SparseAEModel(nn.Module):
    def __init__(self, input_dim, latent_dim, h1, h2, h3, sparsity_target=0.05, beta=1.0):
        super(SparseAEModel, self).__init__()
        self.sparsity_target = sparsity_target
        self.beta = beta
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.LeakyReLU(0.01),
            nn.Linear(h1, h2),
            nn.LeakyReLU(0.01),
            nn.Linear(h2, h3),
            nn.LeakyReLU(0.01),
            nn.Linear(h3, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, h3),
            nn.LeakyReLU(0.01),
            nn.Linear(h3, h2),
            nn.LeakyReLU(0.01),
            nn.Linear(h2, h1),
            nn.LeakyReLU(0.01),
            nn.Linear(h1, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    def kl_divergence_loss(self, avg_activation: torch.Tensor) -> torch.Tensor:
        rho_hat = torch.clamp(avg_activation, 1e-10, 1.0)
        rho = self.sparsity_target
        return torch.sum(
            rho * torch.log(rho / rho_hat)
            + (1 - rho) * torch.log((1 - rho) / (1 - rho_hat))
        )


class SparseAE:
    """
    Sparse autoencoder封装
    __init__(X_train, X_val, encoding_dim, h1, h2, h3, sparsity_target, beta)
    train(max_epochs=..., patience=..., show_plot=...)
    evaluate_on_data(X) -> (latent, rec_errors, evr, total_evr, recon_np)
    """
    def __init__(self, X_train, X_val, encoding_dim, h1=0, h2=0, h3=0, sparsity_target=0.05, beta=1.0):
        train_ds = QuestionnaireDataset(X_train)
        val_ds = QuestionnaireDataset(X_val)
        self.train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        self.val_loader = DataLoader(val_ds, batch_size=32)

        input_dim = X_train.shape[1]
        self.model = SparseAEModel(input_dim, encoding_dim, h1, h2, h3, sparsity_target, beta).to(device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.1, patience=5
        )

    def train(self, max_epochs=2000, patience=20, show_plot=False):
        best_val = float("inf")
        epochs_no_improve = 0
        train_hist, val_hist = [], []

        for epoch in range(max_epochs):
            self.model.train()
            total_train_loss = 0
            for batch_x, _ in self.train_loader:
                batch_x = batch_x.to(device)
                self.optimizer.zero_grad()
                outputs = self.model(batch_x)
                rec_loss = self.criterion(outputs, batch_x)
                latent = self.model.encoder(batch_x)
                avg_act = torch.mean(torch.abs(latent), dim=0)
                kl_loss = self.model.kl_divergence_loss(avg_act)
                loss = rec_loss + self.model.beta * kl_loss
                loss.backward()
                self.optimizer.step()
                total_train_loss += loss.item() * batch_x.size(0)

            train_avg = total_train_loss / len(self.train_loader.dataset)
            train_hist.append(train_avg)

            self.model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for batch_x, _ in self.val_loader:
                    batch_x = batch_x.to(device)
                    outputs = self.model(batch_x)
                    rec_loss = self.criterion(outputs, batch_x)
                    latent = self.model.encoder(batch_x)
                    avg_act = torch.mean(torch.abs(latent), dim=0)
                    kl_loss = self.model.kl_divergence_loss(avg_act)
                    total_val_loss += (rec_loss + self.model.beta * kl_loss).item() * batch_x.size(0)

            val_avg = total_val_loss / len(self.val_loader.dataset)
            val_hist.append(val_avg)
            self.scheduler.step(val_avg)

            if val_avg < best_val:
                best_val = val_avg
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        if show_plot:
            plt.plot(train_hist, label="Train Loss")
            plt.plot(val_hist, label="Validation Loss")
            plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("SparseAE Loss Curves")
            plt.legend(); plt.show()

    def evaluate_on_data(self, X):
        self.model.eval()
        X_np = np.asarray(X)
        with torch.no_grad():
            X_t = torch.tensor(X_np, dtype=torch.float32).to(device)
            recon = self.model(X_t)
            latent = self.model.encoder(X_t).cpu().numpy()
            latent_vars = np.var(latent, axis=0)
            total_var = np.var(X_np, axis=0).sum()
            evr = latent_vars / total_var
            recon_np = recon.cpu().numpy()
            rec_errors = np.mean((X_np - recon_np)**2, axis=1)
            recon_var = np.var(recon_np, axis=0).sum()
            total_evr = 1 - np.var((X_t - recon).cpu().numpy(), axis=0).sum() / total_var
        return latent, rec_errors, evr, total_evr, recon_np

    def export_to_onnx(self, X_train, onnx_path):
        dummy_input = torch.tensor(X_train[:1], dtype=torch.float32).to(device)
        torch.onnx.export(
            self.model,
            dummy_input,
            onnx_path,
            input_names=["input"],
            output_names=["reconstructed"],
            dynamic_axes={"input": {0: "batch_size"}, "reconstructed": {0: "batch_size"}},
            opset_version=11,
        )
        print(f"Model exported to {onnx_path}")


# ===== COAE （仅示例） =====

class ClusteringLayer(nn.Module):
    def __init__(self, n_clusters, latent_dim):
        super(ClusteringLayer, self).__init__()
        self.cluster_centers = nn.Parameter(torch.randn(n_clusters, latent_dim))

    def forward(self, z):
        logits = z @ self.cluster_centers.t()
        return F.softmax(logits, dim=1)


def orthogonality_loss(z: torch.Tensor) -> torch.Tensor:
    z_centered = z - z.mean(dim=0, keepdim=True)
    cov = (z_centered.T @ z_centered) / z.size(0)
    I = torch.eye(z.size(1), device=z.device)
    return torch.sum((cov - I) ** 2)


def sparsity_kl_divergence(rho: float, rho_hat: torch.Tensor) -> torch.Tensor:
    rho_hat = torch.clamp(rho_hat, 1e-10, 1 - 1e-10)
    return torch.sum(
        rho * torch.log(rho / rho_hat)
        + (1 - rho) * torch.log((1 - rho) / (1 - rho_hat))
    )


def compute_target_distribution(p: torch.Tensor) -> torch.Tensor:
    weight = (p ** 2) / torch.sum(p, dim=0)
    return (weight.t() / torch.sum(weight, dim=1)).t()


class COAEModel(nn.Module):
    def __init__(self, input_dim, latent_dim, h1, h2, h3, n_clusters):
        super(COAEModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, h3),
            nn.ReLU(),
            nn.Linear(h3, latent_dim),
            nn.Sigmoid(),
        )
        self.clustering_layer = ClusteringLayer(n_clusters, latent_dim)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        e0, e2, e4, e6 = self.encoder[0], self.encoder[2], self.encoder[4], self.encoder[6]
        x = F.linear(z, e6.weight.t())
        x = F.relu(x)
        x = F.linear(x, e4.weight.t())
        x = F.relu(x)
        x = F.linear(x, e2.weight.t())
        x = F.relu(x)
        x = F.linear(x, e0.weight.t())
        return x

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        p = self.clustering_layer(z)
        return x_hat, z, p


class COAETrainer:
    """
    X_train: numpy array, X_val: numpy array,
    latent_dim, h1, h2, h3, n_clusters,
    lambda_orth, mu_clust, lambda_sparsity, sparsity_target
    """
    def __init__(self, X_train, X_val, latent_dim, h1, h2, h3, n_clusters,
                 lambda_orth=1e-1, mu_clust=1.0, lambda_sparsity=1e-4, sparsity_target=0.8):
        train_ds = QuestionnaireDataset(X_train)
        val_ds = QuestionnaireDataset(X_val)
        self.train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        self.val_loader = DataLoader(val_ds, batch_size=32)
        input_dim = X_train.shape[1]

        self.model = COAEModel(input_dim, latent_dim, h1, h2, h3, n_clusters).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.1, patience=5
        )
        self.criterion = nn.MSELoss()
        self.lambda_orth = lambda_orth
        self.mu_clust = mu_clust
        self.lambda_sparsity = lambda_sparsity
        self.sparsity_target = sparsity_target

    def train(self, max_epochs=200, patience=50, show_plot=False):
        best_val = float("inf")
        epochs_no_improve = 0
        train_losses, val_losses = [], []

        for epoch in range(max_epochs):
            self.model.train()
            total_train_loss = 0
            for batch_x, _ in self.train_loader:
                batch_x = batch_x.to(device)
                self.optimizer.zero_grad()
                x_hat, z, p = self.model(batch_x)
                q = compute_target_distribution(p).detach()
                rec_loss = self.criterion(x_hat, batch_x)
                orth_loss = orthogonality_loss(z)
                clust_loss = F.kl_div(p.log(), q, reduction="batchmean")
                rho_hat = torch.mean(torch.sigmoid(z), dim=0)
                sparse_loss = sparsity_kl_divergence(self.sparsity_target, rho_hat)
                loss = (
                    rec_loss
                    + self.lambda_orth * orth_loss
                    + self.mu_clust * clust_loss
                    + self.lambda_sparsity * sparse_loss
                )
                loss.backward()
                self.optimizer.step()
                total_train_loss += loss.item() * batch_x.size(0)

            train_avg = total_train_loss / len(self.train_loader.dataset)
            train_losses.append(train_avg)

            self.model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for batch_x, _ in self.val_loader:
                    batch_x = batch_x.to(device)
                    x_hat, z, p = self.model(batch_x)
                    q = compute_target_distribution(p)
                    rec_loss = self.criterion(x_hat, batch_x)
                    orth_loss = orthogonality_loss(z)
                    clust_loss = F.kl_div(p.log(), q, reduction="batchmean")
                    rho_hat = torch.mean(torch.sigmoid(z), dim=0)
                    sparse_loss = sparsity_kl_divergence(self.sparsity_target, rho_hat)
                    val_loss = (
                        rec_loss
                        + self.lambda_orth * orth_loss
                        + self.mu_clust * clust_loss
                        + self.lambda_sparsity * sparse_loss
                    )
                    total_val_loss += val_loss.item() * batch_x.size(0)

            val_avg = total_val_loss / len(self.val_loader.dataset)
            val_losses.append(val_avg)
            self.scheduler.step(val_avg)
            print(f"Epoch {epoch+1}: Train={train_avg:.4f}, Val={val_avg:.4f}")

            if val_avg < best_val:
                best_val = val_avg
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        if show_plot:
            plt.plot(train_losses, label="Train Loss")
            plt.plot(val_losses, label="Validation Loss")
            plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("COAE Loss Curves")
            plt.legend(); plt.show()
        
    def evaluate_on_data(self, X):
        """
        对输入 X 做前向，返回 (latent, rec_errors, evr, total_evr, recon)：
        - latent: (n_samples, latent_dim)
        - rec_errors: 每个样本的重构 MSE，形状 (n_samples,)
        - evr: 每个潜变量在重构中解释的方差比例，形状 (latent_dim,)
        - total_evr: 重构数据方差 / 输入总方差，标量
        - recon: 重构后的数据，形状 (n_samples, input_dim)
        """
        self.model.eval()
        with torch.no_grad():
            # 1) 编码得到 z
            X_t = torch.tensor(X, dtype=torch.float32, device=device)
            z = self.model.encode(X_t)               # Tensor (n_samples, latent_dim)
            z_np = z.cpu().numpy()

            # 2) 计算每个潜变量的方差并归一化
            latent_vars = np.var(z_np, axis=0)          # (latent_dim,)
            total_var = np.var(X, axis=0).sum()      # 输入空间总方差
            evr = latent_vars / total_var               # (latent_dim,)

            # 3) 解码得到重构 x_hat
            x_hat = self.model.decode(z)                # Tensor (n_samples, input_dim)
            recon_np = x_hat.cpu().numpy()

            # 4) 每个样本的重构误差（MSE）
            rec_errors = np.mean((X - recon_np) ** 2, axis=1)  # (n_samples,)

            # 5) 计算重构的方差并归一化，得到 total_evr
            recon_var = np.var(recon_np, axis=0).sum()  # 重构数据各维度方差之和
            total_evr = recon_var / total_var           # 标量

        return z_np, rec_errors, evr, total_evr, recon_np



# ===== VAE =====

class VAEModel(nn.Module):
    def __init__(self, input_dim, latent_dim, h1, h2, h3):
        super(VAEModel, self).__init__()
        self.encoder_net = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.LeakyReLU(0.01),
            nn.Linear(h1, h2),
            nn.LeakyReLU(0.01),
            nn.Linear(h2, h3),
            nn.LeakyReLU(0.01),
        )
        self.enc_mu = nn.Linear(h3, latent_dim)
        self.enc_logvar = nn.Linear(h3, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, h3),
            nn.LeakyReLU(0.01),
            nn.Linear(h3, h2),
            nn.LeakyReLU(0.01),
            nn.Linear(h2, h1),
            nn.LeakyReLU(0.01),
            nn.Linear(h1, input_dim),
        )

    def encode(self, x):
        h = self.encoder_net(x)
        return self.enc_mu(h), self.enc_logvar(h)

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar


class VariationalAutoencoder:
    def __init__(self, X_train, X_val, encoding_dim, h1=0, h2=0, h3=0, beta_kl=1.0):
        train_ds = QuestionnaireDataset(X_train)
        val_ds = QuestionnaireDataset(X_val)
        self.train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        self.val_loader = DataLoader(val_ds, batch_size=32)

        input_dim = X_train.shape[1]
        self.model = VAEModel(input_dim, encoding_dim, h1, h2, h3).to(device)
        self.recon_loss_fn = nn.MSELoss(reduction="sum")
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.1, patience=5
        )
        self.beta_kl = beta_kl

    def train(self, max_epochs=2000, patience=20, show_plot=False):
        best_val = float("inf")
        wait = 0
        train_hist, val_hist = [], []

        for epoch in range(max_epochs):
            self.model.train()
            total_loss = 0
            for x, _ in self.train_loader:
                x = x.to(device)
                self.optimizer.zero_grad()
                recon, mu, logvar = self.model(x)
                rec_loss = self.recon_loss_fn(recon, x) / x.size(0)
                kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                loss = rec_loss + self.beta_kl * kl_loss
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item() * x.size(0)

            train_avg = total_loss / len(self.train_loader.dataset)
            train_hist.append(train_avg)

            self.model.eval()
            val_accum = 0
            with torch.no_grad():
                for x, _ in self.val_loader:
                    x = x.to(device)
                    recon, mu, logvar = self.model(x)
                    rec_loss = self.recon_loss_fn(recon, x) / x.size(0)
                    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                    val_accum += (rec_loss + self.beta_kl * kl_loss).item() * x.size(0)

            val_avg = val_accum / len(self.val_loader.dataset)
            val_hist.append(val_avg)
            self.scheduler.step(val_avg)

            if val_avg < best_val:
                best_val = val_avg
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        if show_plot:
            plt.plot(train_hist, label="Train Loss")
            plt.plot(val_hist, label="Validation Loss")
            plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("VAE Loss Curves")
            plt.legend(); plt.show()

    def evaluate_on_data(self, X):
        self.model.eval()
        X_np = np.asarray(X)
        with torch.no_grad():
            X_t = torch.tensor(X_np, dtype=torch.float32).to(device)
            recon, mu, logvar = self.model(X_t)
            latent = mu.cpu().numpy()
            latent_vars = np.var(latent, axis=0)
            total_var = np.var(X_np, axis=0).sum()
            evr = latent_vars / total_var
            recon_np = recon.cpu().numpy()
            rec_errors = np.mean((X_np - recon_np)**2, axis=1)
            recon_var = np.var(recon_np, axis=0).sum()
            total_evr = 1 - np.var((X_t - recon).cpu().numpy(), axis=0).sum() / total_var
        return latent, rec_errors, evr, total_evr, recon_np


# ===== BetaVAE =====

class BetaVAEModel(nn.Module):
    def __init__(self, input_dim, latent_dim, h1, h2, h3):
        super(BetaVAEModel, self).__init__()
        self.encoder_net = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.LeakyReLU(0.01),
            nn.Linear(h1, h2),
            nn.LeakyReLU(0.01),
            nn.Linear(h2, h3),
            nn.LeakyReLU(0.01),
        )
        self.fc_mu = nn.Linear(h3, latent_dim)
        self.fc_logvar = nn.Linear(h3, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, h3),
            nn.LeakyReLU(0.01),
            nn.Linear(h3, h2),
            nn.LeakyReLU(0.01),
            nn.Linear(h2, h1),
            nn.LeakyReLU(0.01),
            nn.Linear(h1, input_dim),
        )

    def encode(self, x):
        h = self.encoder_net(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

    def get_latent(self, x):
        mu, logvar = self.encode(x)
        return self.reparameterize(mu, logvar)


class BetaVAE:
    def __init__(self, X_train, X_val, encoding_dim, h1=0, h2=0, h3=0):
        train_ds = QuestionnaireDataset(X_train)
        val_ds = QuestionnaireDataset(X_val)
        self.train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        self.val_loader = DataLoader(val_ds, batch_size=32)

        input_dim = X_train.shape[1]
        self.model = BetaVAEModel(input_dim, encoding_dim, h1, h2, h3).to(device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.1, patience=5
        )

    def train(self, max_epochs=2000, patience=20,
              beta_max=0.5, kl_anneal_epochs=200, recon_weight=100,
              show_plot=False):
        best_val = float("inf")
        epochs_no_improve = 0
        train_losses, val_losses = [], []

        for epoch in range(max_epochs):
            self.model.train()
            total_train_loss = 0
            beta = min(beta_max, beta_max * (epoch / kl_anneal_epochs))

            for batch_x, _ in self.train_loader:
                batch_x = batch_x.to(device)
                self.optimizer.zero_grad()
                recon, mu, logvar = self.model(batch_x)
                rec_loss = recon_weight * self.criterion(recon, batch_x)
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_x.size(0)
                loss = rec_loss + beta * kl_loss
                loss.backward()
                self.optimizer.step()
                total_train_loss += loss.item() * batch_x.size(0)

            train_avg = total_train_loss / len(self.train_loader.dataset)
            train_losses.append(train_avg)

            self.model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for batch_x, _ in self.val_loader:
                    batch_x = batch_x.to(device)
                    recon, mu, logvar = self.model(batch_x)
                    rec_loss = recon_weight * self.criterion(recon, batch_x)
                    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_x.size(0)
                    total_val_loss += (rec_loss + beta * kl_loss).item() * batch_x.size(0)

            val_avg = total_val_loss / len(self.val_loader.dataset)
            val_losses.append(val_avg)
            self.scheduler.step(val_avg)

            print(f"Epoch {epoch+1}, Beta={beta:.4f}, Train={train_avg:.4f}, Val={val_avg:.4f}")

            if val_avg < best_val:
                best_val = val_avg
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

            if epoch % 5 == 0:
                batch_x, _ = next(iter(self.val_loader))
                batch_x = batch_x.to(device)
                with torch.no_grad():
                    mu, logvar = self.model.encode(batch_x)
                    z = self.model.reparameterize(mu, logvar).cpu().numpy()
                plt.figure(figsize=(8, 4))
                plt.hist(z.flatten(), bins=50, density=True, alpha=0.6)
                plt.title(f"Latent Histogram at Epoch {epoch}")
                plt.xlabel("z"); plt.ylabel("Density"); plt.show()
                plt.figure(figsize=(6, 6))
                stats.probplot(z.flatten(), dist="norm", plot=plt)
                plt.title(f"Q-Q Plot at Epoch {epoch}"); plt.show()
                shapiro_test = stats.shapiro(z.flatten())
                print(f"Epoch {epoch}: Shapiro-Wilk p-value = {shapiro_test.pvalue:.5f}")
                print(f"Epoch {epoch}: Latent mean = {z.mean():.4f}, std = {z.std():.4f}")

        if show_plot:
            plt.plot(train_losses, label="Train Loss")
            plt.plot(val_losses, label="Validation Loss")
            plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("BetaVAE Loss Curves")
            plt.legend(); plt.show()

    def evaluate_on_data(self, X):
        self.model.eval()
        X_np = np.asarray(X)
        with torch.no_grad():
            X_t = torch.tensor(X_np, dtype=torch.float32).to(device)
            recon, _, _ = self.model(X_t)
            z = self.model.get_latent(X_t).cpu().numpy()
            latent_vars = np.var(z, axis=0)
            total_var = np.var(X_np, axis=0).sum()
            evr = latent_vars / total_var
            recon_np = recon.cpu().numpy()
            rec_errors = np.mean((X_np - recon_np)**2, axis=1)
            recon_var = np.var(recon_np, axis=0).sum()
            total_evr = 1 - np.var((X_t - recon).cpu().numpy(), axis=0).sum() / total_var
        return z, rec_errors, evr, total_evr, recon_np


# ===== FactorVAE =====

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, z_dim, dropout_rate=0.0):
        super(Encoder, self).__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.LeakyReLU(0.01))
            layers.append(nn.Dropout(dropout_rate))
            prev = h
        layers.append(nn.Linear(prev, 2 * z_dim))  # 输出 mu 和 logvar
        self.net = nn.Sequential(*layers)
        self.z_dim = z_dim

    def forward(self, x):
        stats = self.net(x)
        mu = stats[:, : self.z_dim]
        logvar = stats[:, self.z_dim :]
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, z_dim, hidden_dims, output_dim, dropout_rate=0.0):
        super(Decoder, self).__init__()
        layers = []
        prev = z_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU(True))
            layers.append(nn.Dropout(dropout_rate))
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self, z_dim, hidden_dims=[200, 200]):
        super(Discriminator, self).__init__()
        layers = []
        prev = z_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            prev = h
        layers.append(nn.Linear(prev, 2))  # 两个 logit：Joint vs. Permuted
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


class FactorVAEModel(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder):
        super(FactorVAEModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar, z


# … 省略其他代码 …

class FactorVAETrainer:
    def __init__(self, model: FactorVAEModel, discriminator: Discriminator,
                 lr_vae=1e-3, lr_d=1e-3, beta_max=0.1, warmup_epochs=10, device=None, verbose=True):
        self.model = model
        self.discriminator = discriminator
        self.optim_vae = optim.Adam(self.model.parameters(), lr=lr_vae)
        self.optim_d = optim.Adam(self.discriminator.parameters(), lr=lr_d)
        self.beta_max = beta_max
        self.warmup_epochs = warmup_epochs
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.discriminator.to(self.device)
        self.history = {"recon": [], "kl": [], "tc": [], "d": [], "vae_total": [], "beta": []}
        self.val_loader = None  # 由外部赋值
        self.verbose = verbose  # 新增：是否在训练过程中打印 epoch 信息

    def _permute_latent(self, z: torch.Tensor) -> torch.Tensor:
        B, D = z.size()
        z_perm = []
        for j in range(D):
            idx = torch.randperm(B, device=z.device)
            z_perm.append(z[:, j][idx])
        return torch.stack(z_perm, dim=1)

    def train(self, data_loader, num_epochs):
        for epoch in range(1, num_epochs + 1):
            beta = self.beta_max * min(epoch / self.warmup_epochs, 1.0)
            epoch_recon = epoch_kl = epoch_tc = epoch_d = epoch_vae = 0.0
            n_batches = 0

            # —————— 训练阶段 ——————
            for batch in data_loader:
                n_batches += 1
                x = batch[0].to(self.device)

                # Discriminator 更新
                with torch.no_grad():
                    mu, logvar = self.model.encoder(x)
                    z = self.model.reparameterize(mu, logvar)
                z_det = z.detach()
                z_perm = self._permute_latent(z_det)
                logits_real = self.discriminator(z_det)
                logits_fake = self.discriminator(z_perm)
                labels_real = torch.zeros(z_det.size(0), dtype=torch.long, device=self.device)
                labels_fake = torch.ones(z_perm.size(0), dtype=torch.long, device=self.device)
                loss_d = 0.5 * (
                    F.cross_entropy(logits_real, labels_real)
                    + F.cross_entropy(logits_fake, labels_fake)
                )
                epoch_d += loss_d.item()
                self.optim_d.zero_grad()
                loss_d.backward()
                self.optim_d.step()

                # VAE 更新
                x_recon, mu, logvar, z = self.model(x)
                recon_loss = ((x_recon - x) ** 2).view(x.size(0), -1).sum(dim=1).mean()
                kl_loss = 0.5 * (logvar.exp() + mu**2 - 1 - logvar).sum(dim=1).mean()
                for p in self.discriminator.parameters():
                    p.requires_grad = False
                logits_z = self.discriminator(z)
                labels_trick = torch.ones(z.size(0), dtype=torch.long, device=self.device)
                tc_loss = F.cross_entropy(logits_z, labels_trick)
                for p in self.discriminator.parameters():
                    p.requires_grad = True
                loss_vae = recon_loss + kl_loss + beta * tc_loss
                epoch_recon += recon_loss.item()
                epoch_kl += kl_loss.item()
                epoch_tc += tc_loss.item()
                epoch_vae += loss_vae.item()
                self.optim_vae.zero_grad()
                loss_vae.backward()
                self.optim_vae.step()

            # 汇总当轮平均 loss
            self.history["recon"].append(epoch_recon / n_batches)
            self.history["kl"].append(epoch_kl / n_batches)
            self.history["tc"].append(epoch_tc / n_batches)
            self.history["d"].append(epoch_d / n_batches)
            self.history["vae_total"].append(epoch_vae / n_batches)
            self.history["beta"].append(beta)

            # —————— 打印 epoch 信息（加 if self.verbose） ——————
            if self.verbose:
                print(
                    f"Epoch {epoch}, Recon: {epoch_recon/n_batches:.4f}, "
                    f"KL: {epoch_kl/n_batches:.4f}, TC: {epoch_tc/n_batches:.4f}, "
                    f"D: {epoch_d/n_batches:.4f}"
                )

        # 最后画图（如果需要）
        if self.verbose:
            plt.figure(figsize=(8, 5))
            plt.plot(self.history["recon"], label="Reconstruction")
            plt.plot(self.history["kl"], label="KL")
            plt.plot(self.history["tc"], label="TC")
            plt.plot(self.history["vae_total"], label="VAE Total")
            plt.plot(self.history["d"], label="Discriminator")
            plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("FactorVAE Loss Curves")
            plt.legend(); plt.show()

            plt.figure(figsize=(6, 4))
            plt.plot(self.history["beta"], label="Beta")
            plt.xlabel("Epoch"); plt.ylabel("Beta Value"); plt.title("Beta Warm-up Schedule")
            plt.legend(); plt.show()


    def plot_history(self):
        plt.figure(figsize=(8, 5))
        plt.plot(self.history["recon"], label="Reconstruction")
        plt.plot(self.history["kl"], label="KL")
        plt.plot(self.history["tc"], label="TC")
        plt.plot(self.history["vae_total"], label="VAE Total")
        plt.plot(self.history["d"], label="Discriminator")
        plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("FactorVAE Loss Curves")
        plt.legend(); plt.show()

        plt.figure(figsize=(6, 4))
        plt.plot(self.history["beta"], label="Beta")
        plt.xlabel("Epoch"); plt.ylabel("Beta Value"); plt.title("Beta Warm-up Schedule")
        plt.legend(); plt.show()

    def evaluate_on_data(self, X):
        self.model.eval()
        X_np = np.asarray(X)
        X_t = torch.tensor(X_np, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            recon, mu, logvar, z = self.model(X_t)
            z_np = z.cpu().numpy()
            latent_vars = np.var(z_np, axis=0)
            total_var = np.var(X_np, axis=0).sum()
            evr = latent_vars / total_var
            recon_np = recon.cpu().numpy()
            rec_errors = np.mean((X_np - recon_np)**2, axis=1)
            recon_var = np.var(recon_np, axis=0).sum()
            total_evr = recon_var / total_var
        return z_np, rec_errors, evr, total_evr, recon_np


class FactorVAE:
    """
    与其他模型一致的高阶封装：
      - __init__(X_train, X_val, encoding_dim, layer1_neurons, layer2_neurons, layer3_neurons, n_clusters, lr_vae, lr_d, beta_max, warmup_epochs)
      - train(num_epochs=..., show_plot=False)
      - evaluate_on_data(X) -> (latent, rec_errors, evr, total_evr, recon_np)
    """
    def __init__(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        encoding_dim: int,
        layer1_neurons: int,
        layer2_neurons: int,
        layer3_neurons: int,
        n_clusters: int = 10,
        batch_size: int = 32,
        lr_vae: float = 1e-3,
        lr_d: float = 1e-3,
        beta_max: float = 0.1,
        warmup_epochs: int = 10,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size

        train_ds = QuestionnaireDataset(X_train)
        val_ds = QuestionnaireDataset(X_val)
        self.train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_ds, batch_size=self.batch_size)

        D = X_train.shape[1]
        z_dim = encoding_dim

        encoder = Encoder(
            input_dim=D,
            hidden_dims=[layer1_neurons, layer2_neurons, layer3_neurons],
            z_dim=z_dim,
            dropout_rate=0.0
        )

        decoder = Decoder(
            z_dim=z_dim,
            hidden_dims=[layer3_neurons, layer2_neurons, layer1_neurons],
            output_dim=D,
            dropout_rate=0.0
        )

        discriminator = Discriminator(
            z_dim=z_dim,
            hidden_dims=[100, 100]
        )

        model = FactorVAEModel(encoder, decoder)
        model.to(self.device)

        self.trainer = FactorVAETrainer(
            model=model,
            discriminator=discriminator,
            lr_vae=lr_vae,
            lr_d=lr_d,
            beta_max=beta_max,
            warmup_epochs=warmup_epochs,
            device=self.device
        )
        self.trainer.val_loader = self.val_loader

    def train(self, num_epochs: int = 200, show_plot: bool = False):
        self.trainer.train(
            data_loader=self.train_loader,
            num_epochs=num_epochs
        )
        if show_plot:
            self.trainer.plot_history()

    def evaluate_on_data(self, X: np.ndarray):
        return self.trainer.evaluate_on_data(X)
