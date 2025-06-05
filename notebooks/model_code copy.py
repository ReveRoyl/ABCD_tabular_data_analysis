from sklearn.decomposition import NMF
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import seaborn as sns
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class QuestionnaireDataset(Dataset):
    def __init__(self, data):
        self.data = torch.tensor(data, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.data[idx]

class AutoencoderModel(nn.Module):
    def __init__(
        self, input_dim, latent_dim, layer1_neurons, layer2_neurons, layer3_neurons
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
            nn.Linear(layer3_neurons, latent_dim),

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
    ):

        train_dataset = QuestionnaireDataset(X_train)
        val_dataset = QuestionnaireDataset(X_val)

        self.train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=32)

        # Model initialization
        input_dim = X_train.shape[1]
        latent_dim = encoding_dim
        self.model = AutoencoderModel(
            input_dim, latent_dim, layer1_neurons, layer2_neurons, layer3_neurons
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
            reconstructed
        )

class QuestionnaireDataset(Dataset):
    def __init__(self, data):
        self.data = torch.tensor(data, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.data[idx]

class SparseAutoencoderModel(nn.Module):
    """
    Sparse autoencoder with an L-layer encoder and decoder. 
    Uses a KL-divergence-based sparsity penalty on the latent representation. 
    """

    def __init__(
        self,
        input_dim,
        latent_dim,
        layer1_neurons,
        layer2_neurons,
        layer3_neurons,
        sparsity_target=0.05,
        beta=1.0,
    ):
        super(SparseAutoencoderModel, self).__init__()
        self.sparsity_target = sparsity_target
        self.beta = beta

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, layer1_neurons),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(layer1_neurons, layer2_neurons),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(layer2_neurons, layer3_neurons),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(layer3_neurons, latent_dim),
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
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def kl_divergence_loss(self, avg_activation):
        """
        KL Divergence-based sparsity penalty:
            KL(p || q) = p * log(p/q) + (1 - p)*log((1 - p)/(1 - q))
        """
        # We clamp the activation to avoid log(0) or log(∞)
        avg_activation = torch.clamp(avg_activation, 1e-10, 1.0)
        p = self.sparsity_target
        return torch.sum(
            p * torch.log(p / avg_activation)
            + (1 - p) * torch.log((1 - p) / (1 - avg_activation))
        )

    def get_latent_activations(self, x):
        """Utility to get the latent representation for an input batch x."""
        return self.encoder(x)

class SparseAutoencoder:
    def __init__(
        self,
        X_train,
        X_val,
        encoding_dim,
        layer1_neurons=0,
        layer2_neurons=0,
        layer3_neurons=0,
        sparsity_target=0.05,
        beta=1.0,
    ):

        # Build Datasets/Dataloaders
        train_dataset = QuestionnaireDataset(X_train)
        val_dataset = QuestionnaireDataset(X_val)

        self.train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=32)

        # Model initialization
        input_dim = X_train.shape[1]
        latent_dim = encoding_dim
        self.model = SparseAutoencoderModel(
            input_dim=input_dim,
            latent_dim=latent_dim,
            layer1_neurons=layer1_neurons,
            layer2_neurons=layer2_neurons,
            layer3_neurons=layer3_neurons,
            sparsity_target=sparsity_target,
            beta=beta,
        ).to(device)

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode="min", factor=0.1, patience=5)

    def get_model(self):
        return self.model

    def train(self, show_plot=False):
        best_val_loss = float("inf")
        patience = 20
        epochs_without_improvement = 0
        train_losses, val_losses = [], []

        for epoch in range(2000):
            # Training
            self.model.train()
            train_loss = 0
            for batch_features, _ in self.train_loader:
                batch_features = batch_features.to(device)
                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(batch_features)
                reconstruction_loss = self.criterion(outputs, batch_features)

                # ===== Sparsity penalty =====
                # Compute average activation of latent representation
                latent_repr = self.model.get_latent_activations(batch_features)
                avg_activation = torch.mean(torch.abs(latent_repr), dim=0)
                # Alternatively, use raw latent_repr if you do not want absolute value:
                # avg_activation = torch.mean(latent_repr, dim=0)

                kl_loss = self.model.kl_divergence_loss(avg_activation)
                # Weighted penalty
                sparsity_penalty = self.model.beta * kl_loss

                combined_loss = reconstruction_loss + sparsity_penalty
                combined_loss.backward()
                self.optimizer.step()
                train_loss += combined_loss.item() * batch_features.size(0)

            epoch_train_loss = train_loss / len(self.train_loader.dataset)
            train_losses.append(epoch_train_loss)

            # Validation
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_features, _ in self.val_loader:
                    batch_features = batch_features.to(device)
                    outputs = self.model(batch_features)
                    reconstruction_loss = self.criterion(outputs, batch_features)

                    # KL in validation
                    latent_repr = self.model.get_latent_activations(batch_features)
                    avg_activation = torch.mean(torch.abs(latent_repr), dim=0)
                    kl_loss = self.model.kl_divergence_loss(avg_activation)

                    combined_loss = reconstruction_loss + self.model.beta * kl_loss
                    val_loss += combined_loss.item() * batch_features.size(0)

            epoch_val_loss = val_loss / len(self.val_loader.dataset)
            val_losses.append(epoch_val_loss)

            # Step scheduler
            self.scheduler.step(epoch_val_loss)

            # Early stopping
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        # Plot if requested
        if show_plot:
            plt.plot(train_losses, label="Train Loss")
            plt.plot(val_losses, label="Validation Loss")
            plt.legend()
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Sparse AE Training/Validation Loss")
            plt.show()

    def evaluate_on_data(self, X):
        self.model.eval()
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
            reconstructed = self.model(X_tensor)

            # Get latent factors
            latent_factors = self.model.encoder(X_tensor).cpu().numpy()

            # Calculate reconstruction errors
            reconstruction_errors = torch.mean((X_tensor - reconstructed) ** 2, dim=1)
            reconstruction_errors = reconstruction_errors.cpu().numpy()

            # Calculate explained variances
            latent_variances = np.var(latent_factors, axis=0)

            # Total variance of the original data
            total_variance = np.var(X, axis=0).sum()
            explained_variance_ratios = latent_variances / total_variance

            # Total explained variance ratio
            reconstruction_variance = np.var(reconstructed.cpu().numpy(), axis=0).sum()
            # explained_variance_ratio_total = reconstruction_variance / total_variance
            explained_variance_ratio_total = 1 - np.var((X_tensor - reconstructed).cpu().numpy(), axis=0).sum() / total_variance


        return (
            latent_factors,
            reconstruction_errors,
            explained_variance_ratios,
            explained_variance_ratio_total,
            reconstructed,
        )

    def export_to_onnx(self, X_train, onnx_path):
        # Use the first training sample as dummy input
        device = next(self.model.parameters()).device
        dummy_input = torch.tensor(X_train[:1], dtype=torch.float32).to(device)
        
        # Export the model to ONNX format
        torch.onnx.export(
            self.model,  # Trained model
            dummy_input,  # Use real data as example input
            onnx_path,  # Output file path
            input_names=["input"],  # Input name
            output_names=["reconstructed"],  # Output name
            dynamic_axes={
                "input": {0: "batch_size"},
                "reconstructed": {0: "batch_size"},
            },  # Dynamic batch size support
            opset_version=11,  # ONNX opset version
        )

        print(f"Model exported to {onnx_path}")

        # Start Netron to visualize the model
        netron.start(onnx_path)

class QuestionnaireDataset(Dataset):
    def __init__(self, data):
        self.data = torch.tensor(data, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.data[idx]

class ClusteringLayer(nn.Module):
    def __init__(self, n_clusters, latent_dim):
        super(ClusteringLayer, self).__init__()
        self.cluster_centers = nn.Parameter(torch.randn(n_clusters, latent_dim))

    def forward(self, z):
        logits = torch.matmul(z, self.cluster_centers.t())
        return F.softmax(logits, dim=1)

class COAEModel(nn.Module):
    def __init__(self, input_dim, latent_dim, layer1, layer2, layer3, n_clusters):
        super(COAEModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, layer1), nn.ReLU(),
            nn.Linear(layer1, layer2), nn.ReLU(),
            nn.Linear(layer2, layer3), nn.ReLU(),
            nn.Linear(layer3, latent_dim), nn.Sigmoid()  # ✅ 添加 sigmoid 控制输出范围
        )
        self.clustering_layer = ClusteringLayer(n_clusters, latent_dim)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        enc0, enc2, enc4, enc6 = self.encoder[0], self.encoder[2], self.encoder[4], self.encoder[6]
        x = F.linear(z, enc6.weight.t()); x = F.relu(x)
        x = F.linear(x, enc4.weight.t()); x = F.relu(x)
        x = F.linear(x, enc2.weight.t()); x = F.relu(x)
        x = F.linear(x, enc0.weight.t())
        return x

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        p = self.clustering_layer(z)
        return x_hat, z, p

def orthogonality_loss(z):
    z = z - z.mean(dim=0, keepdim=True)
    cov = (z.T @ z) / z.size(0)
    I = torch.eye(z.size(1)).to(z.device)
    return torch.sum((cov - I) ** 2)

def sparsity_kl_divergence(rho, rho_hat):
    rho_hat = torch.clamp(rho_hat, 1e-10, 1 - 1e-10)
    return torch.sum(
        rho * torch.log(rho / rho_hat) + 
        (1 - rho) * torch.log((1 - rho) / (1 - rho_hat))
    )

def compute_target_distribution(p):
    weight = (p ** 2) / torch.sum(p, dim=0)
    return (weight.t() / torch.sum(weight, dim=1)).t()

class COAETrainer:
    def __init__(self, X_train, X_val, latent_dim, layer1, layer2, layer3, n_clusters):
        self.train_loader = DataLoader(QuestionnaireDataset(X_train), batch_size=32, shuffle=True)
        self.val_loader = DataLoader(QuestionnaireDataset(X_val), batch_size=32)
        input_dim = X_train.shape[1]

        self.model = COAEModel(input_dim, latent_dim, layer1, layer2, layer3, n_clusters).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=5)
        self.criterion = nn.MSELoss()

        self.lambda_orth = 1e-1
        self.mu_clust = 1.0
        self.lambda_sparsity = 1e-4
        self.sparsity_target = 0.8

    def train(self, show_plot=False):
        best_val_loss = float('inf')
        patience, epochs_no_improve = 50, 0
        train_losses, val_losses = [], []

        for epoch in range(200):
            self.model.train()
            total_train_loss = 0
            for batch_features, _ in self.train_loader:
                batch_features = batch_features.to(device)
                self.optimizer.zero_grad()

                x_hat, z, p = self.model(batch_features)
                q = compute_target_distribution(p).detach()

                recon_loss = self.criterion(x_hat, batch_features)
                orth_loss = orthogonality_loss(z)
                clust_loss = F.kl_div(p.log(), q, reduction='batchmean')
                rho_hat = torch.mean(torch.sigmoid(z), dim=0)
                sparse_loss = sparsity_kl_divergence(self.sparsity_target, rho_hat)
                

                loss = recon_loss + \
                       self.lambda_orth * orth_loss + \
                       self.mu_clust * clust_loss + \
                       self.lambda_sparsity * sparse_loss

                loss.backward()
                self.optimizer.step()
                total_train_loss += loss.item() * batch_features.size(0)

            train_losses.append(total_train_loss / len(self.train_loader.dataset))

            # 验证
            self.model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for batch_features, _ in self.val_loader:
                    batch_features = batch_features.to(device)
                    x_hat, z, p = self.model(batch_features)
                    q = compute_target_distribution(p)

                    recon_loss = self.criterion(x_hat, batch_features)
                    orth_loss = orthogonality_loss(z)
                    clust_loss = F.kl_div(p.log(), q, reduction='batchmean')
                    rho_hat = torch.mean(torch.sigmoid(z), dim=0)
                    sparse_loss = sparsity_kl_divergence(self.sparsity_target, rho_hat)

                    val_loss = recon_loss + \
                               self.lambda_orth * orth_loss + \
                               self.mu_clust * clust_loss + \
                               self.lambda_sparsity * sparse_loss

                    total_val_loss += val_loss.item() * batch_features.size(0)

            val_loss_avg = total_val_loss / len(self.val_loader.dataset)
            val_losses.append(val_loss_avg)
            self.scheduler.step(val_loss_avg)
            print(f"  KL 稀疏损失: {sparse_loss.item():.4f}")
            print(f"Epoch {epoch+1}: Train Loss={train_losses[-1]:.4f}, Val Loss={val_loss_avg:.4f}")

            if val_loss_avg < best_val_loss:
                best_val_loss = val_loss_avg
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        if show_plot:
            plt.plot(train_losses, label="Train")
            plt.plot(val_losses, label="Validation")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.title("COAE Loss Curves")
            plt.show()

    def get_latent_representation(self, X):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
            return self.model.encode(X_tensor).cpu().numpy()

class QuestionnaireDataset(Dataset):
    def __init__(self, data):
        self.data = torch.tensor(data, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # VAE 依旧是自监督任务，输入 = 目标
        return self.data[idx], self.data[idx]

class VAEModel(nn.Module):
    def __init__(
        self, input_dim, latent_dim,
        layer1_neurons, layer2_neurons, layer3_neurons
    ):
        super(VAEModel, self).__init__()

        # ---------- Encoder ----------
        self.encoder_net = nn.Sequential(
            nn.Linear(input_dim, layer1_neurons),
            nn.LeakyReLU(0.01),
            nn.Linear(layer1_neurons, layer2_neurons),
            nn.LeakyReLU(0.01),
            nn.Linear(layer2_neurons, layer3_neurons),
            nn.LeakyReLU(0.01),
        )
        self.enc_mu     = nn.Linear(layer3_neurons, latent_dim)
        self.enc_logvar = nn.Linear(layer3_neurons, latent_dim)

        # ---------- Decoder ----------
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, layer3_neurons),
            nn.LeakyReLU(0.01),
            nn.Linear(layer3_neurons, layer2_neurons),
            nn.LeakyReLU(0.01),
            nn.Linear(layer2_neurons, layer1_neurons),
            nn.LeakyReLU(0.01),
            nn.Linear(layer1_neurons, input_dim)
        )

    # ---------- Helpers ----------
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
    def __init__(
        self, X_train, X_val, encoding_dim,
        layer1_neurons=0, layer2_neurons=0, layer3_neurons=0,
        beta_kl=1.0  # 可调节 KL 权重
    ):
        self.beta_kl = beta_kl

        train_ds = QuestionnaireDataset(X_train)
        val_ds   = QuestionnaireDataset(X_val)
        self.train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        self.val_loader   = DataLoader(val_ds,   batch_size=32)

        input_dim  = X_train.shape[1]
        latent_dim = encoding_dim
        self.model = VAEModel(
            input_dim, latent_dim,
            layer1_neurons, layer2_neurons, layer3_neurons
        ).to(device)

        self.recon_loss_fn = nn.MSELoss(reduction="sum")  # sum → 方便 KL 归一化
        self.optim     = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.scheduler = ReduceLROnPlateau(self.optim, mode="min", factor=0.1, patience=5)

        self.explained_variance_ratio_total_value = None

    # ---------- Public ----------
    def get_model(self):
        return self.model

    def train(self, show_plot=False):
        best_val = float("inf")
        patience, wait = 20, 0
        train_hist, val_hist = [], []

        for epoch in range(2000):
            # ----- train step -----
            self.model.train()
            epoch_loss = 0
            for x, _ in self.train_loader:
                x = x.to(device)
                self.optim.zero_grad()

                recon, mu, logvar = self.model(x)
                # 重构误差（按样本平均）
                recon_loss = self.recon_loss_fn(recon, x) / x.size(0)
                # KL 散度
                kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + self.beta_kl * kl_loss

                loss.backward()
                self.optim.step()

                epoch_loss += loss.item() * x.size(0)

            train_hist.append(epoch_loss / len(self.train_loader.dataset))

            # ----- val step -----
            self.model.eval()
            val_loss_acc = 0
            with torch.no_grad():
                for x, _ in self.val_loader:
                    x = x.to(device)
                    recon, mu, logvar = self.model(x)
                    recon_loss = self.recon_loss_fn(recon, x) / x.size(0)
                    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                    loss = recon_loss + self.beta_kl * kl_loss
                    val_loss_acc += loss.item() * x.size(0)

            val_loss = val_loss_acc / len(self.val_loader.dataset)
            val_hist.append(val_loss)
            self.scheduler.step(val_loss)

            # Early-stopping
            if val_loss < best_val:
                best_val = val_loss
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        if show_plot:
            plt.plot(train_hist, label="Train")
            plt.plot(val_hist,   label="Val")
            plt.legend(); plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("VAE Loss")
            plt.show()

    # 与原 evaluate_on_data 相同接口，内部逻辑切到 VAE
    def evaluate_on_data(self, X):
        self.model.eval()
        with torch.no_grad():
            X_np = np.asarray(X)
            X_tensor = torch.tensor(X_np, dtype=torch.float32).to(device)

            recon, mu, logvar = self.model(X_tensor)
            latent_factors = mu.cpu().numpy()  # 使用均值向量作为潜在表示

            # 下面的解释方差计算保持不变
            latent_variances = np.var(latent_factors, axis=0)
            total_variance  = np.var(X_np, axis=0).sum()
            explained_variance_ratios = latent_variances / total_variance

            rec_errors = torch.mean((X_tensor - recon) ** 2, dim=1).cpu().numpy()
            recon_variance = np.var(recon.cpu().numpy(), axis=0).sum()
            # evr_total = recon_variance / total_variance
            evr_total = 1 - np.var((X_tensor - recon).cpu().numpy(), axis=0).sum() / total_variance

            

        return (
            latent_factors,          # 潜在均值
            rec_errors,              # 样本重构误差
            explained_variance_ratios,
            evr_total,
            recon.cpu().numpy()      # 重构数据
        )

def is_nni_running():
    """If NNI is running, return True; otherwise, return False."""
    return "NNI_PLATFORM" in os.environ

class QuestionnaireDataset(Dataset):
    def __init__(self, data):
        self.data = torch.tensor(data, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.data[idx]

class BetaVAEModel(nn.Module):
    def __init__(self, input_dim, latent_dim, layer1_neurons, layer2_neurons, layer3_neurons):
        super(BetaVAEModel, self).__init__()

        # Encoder
        self.encoder_net = nn.Sequential(
            nn.Linear(input_dim, layer1_neurons),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(layer1_neurons, layer2_neurons),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(layer2_neurons, layer3_neurons),
            nn.LeakyReLU(negative_slope=0.01),
        )
        self.fc_mu = nn.Linear(layer3_neurons, latent_dim)
        self.fc_logvar = nn.Linear(layer3_neurons, latent_dim)

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

    def encode(self, x):
        h = self.encoder_net(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z)
        return reconstructed, mu, logvar

    def get_latent(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return z

class BetaVAE:
    def __init__(
        self,
        X_train,
        X_val,
        encoding_dim,
        layer1_neurons=0,
        layer2_neurons=0,
        layer3_neurons=0,
    ):
        train_dataset = QuestionnaireDataset(X_train)
        val_dataset = QuestionnaireDataset(X_val)

        self.train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=32)

        input_dim = X_train.shape[1]
        latent_dim = encoding_dim
        self.model = BetaVAEModel(
            input_dim, latent_dim, layer1_neurons, layer2_neurons, layer3_neurons
        ).to(device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode="min", factor=0.1, patience=5)
        self.explained_variance_ratio_total_value = None

    def get_model(self):
        return self.model

    def train(self, show_plot=False, beta_max=0.5, kl_anneal_epochs=200, recon_loss_weight =100):
        best_val_loss = float("inf")
        patience = 20
        epochs_without_improvement = 0
        train_losses, val_losses = [], []

        for epoch in range(2000):
            self.model.train()
            train_loss = 0

            # KL annealing: 让 beta 从 0 慢慢涨到 beta_max
            beta = min(beta_max, beta_max * epoch / kl_anneal_epochs)

            for batch_features, _ in self.train_loader:
                batch_features = batch_features.to(device)
                self.optimizer.zero_grad()
                reconstructed, mu, logvar = self.model(batch_features)

                reconstruction_loss = self.criterion(reconstructed, batch_features)
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_features.size(0)

                combined_loss = recon_loss_weight*reconstruction_loss + beta * kl_loss

                combined_loss.backward()
                self.optimizer.step()
                train_loss += combined_loss.item() * batch_features.size(0)
            train_losses.append(train_loss / len(self.train_loader.dataset))

            # Validation
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_features, _ in self.val_loader:
                    batch_features = batch_features.to(device)
                    reconstructed, mu, logvar = self.model(batch_features)

                    reconstruction_loss = self.criterion(reconstructed, batch_features)
                    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_features.size(0)
                    combined_loss = recon_loss_weight*reconstruction_loss + beta * kl_loss

                    val_loss += combined_loss.item() * batch_features.size(0)
            val_losses.append(val_loss / len(self.val_loader.dataset))
            self.scheduler.step(val_loss)

            print(f"Epoch {epoch+1}, Beta {beta:.4f}, Train Loss {train_losses[-1]:.4f}, Val Loss {val_losses[-1]:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

            # 每5个epoch监控潜变量分布
            if epoch % 5 == 0:
                batch_features, _ = next(iter(self.val_loader))
                batch_features = batch_features.to(device)
                self.model.eval()
                with torch.no_grad():
                    mu, logvar = self.model.encode(batch_features)
                    std = torch.exp(0.5 * logvar)
                    eps = torch.randn_like(std)
                    z = mu + eps * std
                    z = z.cpu().numpy()

                # 潜变量直方图
                plt.figure(figsize=(8, 4))
                plt.hist(z.flatten(), bins=50, density=True, alpha=0.6)
                plt.title(f"Latent variable histogram at epoch {epoch}")
                plt.xlabel("z value")
                plt.ylabel("Density")
                plt.show()

                # Q-Q图
                import scipy.stats as stats
                plt.figure(figsize=(6, 6))
                stats.probplot(z.flatten(), dist="norm", plot=plt)
                plt.title(f"Q-Q plot at epoch {epoch}")
                plt.show()

                # Shapiro-Wilk检验
                shapiro_test = stats.shapiro(z.flatten())
                print(f"Epoch {epoch}: Shapiro-Wilk p-value = {shapiro_test.pvalue:.5f}")
                print(f"Epoch {epoch}: Latent mean = {z.mean():.4f}, std = {z.std():.4f}")

        if show_plot:
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
            if not isinstance(X, np.ndarray):
                X = np.array(X)
            X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

            reconstructed, _, _ = self.model(X_tensor)

            latent_factors = self.model.get_latent(X_tensor).cpu().numpy()

            latent_variances = np.var(latent_factors, axis=0)
            total_variance = np.var(X, axis=0).sum()
            explained_variance_ratios = latent_variances / total_variance

            reconstruction_errors = torch.mean((X_tensor - reconstructed) ** 2, dim=1).cpu().numpy()
            reconstruction_variance = np.var(reconstructed.cpu().numpy(), axis=0).sum()
            # explained_variance_ratio_total = reconstruction_variance / total_variance
            explained_variance_ratio_total = 1 - np.var((X_tensor - reconstructed).cpu().numpy(), axis=0).sum() / total_variance

        return (
            latent_factors,
            reconstruction_errors,
            explained_variance_ratios,
            explained_variance_ratio_total,
            reconstructed
        )

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, z_dim, dropout_rate=0.0):
        """
        编码器：将输入映射到潜在空间的均值和log-方差。
        参数:
        - input_dim: 输入数据的维度 (例如Flatten后长度，对图像可为 C*H*W)。
        - hidden_dims: 隐藏层神经元数量列表。
        - z_dim: 潜在空间维度。
        """
        super(Encoder, self).__init__()
        self.dropout_rate = dropout_rate
        # 构建全连接层序列
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU(True))
            layers.append(nn.Dropout(self.dropout_rate))
            prev_dim = h_dim
        # 最终层输出2*z_dim，因为要同时输出均值和log方差
        layers.append(nn.Linear(prev_dim, 2 * z_dim))
        self.net = nn.Sequential(*layers)
        self.z_dim = z_dim

    def forward(self, x):
        """
        前向传播：输出潜在分布的均值mu和对数方差logvar。
        """
        stats = self.net(x)                # 得到长度为2*z_dim的向量
        mu = stats[:, :self.z_dim]         # 前半部分作为均值
        logvar = stats[:, self.z_dim:]     # 后半部分作为log-方差
        # FactorVAE新增: 输出均值和log方差用于随机采样潜在向量
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, z_dim, hidden_dims, output_dim, dropout_rate=0.0):

        """
        解码器：将潜在向量重建为输出。
        参数:
        - z_dim: 潜在空间维度。
        - hidden_dims: 隐藏层神经元数量列表（解码层）。
        - output_dim: 重建输出的维度（例如Flatten后长度）。
        """
        super(Decoder, self).__init__()
        self.dropout_rate = dropout_rate
        layers = []
        prev_dim = z_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU(True))
            layers.append(nn.Dropout(self.dropout_rate))
            prev_dim = h_dim
        # 最终层输出为原始数据维度大小
        layers.append(nn.Linear(prev_dim, output_dim))
        # 如果输出是图像像素，可视需要添加激活如Sigmoid；这里假设后续损失会处理
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        """
        前向传播：将潜在向量z解码为重建输出。
        """
        x_recon = self.net(z)
        return x_recon

class Discriminator(nn.Module):
    def __init__(self, z_dim, hidden_dims=[200, 200]):
        """
        判别器：判断输入的潜在向量是否来自真实编码分布q(z)或独立因素分布prod(q(z_j))。
        参数:
        - z_dim: 潜在空间维度。
        - hidden_dims: 隐藏层神经元数量列表。
        """
        super(Discriminator, self).__init__()
        layers = []
        prev_dim = z_dim
        # 构建一系列全连接层（使用LeakyReLU作为激活）
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            prev_dim = h_dim
        # 最终输出2分类（两个logits，对应标签0或1）
        layers.append(nn.Linear(prev_dim, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        """
        前向传播：输出对数几率(logits)值（大小为2，表示属于各类别的未归一化分数）。
        """
        logits = self.net(z)
        return logits

class FactorVAE(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder):
        """
        FactorVAE模型：包含Encoder和Decoder。
        """
        super(FactorVAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    def encode(self, x):
        return self.encoder(x)


    def reparameterize(self, mu, logvar):
        """
        FactorVAE新增: 重参数化采样。根据给定的均值mu和对数方差logvar采样潜在向量z。
        公式: z = mu + sigma * epsilon, 其中epsilon ~ N(0, I), sigma = exp(0.5 * logvar)
        """
        # 计算标准差
        std = torch.exp(0.5 * logvar)
        # 从标准正态分布采样epsilon，与std形状相同 (保持device一致)
        eps = torch.randn_like(std)    # 【FactorVAE新增】确保epsilon在相同设备上采样
        # 返回采样的潜在变量
        return mu + std * eps

    def forward(self, x):
        """
        前向传播：输出重建结果以及潜在分布参数和采样结果。
        返回: (x_recon, mu, logvar, z)
        """
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar, z

class Trainer:
    def __init__(self, model: FactorVAE, discriminator: Discriminator,
                 lr_vae=1e-3, lr_d=1e-3, beta_max=0.1, warmup_epochs=10, device=None):
        """
        Trainer初始化:
        - model: FactorVAE模型 (包含encoder和decoder)
        - discriminator: 判别器模型
        - lr_vae: VAE部分的学习率
        - lr_d: 判别器的学习率
        - beta_max: β的最大值 (最终惩罚系数)
        - warmup_epochs: β从0涨到beta_max所持续的epoch数
        - device: 训练使用的设备 (cpu或cuda)
        """
        self.model = model
        self.discriminator = discriminator
        self.beta_max = beta_max
        self.warmup_epochs = warmup_epochs
        # 优化器
        self.optim_vae = torch.optim.Adam(self.model.parameters(), lr=lr_vae)
        self.optim_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr_d)
        # 设置设备
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.model.to(device)
        self.discriminator.to(device)
        # 用一个 dict 来存所有曲线
        self.history = {
            'recon': [],    # 重建 loss
            'kl': [],       # KL loss
            'tc': [],       # Total Correlation loss
            'd': [],        # 判别器 loss
            'vae_total': [],# VAE 综合 loss
            'beta': [],     # 每个 epoch 的 β 值
        }
    def _permute_latent(self, z):
        """
        FactorVAE新增: 打乱潜在向量z各维度。用于生成独立分布样本\bar{q}(z)。
        输入: z: Tensor shape [batch, z_dim]
        输出: 打乱每个维度后的z_perm (每一列来自原z的一组随机重排)
        """
        # 对每个潜在维度独立随机打乱
        B, D = z.size()
        z_perm = []
        for j in range(D):
            idx = torch.randperm(B, device=self.device)  # 在当前设备上生成随机索引
            z_j = z[:, j]          # 取出第j维 (shape [B])
            z_perm_j = z_j[idx]    # 打乱该维的所有batch样本顺序
            z_perm.append(z_perm_j)
        z_perm = torch.stack(z_perm, dim=1)
        return z_perm

    def train(self, data_loader, num_epochs):
        """
        训练模型。每个epoch逐步增加β值，并在每个训练step交替优化VAE和判别器。
        """
        self.model.train()
        self.discriminator.train()
        for epoch in range(1, num_epochs+1):
            # 线性暖启动: 计算当前epoch使用的β (从0增加到beta_max)
            if epoch <= self.warmup_epochs:
                beta = self.beta_max * (epoch / float(self.warmup_epochs))
            else:
                beta = self.beta_max
                        # 计算 warm-up 后的 beta
            # 累加本 epoch 的各项 loss
            epoch_recon = epoch_kl = epoch_tc = epoch_d = epoch_vae = 0.0
            n_batches = 0
            for batch in data_loader:
                n_batches += 1
                if isinstance(batch, (list, tuple)):
                    x = batch[0].to(self.device)
                else:
                    x = batch.to(self.device)
            # 1. 计算当前batch的大小                
                # ----------------------
                # 判别器更新步骤 (D-step)
                # ----------------------
                # 1. 从当前模型获取潜在向量 z（编码器前向+采样），并生成打乱的z_perm
                # 注意：用 detach() 确保在D步骤中不更新VAE的编码器/解码器
                with torch.no_grad():
                    mu, logvar = self.model.encoder(x)
                    z = self.model.reparameterize(mu, logvar)
                z_det = z.detach()                      # detach之后的z用于判别器训练
                z_perm = self._permute_latent(z_det)    # 打乱得到z_perm
                # 2. 计算判别器预测
                logits_z = self.discriminator(z_det)         # 判别器预测真实z的logits
                logits_z_perm = self.discriminator(z_perm)   # 判别器预测独立z_perm的logits
                # 3. 判别器损失: 真实z标签为0，打乱z标签为1
                # 使用交叉熵损失来区分，两者损失取平均
                labels_z = torch.zeros(logits_z.size(0), dtype=torch.long, device=self.device)      # 全0标签
                labels_z_perm = torch.ones(logits_z_perm.size(0), dtype=torch.long, device=self.device)  # 全1标签
                loss_d_real = F.cross_entropy(logits_z, labels_z)         # 真实z的判别损失
                loss_d_fake = F.cross_entropy(logits_z_perm, labels_z_perm)  # 打乱z的判别损失
                loss_d = 0.5 * (loss_d_real + loss_d_fake)                # 综合判别器损失
                # … 计算 loss_d …
                epoch_d += loss_d.item()
                # 4. 反向传播和优化判别器
                self.optim_d.zero_grad()
                loss_d.backward()
                self.optim_d.step()

                # ----------------------
                # VAE更新步骤 (VAE-step)
                # ----------------------
                # 5. 前向传播获取模型输出和重新采样的z（注意：需要重新计算以建立梯度图）
                x_recon, mu, logvar, z = self.model(x)
                # 6. 计算重建损失 (例如MSE误差)
                # 将每个样本的重建误差求和再取平均
                recon_loss = ((x_recon - x) ** 2).view(x.size(0), -1).sum(dim=1).mean()
                # 7. 计算KL散度损失 (使用解析形式)
                # KL(q(z|x) || p(z)) = 0.5 * sum( exp(logvar) + mu^2 - 1 - logvar )
                kl_loss = 0.5 * (logvar.exp() + mu**2 - 1. - logvar).sum(dim=1).mean()
                # 8. 计算VAE的对抗损失（Total Correlation近似）
                # 判别器在此时的参数保持不变，仅用于提供梯度给VAE
                # 将判别器参数暂时禁止梯度，以确保只优化VAE
                for p in self.discriminator.parameters():
                    p.requires_grad = False
                logits_z = self.discriminator(z)   # 判别器对当前z的输出
                # VAE希望“欺骗”判别器，把真实z也判为类别1（即独立分布样本）
                labels_trick = torch.ones(logits_z.size(0), dtype=torch.long, device=self.device)
                tc_loss = F.cross_entropy(logits_z, labels_trick)
                # 恢复判别器参数的requires_grad，准备下次判别器更新
                for p in self.discriminator.parameters():
                    p.requires_grad = True
                # 9. 综合VAE损失: 重建误差 + KL散度 + β * 对抗损失
                loss_vae = recon_loss + kl_loss + beta * tc_loss   # FactorVAE新增: β权衡TC损失
                # … 计算 recon_loss, kl_loss, tc_loss, loss_vae …
                epoch_recon += recon_loss.item()
                epoch_kl    += kl_loss.item()
                epoch_tc    += tc_loss.item()
                epoch_vae   += loss_vae.item()
                # 10. 反向传播和优化VAE (encoder+decoder)
                self.optim_vae.zero_grad()
                loss_vae.backward()
                self.optim_vae.step()
            print(f"Epoch {epoch}, Batch {n_batches}, Recon Loss: {recon_loss.item():.4f}, KL Loss: {kl_loss.item():.4f}, TC Loss: {tc_loss.item():.4f}, D Loss: {loss_d.item():.4f}")
            # 取平均并存入 history
            self.history['recon'].append(epoch_recon   / n_batches)
            self.history['kl'].append(epoch_kl          / n_batches)
            self.history['tc'].append(epoch_tc          / n_batches)
            self.history['d'].append(epoch_d            / n_batches)
            self.history['vae_total'].append(epoch_vae  / n_batches)
            self.history['beta'].append(beta)
        self.plot_history()
        
    def plot_history(self):
        """使用 Matplotlib 绘制训练过程中的 loss 曲线和 β 升温曲线。"""
        # Loss 曲线
        plt.figure(figsize=(8,5))
        plt.plot(self.history['recon'],    label='Reconstruction')
        plt.plot(self.history['kl'],       label='KL')
        plt.plot(self.history['tc'],       label='TC')
        plt.plot(self.history['vae_total'],label='VAE Total')
        plt.plot(self.history['d'],        label='Discriminator')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training Loss Curves')
        plt.show()

        # β 升温曲线
        plt.figure(figsize=(6,4))
        plt.plot(self.history['beta'], label='β Value')
        plt.xlabel('Epoch')
        plt.ylabel('β')
        plt.title('Warm-up β Schedule')
        plt.show()
    def evaluate_on_data(self, X):
        """
        在数据 X（numpy array 或 list）上评估：
        - 重建结果 reconstructed
        - 潜在因子 latent_factors
        - 每个潜在因子的解释方差比 explained_variance_ratios
        - 总体解释方差比 explained_variance_ratio_total
        - 每个样本的重建误差 reconstruction_errors

        返回：
          (latent_factors, reconstruction_errors,
           explained_variance_ratios,
           explained_variance_ratio_total,
           reconstructed)
        """
        device = self.device
        self.model.eval()

        # 转为 numpy / Tensor
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        X_tensor = torch.tensor(X, dtype=torch.float32, device=device)

        with torch.no_grad():
            # forward 直接返回 (recon, mu, logvar, z)
            reconstructed, mu, logvar, z = self.model(X_tensor)

            # 将 z 转为 numpy
            latent_factors = z.cpu().numpy()

            # 每个因子的方差与总方差
            latent_variances = np.var(latent_factors, axis=0)
            total_variance = np.var(X, axis=0).sum()
            explained_variance_ratios = latent_variances / total_variance

            # 样本级别重建均方误差
            reconstruction_errors = (
                (X_tensor - reconstructed).pow(2)
                .mean(dim=1)
                .cpu()
                .numpy()
            )

            # 重建输出整体方差贡献
            reconstruction_variance = np.var(reconstructed.cpu().numpy(), axis=0).sum()
            # explained_variance_ratio_total = reconstruction_variance / total_variance
            explained_variance_ratio_total = 1 - np.var((X_tensor - reconstructed).cpu().numpy(), axis=0).sum() / total_variance

        return (
            latent_factors,
            reconstruction_errors,
            explained_variance_ratios,
            explained_variance_ratio_total,
            reconstructed.cpu().numpy()
        )

class DeepVAE(nn.Module):
    def __init__(self, input_dim, layer1, layer2, layer3, latent_dim, init_W=None):
        super(DeepVAE, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, layer1),
            # nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(layer1, layer2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(layer2, layer3),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(layer3, 2 * latent_dim),  # 输出 mu 与 logvar
        )
        self.latent_dim = latent_dim
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, layer3),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(layer3, layer2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(layer2, layer1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(layer1, input_dim),
        )
        # 用 NMF 字典初始化 Decoder 最后一层
        if init_W is not None:
            # init_W: (latent_dim, input_dim) -> 需要转置匹配 PyTorch Linear weight (out_features, in_features)
            final_lin = self.decoder[-1]
            final_lin.weight.data.copy_(torch.tensor(init_W, dtype=torch.float32))
            final_lin.bias.data.zero_()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        stats = self.encoder(x)
        mu = stats[:, :self.latent_dim]
        logvar = stats[:, self.latent_dim:]
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar, z

def evaluate_on_data(model, device, X):
    model.eval()
    if not isinstance(X, np.ndarray):
        X = np.array(X)
    X_tensor = torch.tensor(X, dtype=torch.float32, device=device)

    with torch.no_grad():
        reconstructed, mu, logvar, z = model(X_tensor)
        latent_factors = z.cpu().numpy()

        # 计算残差
        X_hat = reconstructed.cpu().numpy()
        residuals = X - X_hat

        # 对每个特征维度计算 VAF
        var_original = np.var(X, axis=0)
        var_residual = np.var(residuals, axis=0)
        # VAF = 1 - Var(residual) / Var(original)
        explained_variance_ratios = 1 - (var_residual / var_original)

        # 总体 VAF
        total_var_original = var_original.sum()
        total_var_residual = var_residual.sum()
        explained_variance_ratio_total = 1 - (total_var_residual / total_var_original)

        reconstruction_errors = (
            (X_tensor - reconstructed).pow(2)
            .mean(dim=1)
            .cpu()
            .numpy()
        )

    return (
        latent_factors,
        reconstruction_errors,
        explained_variance_ratios,
        explained_variance_ratio_total,
        X_hat
    )

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, z_dim, dropout_rate=0.0):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.LeakyReLU(0.01), nn.Dropout(dropout_rate)]
            prev = h
        layers.append(nn.Linear(prev, 2*z_dim))
        self.net = nn.Sequential(*layers)
        self.z_dim = z_dim

    def forward(self, x):
        stats = self.net(x)
        mu, logvar = stats[:, :self.z_dim], stats[:, self.z_dim:]
        return mu, logvar

class NMFDecoder(nn.Module):
    """只含一层线性层，权重矩阵即 NMF 字典。"""
    def __init__(self, init_W):
        super().__init__()
        # init_W: numpy array of shape (z_dim, input_dim)
        self.W = nn.Parameter(torch.tensor(init_W, dtype=torch.float32), requires_grad=True)

    def forward(self, z):
        # z: (batch,z_dim), W: (z_dim,input_dim)
        return torch.matmul(z, self.W)

class FactorVAE_NMF(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar, z

class Discriminator(nn.Module):
    def __init__(self, z_dim, hidden_dims=[200,200]):
        super().__init__()
        layers = []
        prev = z_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.LeakyReLU(0.2)]
            prev = h
        layers.append(nn.Linear(prev, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)

class Trainer:
    def __init__(self, model, discriminator,
                 lr_vae=1e-3, lr_d=1e-3, beta_max=0.1,
                 warmup_epochs=10, device=None):
        self.model = model
        self.discriminator = discriminator
        self.optim_vae = torch.optim.Adam(self.model.parameters(), lr=lr_vae)
        self.optim_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr_d)
        self.beta_max = beta_max
        self.warmup_epochs = warmup_epochs
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.discriminator.to(self.device)

    def _permute_latent(self, z):
        B,D = z.size()
        z_perm = []
        for j in range(D):
            idx = torch.randperm(B, device=z.device)
            z_perm.append(z[:,j][idx])
        return torch.stack(z_perm, dim=1)

    def train(self, data_loader, num_epochs):
        for epoch in range(1, num_epochs+1):
            beta = self.beta_max * min(epoch / self.warmup_epochs, 1.0)
            for batch in data_loader:
                x = batch[0].to(self.device)
                # D-step
                with torch.no_grad():
                    mu, logvar = self.model.encoder(x)
                    z = self.model.reparameterize(mu, logvar)
                z_det = z.detach()
                z_perm = self._permute_latent(z_det)
                logits_real = self.discriminator(z_det)
                logits_fake = self.discriminator(z_perm)
                loss_d = 0.5*(F.cross_entropy(logits_real, torch.zeros(z_det.size(0),dtype=torch.long,device=self.device))
                             + F.cross_entropy(logits_fake, torch.ones(z_perm.size(0),dtype=torch.long,device=self.device)))
                self.optim_d.zero_grad(); loss_d.backward(); self.optim_d.step()
                # VAE-step
                x_recon, mu, logvar, z = self.model(x)
                recon_loss = ((x_recon - x)**2).view(x.size(0),-1).sum(dim=1).mean()
                kl_loss = 0.5*(logvar.exp() + mu**2 - 1 - logvar).sum(dim=1).mean()
                for p in self.discriminator.parameters(): p.requires_grad=False
                logits_z = self.discriminator(z)
                tc_loss = F.cross_entropy(logits_z, torch.ones(z.size(0),dtype=torch.long,device=self.device))
                for p in self.discriminator.parameters(): p.requires_grad=True
                loss_vae = recon_loss + kl_loss + beta * tc_loss
                self.optim_vae.zero_grad(); loss_vae.backward(); self.optim_vae.step()
        print("Training complete.")

    def evaluate_on_data(self, X):
        self.model.eval()
        X_np = np.array(X)
        X_t = torch.tensor(X_np, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            recon, mu, logvar, z = self.model(X_t)
            latent = z.cpu().numpy()
            latent_var = np.var(latent, axis=0)
            total_var = np.var(X_np, axis=0).sum()
            evr = latent_var / total_var
            recon_errs = ((X_t - recon)**2).mean(dim=1).cpu().numpy()
            recon_var = np.var(recon.cpu().numpy(), axis=0).sum()
            evr_total = recon_var / total_var
        return latent, recon_errs, evr, evr_total, recon.cpu().numpy()

class EncoderWrapper(nn.Module):
    def __init__(self, model):
        super(EncoderWrapper, self).__init__()
        self.model = model  # 传入整个 autoencoder 模型

    def forward(self, x):

        output = self.model.encoder(x)  # 只返回编码器输出
        return output.clone()

