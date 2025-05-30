import os
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import netron
import nni
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.onnx
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn.model_selection import KFold
import torch.nn.functional as F
# --------------------------------------------------------------------------------------------------
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
        layer1_neurons=19,
        layer2_neurons=79,
        layer3_neurons=75,
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
        )
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
            reconstructed = self.model(torch.tensor(X, dtype=torch.float32))

            # Get the output of the encoder, i.e., latent factors
            latent_factors = self.model.encoder(
                torch.tensor(X, dtype=torch.float32)
            ).numpy()

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
                (torch.tensor(X, dtype=torch.float32) - reconstructed) ** 2,
                dim=1,
            ).numpy()

            # Calculate the variance contribution of the reconstructed data
            reconstruction_variance = np.var(reconstructed.numpy(), axis=0).sum()

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

    def plot_reconstruction_errors(self, datasets):
        self.model.eval()
        with torch.no_grad():
            for name, dataset in datasets.items():
                # Calculate reconstruction errors
                reconstructed = self.model(torch.tensor(dataset, dtype=torch.float32))
                reconstruction_errors = torch.mean(
                    (torch.tensor(dataset, dtype=torch.float32) - reconstructed) ** 2,
                    dim=1,
                ).numpy()

                # Calculate quartiles
                q1 = np.percentile(reconstruction_errors, 25)
                q3 = np.percentile(reconstruction_errors, 75)
                iqr = q3 - q1

                # Define a range to identify outliers (1.5 times the IQR)
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr

                # Filter out outliers
                filtered_errors = reconstruction_errors[
                    (reconstruction_errors >= lower_bound)
                    & (reconstruction_errors <= upper_bound)
                ]

                # Plot the distribution of reconstruction errors
                sns.histplot(filtered_errors, kde=True)
                plt.xlabel("Reconstruction Error")
                plt.title(
                    f"Distribution of Reconstruction Errors ({name} dataset Without Outliers)"
                )
                plt.show()

    def export_to_onnx(self, X_train, onnx_path):
        # Use the first training sample as dummy input
        dummy_input = torch.tensor(X_train[:1], dtype=torch.float32)

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

    def tunning_train(self):
        best_val_loss = float("inf")
        patience = 20
        epochs_without_improvement = 0
        train_losses, val_losses = [], []

        for epoch in range(2000):
            self.model.train()
            train_loss = 0
            for batch_features, _ in self.train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_features)
                reconstruction_loss = self.criterion(outputs, batch_features)
                # Calculate decorrelation loss
                latent_repr = self.model.encoder(batch_features)
                decorrelation_loss_value = decorrelation_loss(latent_repr)
                
                # Combine losses
                loss = reconstruction_loss + decorrelation_loss_value
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item() * batch_features.size(0)
            train_losses.append(train_loss / len(self.train_loader.dataset))

            # Validation step
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_features, _ in self.val_loader:
                    outputs = self.model(batch_features)
                    reconstruction_loss = self.criterion(outputs, batch_features)
                    # Calculate decorrelation loss
                    latent_repr = self.model.encoder(batch_features)
                    decorrelation_loss_value = decorrelation_loss(latent_repr)
                    
                    # Combine losses
                    loss = reconstruction_loss + decorrelation_loss_value
                    
                    val_loss += loss.item() * batch_features.size(0)
            val_losses.append(val_loss / len(self.val_loader.dataset))
            self.scheduler.step(val_loss)

            if is_nni_running():
                # Report intermediate results to NNI
                nni.report_intermediate_result(
                    {
                        "train_loss": train_loss / len(self.train_loader.dataset),
                        "val_loss": val_loss / len(self.val_loader.dataset),
                    }
                )

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

    def explained_variance_ratio_total(self, X_test):
        self.explained_variance_ratio_total_value = self.evaluate_on_data(X_test)[3]
        return self.explained_variance_ratio_total_value
 
# preict the generated factors with the original factors
class LassoAnalysis:
    """
    function: preict the generated factors with the original factors

    args:
    - qns: np.ndarray, Questionnaire data
    - scores: np.ndarray, Factor scores
    - alpha_values: list, List of alpha values for Lasso regularization

    return:
    - pictures: plot the R^2 values for each factor
    """

    def __init__(self, qns, scores, alpha_values=None):
        self.qns = qns
        self.scores = scores
        self.alpha_values = (
            alpha_values
            if alpha_values is not None
            else [
                0.001,
                0.01,
                0.05,
                0.075,
                0.1,
                0.125,
                0.15,
                0.2,
                0.3,
                0.4,
                0.5,
                0.6,
                0.7,
            ]
        )
        self.r2_values = np.empty((5, len(self.alpha_values)))
        self.n_items = np.empty(len(self.alpha_values))
        self.pal = ["#4f4f4f", "#B80044", "#0e79b2", "#f9a800", "#00a087"]
        self.prop = matplotlib.font_manager.FontProperties(
            fname="c:\\windows\\fonts\\nunitosans-light.ttf"
        )
        matplotlib.rcParams["font.weight"] = "light"
        matplotlib.rcParams["axes.facecolor"] = "#fbfbfb"

    def perform_analysis(self):
        for n, alpha in enumerate(tqdm(self.alpha_values)):
            clf = Lasso(alpha=alpha)
            clf.fit(self.qns, self.scores)
            pred = cross_val_predict(clf, self.qns, self.scores, cv=5)
            for i in range(5):
                self.r2_values[i, n] = r2_score(self.scores.iloc[:, i], pred[:, i])
            self.n_items[n] = np.any(clf.coef_.T != 0, axis=1).sum()

    def plot_r2_values(self):
        f, ax = plt.subplots(dpi=100, facecolor="white")
        for i in range(5):
            ax.plot(
                self.n_items,
                self.r2_values[i, :],
                label="Factor {0}".format(i + 1),
                color=self.pal[i],
            )
        ax.set_xlabel("Number of items")
        ax.set_ylabel("$R^2$")
        ax.legend()
        ax2 = ax.twiny()
        ax2.set_xticklabels(self.alpha_values)
        ax2.set_xticks(self.n_items)
        ax.axvline(63, color="#8c8a8a", linestyle=":")
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=90)
        plt.tight_layout()
        plt.show()

    def plot_factor_predictions(self, alpha=0.125):
        clf = Lasso(alpha=alpha)
        pred = cross_val_predict(clf, self.qns, self.scores, cv=5)
        clf.fit(self.qns, self.scores)
        f, ax = plt.subplots(1, 5, figsize=(16, 3.5), dpi=100, facecolor="white")
        factors = ["V1", "V2", "V3", "V4", "V5"]
        for i in range(5):
            sns.regplot(
                x=self.scores.iloc[:, i],
                y=pred[:, i],
                ax=ax[i],
                color=self.pal[i],
                scatter_kws={"alpha": 0.5},
            )
            ax[i].set_title(
                factors[i]
                + "\n$R^2$ = {0}".format(
                    np.round(r2_score(self.scores.iloc[:, i], pred[:, i]), 5)
                ),
                fontweight="light",
            )
            ax[i].set_xlabel("True score")
            ax[i].set_ylabel("Predicted score")
        plt.tight_layout()
        plt.show()

    def plot_heatmap(self, alpha=0.125):
        clf = Lasso(alpha=alpha)
        clf.fit(self.qns, self.scores)
        plt.figure(dpi=100, figsize=(9, 1.5), facecolor="white")
        sns.heatmap(clf.coef_, cmap="Blues", yticklabels=["V1", "V2", "V3", "V4", "V5"])
        plt.xlabel("Question number")
        plt.ylabel("Factor")
        plt.tight_layout()
        plt.show()


# --------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    # load data
    code_dir = Path(os.getcwd())
    data_path = code_dir / "data"
    assert os.path.exists(data_path), "Cannot find data folder, please run the code in root directory of the project."
    
    with open(data_path / "cbcl_data_remove_unrelated.csv", "r", encoding="utf-8") as f:
        qns = pd.read_csv(f)
    
    X = qns.iloc[:, 2:].values

    # outer cv
    outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
    outer_scores = []

    # passing hyperparameters
    if is_nni_running():
        params = nni.get_next_parameter()
        layer1_neurons = params.get("layer1_neurons")
        layer2_neurons = params.get("layer2_neurons")
        layer3_neurons = params.get("layer3_neurons")
    else:
        layer1_neurons, layer2_neurons, layer3_neurons = 69, 58, 53
    
    for outer_fold, (outer_train_idx, outer_test_idx) in enumerate(outer_cv.split(X), start=1):
        print(f"====== Outer Fold {outer_fold} ======")
        X_outer_train_raw, X_outer_test_raw = X[outer_train_idx], X[outer_test_idx]
        
        scaler_fold = MinMaxScaler()
        X_outer_train = scaler_fold.fit_transform(X_outer_train_raw)
        X_outer_test = scaler_fold.transform(X_outer_test_raw)
        
        # inner cv
        inner_cv = KFold(n_splits=10, shuffle=True, random_state=42)
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
            )
            
            # train in inner fold and evaluate in inner fold
            autoencoder_inner.tunning_train()
            inner_score = autoencoder_inner.explained_variance_ratio_total(X_inner_val)
            print(f"Inner fold explained variance: {inner_score}")
            inner_scores.append(inner_score)
        
        avg_inner_score = np.mean(inner_scores)
        print(f"Average inner CV score for outer fold {outer_fold}: {avg_inner_score}")
        
        # Train on the entire outer training set and evaluate on the outer test set
        autoencoder_outer = Autoencoder(
            X_outer_train,
            X_outer_test,
            encoding_dim=5,
            layer1_neurons=layer1_neurons,
            layer2_neurons=layer2_neurons,
            layer3_neurons=layer3_neurons,
        )
        autoencoder_outer.tunning_train()
        outer_score = autoencoder_outer.explained_variance_ratio_total(X_outer_test)
        print(f"Outer fold {outer_fold} test explained variance: {outer_score}")
        outer_scores.append(outer_score)
    
    final_avg_score = np.mean(outer_scores)
    print("Final average test explained variance:", final_avg_score)
    
    # report final result
    nni.report_final_result(final_avg_score)
