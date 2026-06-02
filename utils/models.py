"""Autoencoder training and nested cross-validation for CBCL questionnaire data."""

# ==================================================================================================
# Imports
# ==================================================================================================

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset

try:
    import netron
except ImportError:  # Optional dependency used only for ONNX model visualization.
    netron = None

try:
    import nni
except ImportError:  # Optional dependency used only during NNI hyperparameter search.
    nni = None


# ==================================================================================================
# Configuration and paths
# ==================================================================================================

RANDOM_STATE = 42
DATA_DIR_NAME = "data"
DATA_FILE_NAME = "cbcl_data_remove_unrelated.csv"
FEATURE_START_COLUMN = 2

ENCODING_DIM = 5
BATCH_SIZE = 32
MAX_EPOCHS = 2000
EARLY_STOPPING_PATIENCE = 20
SCHEDULER_PATIENCE = 5
LEARNING_RATE = 1e-3
DEFAULT_LAYER_NEURONS = (69, 58, 53)

OUTER_CV_SPLITS = 5
INNER_CV_SPLITS = 10


# ==================================================================================================
# Utility functions
# ==================================================================================================


def is_nni_running() -> bool:
    """Return whether the script is running inside an active NNI experiment."""
    return nni is not None and "NNI_PLATFORM" in os.environ


def get_project_paths() -> Tuple[Path, Path, Path]:
    """
    Resolve the working directory, data directory, and questionnaire CSV file.

    Returns
    -------
    tuple[pathlib.Path, pathlib.Path, pathlib.Path]
        Current working directory, data directory, and input CSV path.

    Raises
    ------
    FileNotFoundError
        If the data directory or input CSV file is not available.
    """
    code_dir = Path(os.getcwd())
    data_path = code_dir / DATA_DIR_NAME
    data_file = data_path / DATA_FILE_NAME

    if not data_path.exists():
        raise FileNotFoundError(
            f"Cannot find data folder at {data_path}. Run the code from the project root directory."
        )

    if not data_file.exists():
        raise FileNotFoundError(f"Cannot find input data file: {data_file}")

    return code_dir, data_path, data_file


def get_autoencoder_hyperparameters() -> Tuple[int, int, int]:
    """
    Get encoder and decoder hidden-layer sizes from NNI or default settings.

    Returns
    -------
    tuple[int, int, int]
        Neuron counts for the three hidden layers.
    """
    if not is_nni_running():
        return DEFAULT_LAYER_NEURONS

    params = nni.get_next_parameter()
    layer1_neurons = params.get("layer1_neurons", DEFAULT_LAYER_NEURONS[0])
    layer2_neurons = params.get("layer2_neurons", DEFAULT_LAYER_NEURONS[1])
    layer3_neurons = params.get("layer3_neurons", DEFAULT_LAYER_NEURONS[2])

    return int(layer1_neurons), int(layer2_neurons), int(layer3_neurons)


def report_nni_final_result(value: float) -> None:
    """
    Report the final score to NNI when an NNI experiment is active.

    Parameters
    ----------
    value : float
        Final average test explained variance from the nested cross-validation workflow.
    """
    if is_nni_running():
        nni.report_final_result(value)


# ==================================================================================================
# Data loading
# ==================================================================================================


def load_questionnaire_features(data_file: Path) -> np.ndarray:
    """
    Load CBCL questionnaire features from the configured CSV file.

    Parameters
    ----------
    data_file : pathlib.Path
        Path to the CBCL questionnaire CSV file.

    Returns
    -------
    numpy.ndarray
        Feature matrix beginning from the configured questionnaire-feature column.
    """
    qns = pd.read_csv(data_file, encoding="utf-8")
    return qns.iloc[:, FEATURE_START_COLUMN:].values


# ==================================================================================================
# Model / analysis functions
# ==================================================================================================


class QuestionnaireDataset(Dataset):
    """
    Dataset wrapper for questionnaire autoencoder training.

    Parameters
    ----------
    data : array-like
        Questionnaire feature matrix.

    Notes
    -----
    Each sample returns the same tensor as both input and target for reconstruction learning.
    """

    def __init__(self, data: np.ndarray) -> None:
        self.data = torch.tensor(data, dtype=torch.float32)

    def __len__(self) -> int:
        """Return the number of questionnaire records."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return one reconstruction-learning sample."""
        return self.data[idx], self.data[idx]


class AutoencoderModel(nn.Module):
    """
    Fully connected autoencoder for low-dimensional questionnaire representation learning.

    Parameters
    ----------
    input_dim : int
        Number of questionnaire features.
    latent_dim : int
        Number of latent dimensions.
    layer1_neurons : int
        Number of neurons in the first hidden layer.
    layer2_neurons : int
        Number of neurons in the second hidden layer.
    layer3_neurons : int
        Number of neurons in the third hidden layer.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        layer1_neurons: int,
        layer2_neurons: int,
        layer3_neurons: int,
    ) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, layer1_neurons),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(layer1_neurons, layer2_neurons),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(layer2_neurons, layer3_neurons),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(layer3_neurons, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, layer3_neurons),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(layer3_neurons, layer2_neurons),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(layer2_neurons, layer1_neurons),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(layer1_neurons, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct input questionnaire features through the latent representation.

        Parameters
        ----------
        x : torch.Tensor
            Input questionnaire feature tensor.

        Returns
        -------
        torch.Tensor
            Reconstructed feature tensor.
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def decorrelation_loss(latent_repr: torch.Tensor) -> torch.Tensor:
    """
    Compute a covariance-based decorrelation penalty for latent dimensions.

    Parameters
    ----------
    latent_repr : torch.Tensor
        Latent representation with shape ``(batch_size, latent_dim)``.

    Returns
    -------
    torch.Tensor
        Sum of squared off-diagonal covariance terms.
    """
    batch_size, latent_dim = latent_repr.shape
    centered_latent = latent_repr - latent_repr.mean(dim=0, keepdim=True)
    cov_matrix = (centered_latent.T @ centered_latent) / batch_size
    diagonal_mask = torch.eye(latent_dim, device=latent_repr.device)
    return torch.sum((cov_matrix * (1 - diagonal_mask)) ** 2)


class Autoencoder:
    """
    Training, evaluation, plotting, and export interface for the questionnaire autoencoder.

    Parameters
    ----------
    X_train : numpy.ndarray
        Training feature matrix.
    X_val : numpy.ndarray
        Validation feature matrix.
    encoding_dim : int
        Number of latent dimensions.
    layer1_neurons : int, default=19
        Number of neurons in the first hidden layer.
    layer2_neurons : int, default=79
        Number of neurons in the second hidden layer.
    layer3_neurons : int, default=75
        Number of neurons in the third hidden layer.
    """

    def __init__(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        encoding_dim: int,
        layer1_neurons: int = 19,
        layer2_neurons: int = 79,
        layer3_neurons: int = 75,
    ) -> None:
        train_dataset = QuestionnaireDataset(X_train)
        val_dataset = QuestionnaireDataset(X_val)

        self.train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

        input_dim = X_train.shape[1]
        self.model = AutoencoderModel(
            input_dim=input_dim,
            latent_dim=encoding_dim,
            layer1_neurons=layer1_neurons,
            layer2_neurons=layer2_neurons,
            layer3_neurons=layer3_neurons,
        )
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.1,
            patience=SCHEDULER_PATIENCE,
        )
        self.explained_variance_ratio_total_value: Optional[float] = None

    def get_model(self) -> AutoencoderModel:
        """
        Return the PyTorch autoencoder model.

        Returns
        -------
        AutoencoderModel
            Current autoencoder model instance.
        """
        return self.model

    def _compute_batch_loss(self, batch_features: torch.Tensor) -> torch.Tensor:
        """
        Compute reconstruction loss plus latent decorrelation penalty for one batch.

        Parameters
        ----------
        batch_features : torch.Tensor
            Batch of questionnaire features.

        Returns
        -------
        torch.Tensor
            Combined training objective for the batch.
        """
        outputs = self.model(batch_features)
        reconstruction_loss = self.criterion(outputs, batch_features)
        latent_repr = self.model.encoder(batch_features)
        decorrelation_loss_value = decorrelation_loss(latent_repr)
        return reconstruction_loss + decorrelation_loss_value

    def _fit(self, report_to_nni: bool = False, show_plot: bool = False) -> None:
        """
        Train the autoencoder with validation monitoring and early stopping.

        Parameters
        ----------
        report_to_nni : bool, default=False
            Whether to report intermediate train and validation losses to NNI.
        show_plot : bool, default=False
            Whether to display train and validation loss curves after training.
        """
        best_val_loss = float("inf")
        epochs_without_improvement = 0
        train_losses = []
        val_losses = []

        for epoch in range(MAX_EPOCHS):
            self.model.train()
            train_loss = 0.0

            for batch_features, _ in self.train_loader:
                self.optimizer.zero_grad()
                loss = self._compute_batch_loss(batch_features)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item() * batch_features.size(0)

            train_loss_avg = train_loss / len(self.train_loader.dataset)
            train_losses.append(train_loss_avg)

            # Validation loss uses the same reconstruction-plus-decorrelation objective.
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_features, _ in self.val_loader:
                    loss = self._compute_batch_loss(batch_features)
                    val_loss += loss.item() * batch_features.size(0)

            val_loss_avg = val_loss / len(self.val_loader.dataset)
            val_losses.append(val_loss_avg)
            self.scheduler.step(val_loss)

            if report_to_nni and is_nni_running():
                nni.report_intermediate_result(
                    {"train_loss": train_loss_avg, "val_loss": val_loss_avg}
                )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        if show_plot:
            self._plot_loss_curves(train_losses, val_losses)

    @staticmethod
    def _plot_loss_curves(train_losses: Iterable[float], val_losses: Iterable[float]) -> None:
        """
        Display training and validation loss curves.

        Parameters
        ----------
        train_losses : iterable of float
            Average training loss per epoch.
        val_losses : iterable of float
            Average validation loss per epoch.
        """
        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Validation Loss")
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss Curves")
        plt.show()

    def train(self, show_plot: bool = False) -> None:
        """
        Train the autoencoder and optionally display loss curves.

        Parameters
        ----------
        show_plot : bool, default=False
            Whether to display training and validation loss curves.
        """
        self._fit(report_to_nni=False, show_plot=show_plot)

    def tunning_train(self) -> None:
        """
        Train the autoencoder and report intermediate losses during NNI experiments.

        Notes
        -----
        The method name is retained for compatibility with existing scripts that call it directly.
        """
        self._fit(report_to_nni=True, show_plot=False)

    def tuning_train(self) -> None:
        """
        Train the autoencoder and report intermediate losses during NNI experiments.

        This method provides a standard spelling while keeping the existing training behavior.
        """
        self.tunning_train()

    def evaluate_on_data(
        self,
        X: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Evaluate latent factors, reconstruction errors, and variance metrics on a dataset.

        Parameters
        ----------
        X : numpy.ndarray
            Feature matrix to evaluate.

        Returns
        -------
        tuple
            Latent factors, per-sample reconstruction errors, latent-factor variance ratios,
            and total reconstruction explained variance ratio.
        """
        self.model.eval()
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        X_tensor = torch.tensor(X, dtype=torch.float32)

        with torch.no_grad():
            reconstructed = self.model(X_tensor)
            latent_factors = self.model.encoder(X_tensor).numpy()

            latent_variances = np.var(latent_factors, axis=0)
            total_variance = np.var(X, axis=0).sum()
            explained_variance_ratios = latent_variances / total_variance

            reconstruction_errors = torch.mean((X_tensor - reconstructed) ** 2, dim=1).numpy()
            reconstruction_variance = np.var(reconstructed.numpy(), axis=0).sum()
            explained_variance_ratio_total = reconstruction_variance / total_variance

        return (
            latent_factors,
            reconstruction_errors,
            explained_variance_ratios,
            explained_variance_ratio_total,
        )

    def explained_variance_ratio_total(self, X_test: np.ndarray) -> float:
        """
        Return total reconstruction explained variance ratio on the test matrix.

        Parameters
        ----------
        X_test : numpy.ndarray
            Test feature matrix.

        Returns
        -------
        float
            Total reconstruction explained variance ratio.
        """
        self.explained_variance_ratio_total_value = self.evaluate_on_data(X_test)[3]
        return self.explained_variance_ratio_total_value

    def plot_reconstruction_errors(self, datasets: Dict[str, np.ndarray]) -> None:
        """
        Plot reconstruction-error distributions after removing IQR-defined outliers.

        Parameters
        ----------
        datasets : dict[str, numpy.ndarray]
            Mapping from dataset names to feature matrices.
        """
        self.model.eval()
        with torch.no_grad():
            for name, dataset in datasets.items():
                dataset_tensor = torch.tensor(dataset, dtype=torch.float32)
                reconstructed = self.model(dataset_tensor)
                reconstruction_errors = torch.mean(
                    (dataset_tensor - reconstructed) ** 2,
                    dim=1,
                ).numpy()

                q1 = np.percentile(reconstruction_errors, 25)
                q3 = np.percentile(reconstruction_errors, 75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr

                filtered_errors = reconstruction_errors[
                    (reconstruction_errors >= lower_bound)
                    & (reconstruction_errors <= upper_bound)
                ]

                sns.histplot(filtered_errors, kde=True)
                plt.xlabel("Reconstruction Error")
                plt.title(
                    f"Distribution of Reconstruction Errors ({name} dataset Without Outliers)"
                )
                plt.show()

    def export_to_onnx(self, X_train: np.ndarray, onnx_path: Path | str) -> None:
        """
        Export the trained autoencoder to ONNX and open a Netron visualization when available.

        Parameters
        ----------
        X_train : numpy.ndarray
            Training feature matrix used to create a dummy input with the correct feature shape.
        onnx_path : pathlib.Path or str
            Destination path for the ONNX model file.

        Raises
        ------
        ImportError
            If Netron visualization is requested but Netron is not installed.
        """
        onnx_path = Path(onnx_path)
        onnx_path.parent.mkdir(parents=True, exist_ok=True)
        dummy_input = torch.tensor(X_train[:1], dtype=torch.float32)

        torch.onnx.export(
            self.model,
            dummy_input,
            onnx_path,
            input_names=["input"],
            output_names=["reconstructed"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "reconstructed": {0: "batch_size"},
            },
            opset_version=11,
        )

        print(f"Model exported to {onnx_path}")

        if netron is None:
            raise ImportError("Netron is required to visualize the exported ONNX model.")
        netron.start(str(onnx_path))


# ==================================================================================================
# Evaluation functions
# ==================================================================================================


def run_nested_cross_validation(
    X: np.ndarray,
    layer1_neurons: int,
    layer2_neurons: int,
    layer3_neurons: int,
) -> Tuple[float, list[float]]:
    """
    Run nested cross-validation for autoencoder reconstruction explained variance.

    Parameters
    ----------
    X : numpy.ndarray
        Raw questionnaire feature matrix.
    layer1_neurons : int
        Number of neurons in the first hidden layer.
    layer2_neurons : int
        Number of neurons in the second hidden layer.
    layer3_neurons : int
        Number of neurons in the third hidden layer.

    Returns
    -------
    tuple[float, list[float]]
        Final average outer-test explained variance and the per-fold outer-test scores.
    """
    outer_cv = KFold(n_splits=OUTER_CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    outer_scores = []

    for outer_fold, (outer_train_idx, outer_test_idx) in enumerate(outer_cv.split(X), start=1):
        print(f"====== Outer Fold {outer_fold} ======")
        X_outer_train_raw = X[outer_train_idx]
        X_outer_test_raw = X[outer_test_idx]

        # Scaling is fitted within each outer training fold before inner validation and test scoring.
        scaler_fold = MinMaxScaler()
        X_outer_train = scaler_fold.fit_transform(X_outer_train_raw)
        X_outer_test = scaler_fold.transform(X_outer_test_raw)

        inner_score = run_inner_cross_validation(
            X_outer_train=X_outer_train,
            layer1_neurons=layer1_neurons,
            layer2_neurons=layer2_neurons,
            layer3_neurons=layer3_neurons,
        )
        print(f"Average inner CV score for outer fold {outer_fold}: {inner_score}")

        autoencoder_outer = Autoencoder(
            X_outer_train,
            X_outer_test,
            encoding_dim=ENCODING_DIM,
            layer1_neurons=layer1_neurons,
            layer2_neurons=layer2_neurons,
            layer3_neurons=layer3_neurons,
        )
        autoencoder_outer.tunning_train()
        outer_score = autoencoder_outer.explained_variance_ratio_total(X_outer_test)
        print(f"Outer fold {outer_fold} test explained variance: {outer_score}")
        outer_scores.append(outer_score)

    final_avg_score = float(np.mean(outer_scores))
    return final_avg_score, outer_scores


def run_inner_cross_validation(
    X_outer_train: np.ndarray,
    layer1_neurons: int,
    layer2_neurons: int,
    layer3_neurons: int,
) -> float:
    """
    Run inner cross-validation within one outer training split.

    Parameters
    ----------
    X_outer_train : numpy.ndarray
        MinMax-scaled outer-training feature matrix.
    layer1_neurons : int
        Number of neurons in the first hidden layer.
    layer2_neurons : int
        Number of neurons in the second hidden layer.
    layer3_neurons : int
        Number of neurons in the third hidden layer.

    Returns
    -------
    float
        Average inner-validation reconstruction explained variance.
    """
    inner_cv = KFold(n_splits=INNER_CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    inner_scores = []

    for inner_fold, (inner_train_idx, inner_val_idx) in enumerate(
        inner_cv.split(X_outer_train),
        start=1,
    ):
        print(f"  -- Inner Fold {inner_fold}")
        X_inner_train = X_outer_train[inner_train_idx]
        X_inner_val = X_outer_train[inner_val_idx]

        autoencoder_inner = Autoencoder(
            X_inner_train,
            X_inner_val,
            encoding_dim=ENCODING_DIM,
            layer1_neurons=layer1_neurons,
            layer2_neurons=layer2_neurons,
            layer3_neurons=layer3_neurons,
        )

        autoencoder_inner.tunning_train()
        inner_score = autoencoder_inner.explained_variance_ratio_total(X_inner_val)
        print(f"Inner fold explained variance: {inner_score}")
        inner_scores.append(inner_score)

    return float(np.mean(inner_scores))


# ==================================================================================================
# Main workflow
# ==================================================================================================


def main() -> None:
    """Run the full autoencoder nested cross-validation workflow."""
    _, _, data_file = get_project_paths()
    X = load_questionnaire_features(data_file)

    layer1_neurons, layer2_neurons, layer3_neurons = get_autoencoder_hyperparameters()
    print(
        "Using hidden-layer sizes: "
        f"{layer1_neurons}, {layer2_neurons}, {layer3_neurons}"
    )

    final_avg_score, _ = run_nested_cross_validation(
        X=X,
        layer1_neurons=layer1_neurons,
        layer2_neurons=layer2_neurons,
        layer3_neurons=layer3_neurons,
    )

    print("Final average test explained variance:", final_avg_score)
    report_nni_final_result(final_avg_score)


if __name__ == "__main__":
    main()
