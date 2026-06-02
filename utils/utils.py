"""Utility functions for CBCL item lookup, data search, model interpretation, and reconstruction checks."""

from __future__ import annotations

# =============================================================================
# Imports
# =============================================================================

import argparse
import os
import re
import subprocess
import time
import textwrap
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from bs4 import BeautifulSoup
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


# =============================================================================
# Configuration and paths
# =============================================================================

ELEMENT_HTML_PATH = Path("../data/element.html")
CBCL_PATTERN = re.compile(r"(cbcl_q\d+[a-z]*_p)")


# =============================================================================
# Utility functions
# =============================================================================


def _require_file(path: Path, description: str) -> None:
    """Validate that a required file exists before it is read.

    Parameters
    ----------
    path : Path
        File path to validate.
    description : str
        Human-readable file description used in the error message.

    Raises
    ------
    FileNotFoundError
        If the requested file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{description} not found at {path}. Run this script from the project root."
        )


def _load_element_soup(element_html_path: Path = ELEMENT_HTML_PATH) -> BeautifulSoup:
    """Load the ABCD element HTML dictionary as a BeautifulSoup object.

    Parameters
    ----------
    element_html_path : Path, default=ELEMENT_HTML_PATH
        Path to the `element.html` data dictionary file.

    Returns
    -------
    BeautifulSoup
        Parsed HTML content for CBCL item lookup.
    """
    _require_file(element_html_path, "element.html")
    with element_html_path.open("r", encoding="utf-8") as file:
        return BeautifulSoup(file, "html.parser")


def _get_cbcl_detail_from_soup(soup: BeautifulSoup, cbcl_item: str) -> str:
    """Return the text description associated with one CBCL item name.

    Parameters
    ----------
    soup : BeautifulSoup
        Parsed element dictionary.
    cbcl_item : str
        CBCL item variable name, such as `cbcl_q08_p`.

    Returns
    -------
    str
        Item description, or `N/A` when no matching entry is found.
    """
    target = soup.find(
        lambda tag: tag.name == "td" and cbcl_item in tag.get_text(strip=True)
    )
    if target:
        return target.find_next("td").get_text(strip=True)
    return "N/A"


def _to_numpy_2d(values, name: str) -> np.ndarray:
    """Convert array-like model inputs to a two-dimensional NumPy array.

    Parameters
    ----------
    values : np.ndarray, pd.DataFrame, or torch.Tensor
        Input data to convert.
    name : str
        Variable name used in the validation message.

    Returns
    -------
    np.ndarray
        Two-dimensional NumPy array.

    Raises
    ------
    ValueError
        If the converted object is not two-dimensional.
    """
    if isinstance(values, torch.Tensor):
        array = values.detach().cpu().numpy()
    elif isinstance(values, pd.DataFrame):
        array = values.values
    else:
        array = np.asarray(values)

    if array.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {array.shape}.")
    return array


def wrap_labels(labels: Iterable[str], width: int = 20) -> list[str]:
    """Wrap long plot labels to improve figure readability.

    Parameters
    ----------
    labels : iterable of str
        Labels to wrap.
    width : int, default=20
        Maximum line width passed to `textwrap.fill`.

    Returns
    -------
    list of str
        Wrapped labels.
    """
    return [textwrap.fill(str(label), width) for label in labels]


# =============================================================================
# CBCL item lookup and translation
# =============================================================================


def get_cbcl_details(cbcl_item: str) -> str:
    """Look up CBCL item descriptions from the ABCD element dictionary.

    Parameters
    ----------
    cbcl_item : str
        Single CBCL item name or a combined item string containing multiple CBCL
        variable names, for example `avg_cbcl_q08_p_cbcl_q10_p_cbcl_q78_p`.

    Returns
    -------
    str
        Semicolon-separated item descriptions. Returns `N/A` when no CBCL item
        name is detected or no matching description is available.
    """
    soup = _load_element_soup()
    cbcl_items = CBCL_PATTERN.findall(cbcl_item)
    details = [_get_cbcl_detail_from_soup(soup, cbcl) for cbcl in cbcl_items]
    return "; ".join(details) if details else "N/A"


def translate_text(df: pd.DataFrame, language: str) -> pd.DataFrame:
    """Map CBCL variable names to item descriptions and translated descriptions.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame whose columns contain CBCL variable names or combined CBCL item
        strings. Each column is processed as one factor/item list.
    language : str
        Target language code accepted by `deep_translator.GoogleTranslator`.

    Returns
    -------
    pd.DataFrame
        DataFrame containing one description column and one translated-description
        column for each input column.

    Raises
    ------
    ImportError
        If `deep_translator` is unavailable in the current environment.
    """
    if GoogleTranslator is None:
        raise ImportError(
            "deep_translator is required for translate_text. Install it before using this function."
        )

    soup = _load_element_soup()
    result_df = pd.DataFrame()

    for factor_idx in range(len(df.columns)):
        original_text = []
        translated_text = []

        for column_name in df.iloc[:, factor_idx]:
            cbcl_items = CBCL_PATTERN.findall(str(column_name))
            original_details = []
            translated_details = []

            for cbcl_item in cbcl_items:
                detail_info = _get_cbcl_detail_from_soup(soup, cbcl_item)
                if detail_info == "N/A":
                    continue

                original_details.append(detail_info)
                try:
                    translated_detail = GoogleTranslator(
                        source="es", target=language
                    ).translate(detail_info)
                except AttributeError as exc:
                    print(f"Translation failed for {cbcl_item}: {exc}")
                    translated_detail = detail_info

                translated_details.append(translated_detail)
                time.sleep(0.25)

            original_text.append("; ".join(original_details) if original_details else "N/A")
            translated_text.append(
                "; ".join(translated_details) if translated_details else "N/A"
            )

        factor_df = pd.DataFrame(
            {
                f"Factor {factor_idx} Detail": original_text,
                f"Factor {factor_idx} Translated_Detail": translated_text,
            }
        )
        result_df = pd.concat(
            [result_df.reset_index(drop=True), factor_df.reset_index(drop=True)],
            axis=1,
        )

    return result_df


# =============================================================================
# NDA fMRI data access
# =============================================================================


class GetfMRIdata:
    """Fetch baseline rs-fMRI and T1 S3 endpoints from an NDA Oracle package.

    Parameters
    ----------
    package_id : str or int
        NDA package identifier appended to the user name.
    password : str
        NDA package password used for the Oracle connection.

    Attributes
    ----------
    conn : cx_Oracle.Connection
        Active Oracle database connection.
    s3_samples : list of str
        S3 endpoint links returned by the query.
    """

    def __init__(self, package_id, password):
        if cx_Oracle is None:
            raise ImportError(
                "cx_Oracle is required for GetfMRIdata. Install it and configure the Oracle client before using this class."
            )

        user = f"k21116947_{package_id}"
        dsn = cx_Oracle.makedsn(
            "mindarvpc.cqahbwk3l1mb.us-east-1.rds.amazonaws.com",
            1521,
            service_name="ORCL",
        )
        self.conn = cx_Oracle.connect(user=user, password=password, dsn=dsn)
        self.s3_samples = []

    def fetch_data(self) -> None:
        """Query NDA S3 endpoint links for baseline rs-fMRI and T1 MPROC files."""
        cursor = self.conn.cursor()
        query = """
        SELECT ENDPOINT
        FROM S3_LINKS
        WHERE ENDPOINT LIKE '%baseline%'
          AND (ENDPOINT LIKE '%rsfMRI%' OR ENDPOINT LIKE '%T1%')
          AND ENDPOINT LIKE '%MPROC%'
        """
        try:
            cursor.execute(query)
            self.s3_samples = [row[0] for row in cursor.fetchall()]
        finally:
            cursor.close()

    def save_data(
        self,
        link_file: str | Path = "data/s3_links.txt",
        download_dir: str | Path = "./data/fMRI_data",
    ) -> None:
        """Save S3 endpoint links and run the NDA download command.

        Parameters
        ----------
        link_file : str or Path, default="data/s3_links.txt"
            Text file used by `downloadcmd` as the endpoint list.
        download_dir : str or Path, default="./data/fMRI_data"
            Directory where fMRI files are downloaded.
        """
        if not self.s3_samples:
            self.fetch_data()

        link_file = Path(link_file)
        download_dir = Path(download_dir)
        link_file.parent.mkdir(parents=True, exist_ok=True)
        download_dir.mkdir(parents=True, exist_ok=True)

        np.savetxt(link_file, self.s3_samples, fmt="%s")
        print(f"Saved {len(self.s3_samples)} S3 links to {link_file}.")

        try:
            subprocess.run(
                [
                    "downloadcmd",
                    "-dp",
                    "1236370",
                    "-t",
                    str(link_file),
                    "-d",
                    str(download_dir),
                ],
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            print(f"downloadcmd failed: {exc}")

    def close_connection(self) -> None:
        """Close the Oracle database connection."""
        self.conn.close()

    def run_all(self) -> None:
        """Run endpoint querying, link saving, file downloading, and connection closing."""
        try:
            self.fetch_data()
            self.save_data()
        finally:
            self.close_connection()


# =============================================================================
# Data search functions
# =============================================================================


def find_column_in_csvs(
    root_folder: str | Path,
    target_column: str,
    case_insensitive: bool = True,
    verbose: bool = True,
) -> list[tuple[str, str]]:
    """Search CSV files under a directory for columns matching a target string.

    Parameters
    ----------
    root_folder : str or Path
        Root directory containing CSV files.
    target_column : str
        Column name or substring to search for.
    case_insensitive : bool, default=True
        Whether to ignore case during matching.
    verbose : bool, default=True
        Whether to print matched files and read errors.

    Returns
    -------
    list of tuple[str, str]
        Tuples containing the CSV path and the matched column name.
    """
    found_files = []
    root_folder = Path(root_folder)

    for dirpath, _, filenames in os.walk(root_folder):
        for filename in filenames:
            if not filename.lower().endswith(".csv"):
                continue

            file_path = Path(dirpath) / filename
            try:
                csv_header = pd.read_csv(file_path, nrows=1)
            except Exception as exc:
                if verbose:
                    print(f"Failed to read file {file_path}: {exc}")
                continue

            for column in csv_header.columns:
                column_check = column.lower() if case_insensitive else column
                target_check = target_column.lower() if case_insensitive else target_column
                if target_check in column_check:
                    found_files.append((str(file_path), column))

    if verbose:
        if found_files:
            for path, column in found_files:
                print(f'Found column "{column}" in file: {path}')
        else:
            print(f'Column "{target_column}" was not found in any CSV file.')

    return found_files


# =============================================================================
# Model and analysis functions
# =============================================================================


def compute_autoencoder_loadings_with_plot(
    latent_factors,
    X_train,
    items,
    top_k: int = 8,
) -> pd.DataFrame:
    """Estimate feature loadings from autoencoder latent factors using linear models.

    Each observed feature is regressed on the standardized latent representation.
    The regression coefficients are used as a loading-style summary and plotted for
    the strongest features within each latent dimension.

    Parameters
    ----------
    latent_factors : np.ndarray or pd.DataFrame
        Autoencoder latent representation with shape `(n_samples, n_latent_factors)`.
    X_train : np.ndarray or pd.DataFrame
        Input feature matrix with shape `(n_samples, n_original_features)`.
    items : list-like
        Feature names corresponding to the columns of `X_train`.
    top_k : int, default=8
        Number of features with the largest absolute coefficient to show per
        latent dimension.

    Returns
    -------
    pd.DataFrame
        Loading-style coefficient matrix indexed by feature name, with latent
        dimensions as columns.
    """
    latent_array = _to_numpy_2d(latent_factors, "latent_factors")
    feature_array = _to_numpy_2d(X_train, "X_train")

    if latent_array.shape[0] != feature_array.shape[0]:
        raise ValueError(
            "latent_factors and X_train must have the same number of rows."
        )
    if len(items) != feature_array.shape[1]:
        raise ValueError("items length must match the number of columns in X_train.")

    n_original_features = feature_array.shape[1]
    n_latent_factors = latent_array.shape[1]

    # Standardize latent dimensions before estimating feature-wise coefficients.
    latent_factors_scaled = StandardScaler().fit_transform(latent_array)

    loadings = []
    for feature_idx in range(n_original_features):
        y = feature_array[:, feature_idx]
        reg = LinearRegression().fit(latent_factors_scaled, y)
        loadings.append(reg.coef_)

    loadings_df = pd.DataFrame(
        loadings,
        index=items,
        columns=[f"Latent_{idx + 1}" for idx in range(n_latent_factors)],
    )

    # Visualize the highest-magnitude feature coefficients for each latent dimension.
    sns.set(style="whitegrid")
    for column in loadings_df.columns:
        top_items = loadings_df[column].abs().sort_values(ascending=False).head(top_k).index
        top_data = loadings_df.loc[top_items, [column]]
        top_items_wrapped = wrap_labels(top_items, width=100)

        plt.figure(figsize=(6, 0.5 * len(top_items_wrapped)))
        sns.heatmap(
            top_data,
            annot=True,
            cmap="coolwarm",
            center=0,
            cbar=True,
            yticklabels=top_items_wrapped,
        )
        plt.title(f"Top {top_k} Loadings for {column}")
        plt.xlabel("Latent Dimension")
        plt.ylabel("Original Feature")
        plt.tight_layout()
        plt.show()

    return loadings_df


class DecoderHead(nn.Module):
    """Single-output wrapper around an autoencoder decoder.

    Parameters
    ----------
    decoder : nn.Module
        Decoder module that maps latent vectors to reconstructed item values.
    out_idx : int, default=0
        Output item index to expose during the forward pass.
    """

    def __init__(self, decoder: nn.Module, out_idx: int = 0):
        super().__init__()
        self.decoder = decoder
        self.out_idx = out_idx

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Return one decoder output column for SHAP attribution."""
        xhat = self.decoder(z)
        return xhat[:, self.out_idx : self.out_idx + 1]


def compute_shap_loadings_decoder_only(
    decoder: nn.Module,
    Z,
    items,
    device: str = "cuda",
    background_size: int = 32,
    sample_size: int = 200,
    nsamples: int = 16,
    eval_batch: int = 128,
    top_k: int = 8,
    plot: bool = True,
    freeze_decoder: bool = True,
    seed: int = 6,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute SHAP-based latent-to-item loading summaries from a decoder.

    Parameters
    ----------
    decoder : nn.Module
        Trained decoder that maps latent vectors to reconstructed item values.
    Z : np.ndarray, pd.DataFrame, or torch.Tensor
        Latent factor matrix with shape `(n_samples, n_latent_factors)`.
    items : list-like
        Item names corresponding to decoder output dimensions.
    device : {"cuda", "cpu"}, default="cuda"
        Preferred device for SHAP computation. CUDA is used only when available.
    background_size : int, default=32
        Number of latent samples used as SHAP background data.
    sample_size : int, default=200
        Number of latent samples used for SHAP evaluation.
    nsamples : int, default=16
        Number of SHAP sampling steps.
    eval_batch : int, default=128
        Batch size for SHAP evaluation samples.
    top_k : int, default=8
        Number of high-strength items to plot per latent dimension.
    plot : bool, default=True
        Whether to show heatmaps of signed SHAP values.
    freeze_decoder : bool, default=True
        Whether to disable gradient updates for decoder parameters.
    seed : int, default=6
        Random seed used for background and evaluation sampling.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Signed mean SHAP values and mean absolute SHAP strengths, both indexed by
        item name with latent dimensions as columns.

    Raises
    ------
    ImportError
        If `shap` is unavailable in the current environment.
    ValueError
        If `Z` has no rows or if `eval_batch` is not positive.
    """
    if shap is None:
        raise ImportError("shap is required for compute_shap_loadings_decoder_only.")
    if eval_batch <= 0:
        raise ValueError("eval_batch must be positive.")

    use_cuda = device == "cuda" and torch.cuda.is_available()
    dev = torch.device("cuda" if use_cuda else "cpu")
    dec = decoder.eval().to(dev)

    if freeze_decoder:
        for parameter in dec.parameters():
            parameter.requires_grad_(False)

    Z_np = _to_numpy_2d(Z, "Z")
    n_all, n_latent = Z_np.shape
    if n_all == 0:
        raise ValueError("Z must contain at least one row.")

    rng = np.random.default_rng(seed)
    bg_idx = rng.choice(n_all, size=min(background_size, n_all), replace=False)
    ev_idx = rng.choice(n_all, size=min(sample_size, n_all), replace=False)

    Z_bg = torch.from_numpy(Z_np[bg_idx]).to(dev, dtype=torch.float32)
    Z_eval = torch.from_numpy(Z_np[ev_idx]).to(dev, dtype=torch.float32)

    with torch.no_grad():
        n_items = dec(Z_eval[:1]).shape[1]

    item_index = list(items)
    if len(item_index) != n_items:
        item_index = [str(item) for item in item_index[:n_items]] + [
            f"item_{idx}" for idx in range(len(item_index), n_items)
        ]

    # Reuse one decoder head while changing the output index for each item.
    head = DecoderHead(dec, out_idx=0).eval().to(dev)
    explainer = shap.GradientExplainer(head, Z_bg)

    load_signed = np.zeros((n_items, n_latent), dtype=np.float32)
    strength = np.zeros((n_items, n_latent), dtype=np.float32)

    for item_idx in range(n_items):
        head.out_idx = item_idx
        shap_batches = []

        for chunk in torch.split(Z_eval, eval_batch):
            shap_values = explainer.shap_values(chunk, nsamples=nsamples)
            if isinstance(shap_values, list):
                shap_values = shap_values[0]

            shap_values = np.asarray(shap_values)
            if shap_values.ndim == 3:
                if shap_values.shape[-1] == 1:
                    shap_values = shap_values[..., 0]
                elif shap_values.shape[0] == 1:
                    shap_values = shap_values[0]
                elif shap_values.shape[1] == 1:
                    shap_values = shap_values[:, 0, :]
                else:
                    shap_values = shap_values[..., 0]
            elif shap_values.ndim == 1:
                shap_values = shap_values[None, :]

            shap_batches.append(shap_values)

        shap_values = np.concatenate(shap_batches, axis=0)
        load_signed[item_idx, :] = shap_values.mean(axis=0)
        strength[item_idx, :] = np.abs(shap_values).mean(axis=0)

    columns = [f"Latent_{idx + 1}" for idx in range(n_latent)]
    load_signed_df = pd.DataFrame(load_signed, index=item_index, columns=columns)
    strength_df = pd.DataFrame(strength, index=item_index, columns=columns)

    # Plot high-strength decoder attributions while preserving signed direction.
    if plot:
        for column in columns:
            k = min(top_k, len(item_index))
            top_idx = strength_df[column].nlargest(k).index
            matrix = load_signed_df.loc[top_idx, [column]].sort_values(
                column, key=np.abs, ascending=False
            )
            max_abs = np.max(np.abs(matrix.values))
            if max_abs == 0:
                max_abs = 1.0

            plt.figure(figsize=(6, 0.45 * len(top_idx)))
            plt.imshow(
                matrix.values,
                aspect="auto",
                interpolation="nearest",
                cmap="coolwarm",
                vmin=-max_abs,
                vmax=max_abs,
            )
            plt.colorbar(label="mean SHAP (signed)")
            plt.yticks(range(len(top_idx)), matrix.index)
            plt.xticks([0], [column])
            plt.title(f"Top {len(top_idx)} SHAP-based Loadings for {column}")
            plt.tight_layout()
            plt.show()

    return load_signed_df, strength_df


# =============================================================================
# Evaluation functions
# =============================================================================


def check_reconstruction(
    X_test,
    reconstructed,
    qns: pd.DataFrame,
    vals: tuple[float, ...] = (0.0, 0.5, 1.0),
    atol: float = 1e-8,
) -> list[tuple[str, float, float, int]]:
    """Summarize reconstructed values conditional on selected input values.

    Parameters
    ----------
    X_test : np.ndarray or pd.DataFrame
        Test input matrix with shape `(n_samples, n_features)`.
    reconstructed : np.ndarray or pd.DataFrame
        Reconstructed matrix with shape `(n_samples, n_features)`.
    qns : pd.DataFrame
        Questionnaire table whose first column is an identifier and feature
        columns start at column index 1.
    vals : tuple of float, default=(0.0, 0.5, 1.0)
        Discrete input values to evaluate.
    atol : float, default=1e-8
        Absolute tolerance used by `np.isclose`.

    Returns
    -------
    list of tuple[str, float, float, int]
        Each tuple contains `(column_name, input_value, mean_reconstruction, count)`.
    """
    X_test_array = _to_numpy_2d(X_test, "X_test")
    reconstructed_array = _to_numpy_2d(reconstructed, "reconstructed")

    if X_test_array.shape != reconstructed_array.shape:
        raise ValueError("X_test and reconstructed must have the same shape.")
    if qns.shape[1] < X_test_array.shape[1] + 1:
        raise ValueError("qns must contain an ID column followed by all feature columns.")

    results = []
    for feature_idx in range(X_test_array.shape[1]):
        column_name = qns.columns[feature_idx + 1]
        for value in vals:
            mask = np.isclose(X_test_array[:, feature_idx], value, atol=atol)
            if mask.any():
                mean_reconstruction = reconstructed_array[mask, feature_idx].mean()
                count = int(mask.sum())
                print(f"{column_name}: {value} -> {mean_reconstruction:.3f} (n={count})")
                results.append((column_name, value, mean_reconstruction, count))

    return results