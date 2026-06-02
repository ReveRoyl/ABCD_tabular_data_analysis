"""Utilities for Canonical Correlation Analysis interpretability."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import CCA
from sklearn.impute import SimpleImputer


class CCA_Interpreter:
    """
    Tools for Canonical Correlation Analysis (CCA) and loading-based
    interpretation of two multivariate datasets.

    The typical use case is to examine the relationship between latent
    dimensions and observed variables:

    - Z: latent factors or model-derived representations.
    - X: observed features, such as questionnaire items or brain features.

    The workflow supports median imputation, z-score standardization,
    canonical variate extraction, canonical correlations, structure
    coefficients, top-loading variable extraction, optional permutation
    testing, and loading visualizations.
    """

    # ------------------------------------------------------------------
    # Utility functions
    # ------------------------------------------------------------------

    @staticmethod
    def _to_2d(a: Any) -> np.ndarray:
        """
        Convert an array-like object to a two-dimensional NumPy array.

        Parameters
        ----------
        a : array-like
            Input data. One-dimensional input is treated as a single feature.

        Returns
        -------
        ndarray
            Array with shape (n_samples, n_features).
        """
        array = np.asarray(a)
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        return array

    @staticmethod
    def _standardize(X: np.ndarray, ddof: int = 1) -> np.ndarray:
        """
        Standardize columns using z-scores.

        Parameters
        ----------
        X : ndarray
            Data matrix with samples in rows and variables in columns.
        ddof : int, default=1
            Degrees of freedom used in the standard deviation calculation.

        Returns
        -------
        ndarray
            Column-standardized matrix. Constant columns are left centered
            with a unit denominator to avoid division by zero.
        """
        X = np.asarray(X, dtype=float)
        mean = X.mean(axis=0)
        std = X.std(axis=0, ddof=ddof)
        std[std == 0] = 1.0
        return (X - mean) / std

    @staticmethod
    def _corr_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Compute Pearson correlations between all columns of two matrices.

        Parameters
        ----------
        A : ndarray of shape (n_samples, n_features_a)
            First data matrix.
        B : ndarray of shape (n_samples, n_features_b)
            Second data matrix.

        Returns
        -------
        ndarray of shape (n_features_a, n_features_b)
            Correlation matrix between columns of A and columns of B.
        """
        A = np.asarray(A, dtype=float)
        B = np.asarray(B, dtype=float)

        A = A - A.mean(axis=0, keepdims=True)
        B = B - B.mean(axis=0, keepdims=True)

        a_norm = np.linalg.norm(A, axis=0, keepdims=True)
        b_norm = np.linalg.norm(B, axis=0, keepdims=True)
        a_norm[a_norm == 0] = 1.0
        b_norm[b_norm == 0] = 1.0

        return (A.T @ B) / (a_norm.T @ b_norm)

    # ------------------------------------------------------------------
    # CCA analysis
    # ------------------------------------------------------------------

    @staticmethod
    def run_cca_interpret(
        Z: Any,
        X: Any,
        n_components: int = 5,
        impute: bool = True,
        standardize: bool = True,
        random_state: int | None = None,
        n_perm: int = 0,
        top_k: int = 10,
        item_names_X: list[str] | None = None,
        factor_names_Z: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Run CCA and return canonical correlations and loading summaries.

        Parameters
        ----------
        Z : array-like of shape (n_samples, n_latent_features)
            Latent variables or model-derived representations.
        X : array-like of shape (n_samples, n_observed_features)
            Observed variables to be interpreted against Z.
        n_components : int, default=5
            Number of canonical dimensions requested. The effective value is
            limited by the number of variables available on each side.
        impute : bool, default=True
            Whether to apply median imputation before fitting CCA.
        standardize : bool, default=True
            Whether to apply column-wise z-score standardization before CCA.
        random_state : int or None, default=None
            Random seed used for permutation testing.
        n_perm : int, default=0
            Number of permutations for the optional significance test. Set to
            0 to skip permutation testing.
        top_k : int, default=10
            Number of highest absolute X-side loadings to return per canonical
            dimension.
        item_names_X : list of str or None, default=None
            Names for the observed variables in X.
        factor_names_Z : list of str or None, default=None
            Names for the latent variables in Z.

        Returns
        -------
        dict
            Dictionary containing canonical correlations, canonical variates,
            X-side and Z-side loading matrices, top-loading X-side items, and
            optional permutation-based p-values.
        """
        Z = CCA_Interpreter._to_2d(Z)
        X = CCA_Interpreter._to_2d(X)

        if X.shape[0] != Z.shape[0]:
            raise ValueError("X and Z must contain the same number of samples.")

        if impute:
            X = SimpleImputer(strategy="median").fit_transform(X)
            Z = SimpleImputer(strategy="median").fit_transform(Z)

        if standardize:
            Xz = CCA_Interpreter._standardize(X, ddof=1)
            Zz = CCA_Interpreter._standardize(Z, ddof=1)
        else:
            Xz = X.astype(float)
            Zz = Z.astype(float)

        n_samples, n_x_features = Xz.shape
        _, n_z_features = Zz.shape
        n_components = min(n_components, n_x_features, n_z_features)

        if n_components < 1:
            raise ValueError("n_components must be at least 1 after dimension checks.")

        if item_names_X is None:
            item_names_X = [f"X{j}" for j in range(n_x_features)]
        if factor_names_Z is None:
            factor_names_Z = [f"z{j}" for j in range(n_z_features)]

        if len(item_names_X) != n_x_features:
            raise ValueError("item_names_X must match the number of columns in X.")
        if len(factor_names_Z) != n_z_features:
            raise ValueError("factor_names_Z must match the number of columns in Z.")

        # Fit CCA on the prepared matrices and extract matched canonical variates.
        cca = CCA(n_components=n_components, max_iter=5000, scale=False)
        cca.fit(Xz, Zz)
        U, V = cca.transform(Xz, Zz)

        can_corr = np.array(
            [np.corrcoef(U[:, i], V[:, i])[0, 1] for i in range(n_components)]
        )

        # Structure coefficients show how strongly original variables align
        # with each canonical variate.
        X_load = CCA_Interpreter._corr_matrix(Xz, U)
        Z_load = CCA_Interpreter._corr_matrix(Zz, V)

        component_names = [f"Factor{i + 1}" for i in range(n_components)]
        X_loadings_df = pd.DataFrame(
            X_load,
            index=item_names_X,
            columns=component_names,
        )
        Z_loadings_df = pd.DataFrame(
            Z_load,
            index=factor_names_Z,
            columns=component_names,
        )

        # Top items are ranked by absolute X-side structure coefficients.
        X_top_items = {}
        top_k = min(top_k, n_x_features)
        for i in range(n_components):
            contrib = pd.Series(np.abs(X_load[:, i]), index=item_names_X)
            X_top_items[i] = contrib.sort_values(ascending=False).head(top_k)

        p_values = None
        if n_perm > 0:
            rng = np.random.default_rng(random_state)
            perm_corrs = np.zeros((n_perm, n_components))

            for b in range(n_perm):
                perm_idx = rng.permutation(n_samples)
                Zp = Zz[perm_idx, :]

                cca_b = CCA(n_components=n_components, max_iter=3000, scale=False)
                cca_b.fit(Xz, Zp)
                U_b, V_b = cca_b.transform(Xz, Zp)

                perm_corrs[b] = [
                    np.corrcoef(U_b[:, i], V_b[:, i])[0, 1]
                    for i in range(n_components)
                ]

            p_values = np.mean(perm_corrs >= can_corr[None, :], axis=0)

        return {
            "can_corr": can_corr,
            "U": U,
            "V": V,
            "X_loadings": X_loadings_df,
            "Z_loadings": Z_loadings_df,
            "X_top_items": X_top_items,
            "p_values": p_values,
        }

    # ------------------------------------------------------------------
    # Evaluation and plotting helpers
    # ------------------------------------------------------------------

    @staticmethod
    def plot_loadings_heatmap(
        df: pd.DataFrame,
        title: str,
        vmax: float = 1.0,
    ) -> None:
        """
        Plot a loading matrix as a heatmap.

        Parameters
        ----------
        df : DataFrame
            Loading matrix, usually structure coefficients.
        title : str
            Figure title.
        vmax : float, default=1.0
            Maximum absolute value used for the heatmap color range.

        Returns
        -------
        None
            Displays the figure.
        """
        plt.figure(figsize=(8, max(4, 0.25 * df.shape[0])))
        plt.imshow(df.values, aspect="auto", vmin=-vmax, vmax=vmax)
        plt.yticks(range(df.shape[0]), df.index)
        plt.xticks(range(df.shape[1]), df.columns)
        plt.colorbar(label="Structure Coefficient (corr)")
        plt.title(title)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def get_top_items_per_cc(
        X_loadings_df: pd.DataFrame,
        top_k: int = 10,
    ) -> tuple[dict[str, pd.Series], pd.DataFrame]:
        """
        Extract top-loading observed variables for each canonical dimension.

        Parameters
        ----------
        X_loadings_df : DataFrame
            X-side structure coefficients with variables in rows and canonical
            dimensions in columns.
        top_k : int, default=10
            Number of variables to extract for each canonical dimension.

        Returns
        -------
        top_items_dict : dict
            Mapping from canonical dimension name to signed loading series.
        top_items_table : DataFrame
            Combined table of top items across canonical dimensions.
        """
        cols = list(X_loadings_df.columns)
        frames = []
        top_items_dict = {}
        top_k = min(top_k, X_loadings_df.shape[0])

        for cc in cols:
            idx = X_loadings_df[cc].abs().nlargest(top_k).index
            series = X_loadings_df.loc[idx, cc].sort_values(
                key=lambda x: x.abs(),
                ascending=False,
            )
            top_items_dict[cc] = series

            df_cc = series.reset_index()
            df_cc.columns = ["item", cc]
            frames.append(df_cc.set_index("item"))

        top_items_table = pd.concat(frames, axis=1)
        return top_items_dict, top_items_table

    @staticmethod
    def plot_top_items_per_cc(top_items_dict: dict[str, pd.Series]) -> None:
        """
        Plot horizontal bar charts for top-loading variables.

        Parameters
        ----------
        top_items_dict : dict
            Output dictionary from ``get_top_items_per_cc``. Each value should
            be a signed loading series for one canonical dimension.

        Returns
        -------
        None
            Displays one figure per canonical dimension.
        """
        for cc, series in top_items_dict.items():
            sorted_series = series.sort_values(key=lambda x: x.abs(), ascending=True)

            plt.figure(figsize=(7, max(3, 0.4 * len(sorted_series))))
            plt.barh(range(len(sorted_series)), sorted_series.values)
            plt.yticks(range(len(sorted_series)), sorted_series.index)
            plt.title(f"{cc}: Top {len(sorted_series)} Items by |Structure Coefficient|")
            plt.xlabel("Structure Coefficient (corr with U)")
            plt.tight_layout()
            plt.show()


# ----------------------------------------------------------------------
# Output utilities
# ----------------------------------------------------------------------


def save_dataframe_as_long_image(
    df: pd.DataFrame,
    save_path: str | Path = "table_long_image.png",
    title: str | None = None,
    dpi: int = 300,
) -> None:
    """
    Save a DataFrame as a long table image.

    Parameters
    ----------
    df : DataFrame
        Table to render as an image.
    save_path : str or Path, default="table_long_image.png"
        Destination path for the saved image.
    title : str or None, default=None
        Optional title displayed above the table.
    dpi : int, default=300
        Resolution of the saved image.

    Returns
    -------
    None
        Saves the image and displays the figure.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    n_rows, n_cols = df.shape
    fig_width = max(10, n_cols * 1.4)
    fig_height = max(4, n_rows * 0.35)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")

    if title is not None:
        ax.set_title(title, fontsize=14, pad=20)

    table = ax.table(
        cellText=np.round(df.values, 3),
        rowLabels=df.index,
        colLabels=df.columns,
        cellLoc="left",
        rowLoc="left",
        loc="left",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.show()

    print(f"Saved to: {save_path}")