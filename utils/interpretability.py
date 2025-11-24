# utils/Interpretability.py
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import CCA
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt


class CCA_Interpreter:
    """
    CCA_Interpreter provides a complete toolkit for performing Canonical
    Correlation Analysis (CCA) and interpreting the coupling between two
    multivariate datasets, typically:

    - Z : latent factors or model-derived representations
    - X : observed features (e.g., questionnaire items, brain features)

    Key functionalities include:
    - Median imputation
    - Standardization using sample standard deviation (ddof=1), consistent
      with academic statistical practice
    - Extraction of canonical variates (U, V) and canonical correlations
    - Computation of structure coefficients (correlations between original
      variables and canonical variates)
    - Identification of Top-K contributing variables for each canonical dimension
    - Optional permutation testing for significance
    - Heatmap visualization of loading matrices
    - Horizontal bar plots of top contributing items

    This class is designed for transparent, statistically rigorous CCA
    workflows used in psychology, neuroscience, and ML-based interpretability.
    """

    # =====================================================
    # -------------------- Utilities -----------------------
    # =====================================================

    @staticmethod
    def _to_2d(a):
        """
        Ensure the input array is two-dimensional.

        Parameters
        ----------
        a : array-like
            Input array, possibly 1D.

        Returns
        -------
        ndarray
            2D array shaped (n_samples, n_features). If the input is 1D,
            it is reshaped to a column vector.
        """
        a = np.asarray(a)
        if a.ndim == 1:
            a = a.reshape(-1, 1)
        return a

    @staticmethod
    def _standardize(X, ddof=1):
        """
        Standardize each column using z-scores computed with sample standard
        deviation (ddof=1), which is preferred in academic literature.

        Parameters
        ----------
        X : ndarray
            Data matrix.
        ddof : int, default=1
            Degrees of freedom for the standard deviation.

        Returns
        -------
        ndarray
            Standardized matrix with mean=0 and sd=1 per column.
        """
        X = np.asarray(X, dtype=float)
        mean = X.mean(axis=0)
        std = X.std(axis=0, ddof=ddof)
        std[std == 0] = 1.0
        return (X - mean) / std

    @staticmethod
    def _corr_matrix(A, B):
        """
        Compute the matrix of Pearson correlations between columns of A and B.

        Parameters
        ----------
        A : ndarray of shape (n_samples, p)
        B : ndarray of shape (n_samples, q)

        Returns
        -------
        ndarray of shape (p, q)
            Correlation matrix where entry (i, j) = corr(A[:, i], B[:, j]).
        """
        A = np.asarray(A, dtype=float)
        B = np.asarray(B, dtype=float)

        A = A - A.mean(axis=0, keepdims=True)
        B = B - B.mean(axis=0, keepdims=True)

        Anorm = np.linalg.norm(A, axis=0, keepdims=True)
        Bnorm = np.linalg.norm(B, axis=0, keepdims=True)

        Anorm[Anorm == 0] = 1.0
        Bnorm[Bnorm == 0] = 1.0

        return (A.T @ B) / (Anorm.T @ Bnorm)

    # =====================================================
    # ---------------------- Main CCA ----------------------
    # =====================================================

    @staticmethod
    def run_cca_interpret(
        Z, X,
        n_components=5,
        impute=True,
        standardize=True,
        random_state=None,
        n_perm=0,
        top_k=10,
        item_names_X=None,
        factor_names_Z=None,
    ):
        """
        Run Canonical Correlation Analysis (CCA) and produce interpretability
        outputs including canonical correlations, structure coefficients, and
        top-K contributing items.

        Parameters
        ----------
        Z : ndarray
            Latent variables (n_samples, k).
        X : ndarray
            Observed variables (n_samples, p).
        n_components : int
            Number of canonical dimensions.
        impute : bool
            Apply median imputation for missing values.
        standardize : bool
            Apply ddof=1 z-score standardization.
        random_state : int or None
            RNG seed for permutation testing.
        n_perm : int
            Number of permutations for significance testing.
        top_k : int
            Number of highest-loading items to extract per dimension.
        item_names_X : list of str or None
            Variable names for X.
        factor_names_Z : list of str or None
            Variable names for Z.

        Returns
        -------
        dict
            {
                "can_corr": array of canonical correlations,
                "U": canonical variates for X,
                "V": canonical variates for Z,
                "X_loadings": DataFrame of structure coefficients,
                "Z_loadings": DataFrame of structure coefficients,
                "X_top_items": dict of Top-K items,
                "p_values": permutation-based p-values or None
            }
        """

        Z = CCA_Interpreter._to_2d(Z)
        X = CCA_Interpreter._to_2d(X)

        # ---------- Missing values ----------
        if impute:
            X = SimpleImputer(strategy="median").fit_transform(X)
            Z = SimpleImputer(strategy="median").fit_transform(Z)

        # ---------- Standardization ----------
        if standardize:
            Xz = CCA_Interpreter._standardize(X, ddof=1)
            Zz = CCA_Interpreter._standardize(Z, ddof=1)
        else:
            Xz, Zz = X.astype(float), Z.astype(float)

        n, p = Xz.shape
        _, k = Zz.shape
        n_components = min(n_components, p, k)

        # ---------- Fit CCA ----------
        cca = CCA(n_components=n_components, max_iter=5000, scale=False)
        cca.fit(Xz, Zz)
        U, V = cca.transform(Xz, Zz)

        can_corr = np.array([
            np.corrcoef(U[:, i], V[:, i])[0, 1]
            for i in range(n_components)
        ])

        # ---------- Structure coefficients ----------
        X_load = CCA_Interpreter._corr_matrix(Xz, U)
        Z_load = CCA_Interpreter._corr_matrix(Zz, V)

        if item_names_X is None:
            item_names_X = [f"X{j}" for j in range(p)]
        if factor_names_Z is None:
            factor_names_Z = [f"z{j}" for j in range(k)]

        X_loadings_df = pd.DataFrame(
            X_load, index=item_names_X,
            columns=[f"CC{i+1}" for i in range(n_components)]
        )
        Z_loadings_df = pd.DataFrame(
            Z_load, index=factor_names_Z,
            columns=[f"CC{i+1}" for i in range(n_components)]
        )

        # ---------- Top-K items ----------
        X_top_items = {}
        top_k = min(top_k, p)
        for i in range(n_components):
            contrib = pd.Series(
                np.abs(X_load[:, i]), index=item_names_X
            )
            X_top_items[i] = contrib.sort_values(ascending=False).head(top_k)

        # ---------- Permutation test (optional) ----------
        p_values = None
        if n_perm > 0:
            rng = np.random.default_rng(random_state)
            perm_corrs = np.zeros((n_perm, n_components))
            for b in range(n_perm):
                perm_idx = rng.permutation(n)
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
            "p_values": p_values
        }

    # =====================================================
    # ---------------------- Plotting ----------------------
    # =====================================================

    @staticmethod
    def plot_loadings_heatmap(df, title, vmax=1.0):
        """
        Visualize a loading matrix (structure coefficients) as a heatmap.

        Parameters
        ----------
        df : DataFrame
            Loading matrix.
        title : str
            Title of the plot.
        vmax : float
            Maximum absolute value for color scale.
        """
        plt.figure(figsize=(8, max(4, 0.25 * df.shape[0])))
        plt.imshow(df.values, aspect='auto', vmin=-vmax, vmax=vmax)
        plt.yticks(range(df.shape[0]), df.index)
        plt.xticks(range(df.shape[1]), df.columns)
        plt.colorbar(label="Structure Coefficient (corr)")
        plt.title(title)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def get_top_items_per_cc(X_loadings_df: pd.DataFrame, top_k: int = 10):
        """
        Extract top-K highest absolute structure coefficients for each canonical
        dimension, preserving sign.

        Parameters
        ----------
        X_loadings_df : DataFrame
            Structure coefficients for X-side variables.
        top_k : int
            Number of variables to extract.

        Returns
        -------
        top_items_dict : dict
        top_items_table : DataFrame
        """
        cols = list(X_loadings_df.columns)
        frames = []
        top_items_dict = {}
        top_k = min(top_k, X_loadings_df.shape[0])

        for cc in cols:
            idx = X_loadings_df[cc].abs().nlargest(top_k).index
            s = X_loadings_df.loc[idx, cc].sort_values(
                key=lambda x: x.abs(), ascending=False
            )
            top_items_dict[cc] = s

            df_cc = s.reset_index()
            df_cc.columns = ["item", cc]
            frames.append(df_cc.set_index("item"))

        top_items_table = pd.concat(frames, axis=1)
        return top_items_dict, top_items_table

    @staticmethod
    def plot_top_items_per_cc(top_items_dict: dict):
        """
        Plot horizontal bar charts for the Top-K contributing items of each
        canonical component.

        Each plot shows:
        - Horizontal bars
        - Items sorted by absolute loading
        - Original sign preserved

        Parameters
        ----------
        top_items_dict : dict
            Output of get_top_items_per_cc().
        """
        for cc, series in top_items_dict.items():
            s = series.sort_values(
                key=lambda x: x.abs(),
                ascending=True
            )
            plt.figure(figsize=(7, max(3, 0.4 * len(s))))
            plt.barh(range(len(s)), s.values)
            plt.yticks(range(len(s)), s.index)
            plt.title(f"{cc}: Top {len(s)} Items by |Structure Coefficient|")
            plt.xlabel("Structure Coefficient (corr with U)")
            plt.tight_layout()
            plt.show()
