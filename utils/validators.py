"""Validation-variable extraction and external-validation modelling utilities."""

# =============================================================================
# Imports
# =============================================================================

import os
import re
import warnings
from functools import reduce
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from matplotlib.ticker import PercentFormatter
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.inspection import permutation_importance
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression, LogisticRegression, RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    LeaveOneGroupOut,
    StratifiedGroupKFold,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVC, SVR
from xgboost import XGBRegressor


_HAS_SGF = StratifiedGroupKFold is not None


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_EVENTNAME = "baseline_year_1_arm_1"
DEFAULT_OUTPUT_DIR = Path("../output")
VALIDATOR_TYPES = {
    "dev_delay": "cont",
    "fes_conflict": "cont",
    "n_friends": "cont",
    "school_conn": "cont",
    "avg_grades": "cont",
    "fluid_cog": "cont",
    "cryst_cog": "cont",
    "mh_service": "bin",
    "med_history": "bin",
    "brought_meds": "bin",
}
VALIDATOR_LABELS = {
    "avg_grades": "School grades",
    "fluid_cog": "Fluid intelligence",
    "cryst_cog": "Crystallized intelligence",
    "dev_delay": "Developmental delays",
    "fes_conflict": "Family conflict",
    "n_friends": "Number of friends",
    "school_conn": "School connectedness",
    "mh_service": "Mental health services",
    "med_history": "Medical history",
    "brought_meds": "Medication use",
}
PLOT_OUTCOME_ORDER = [
    "mh_service",
    "med_history",
    "brought_meds",
    "fes_conflict",
    "school_conn",
    "fluid_cog",
    "cryst_cog",
    "avg_grades",
    "dev_delay",
    "n_friends",
]


# =============================================================================
# Data loading and validator construction
# =============================================================================

def build_validators_baseline(
    root: Path,
    dict_path: Path,
    validators: Dict[str, List[str]],
    eventname: str = DEFAULT_EVENTNAME,
    out_dir: Path = DEFAULT_OUTPUT_DIR,
    dict_sheet: Optional[str] = None,
    dict_engine: str = "openpyxl",
    verbose: bool = True,
    wide_table_name: str = "validators_baseline.csv",
    output_separate_dirs: bool = False,
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Recursively search the ABCD raw directory for table files, locate variable source tables
    using the data dictionary, extract the baseline visit and output:
    - one {tag}_baseline.csv per tag
    - an aggregated validators_baseline.csv (outer-joined on src_subject_id)

    Parameters
    ----------
    root : Path
        Root directory where ABCD files are extracted (will be searched recursively).
    dict_path : Path
        Data dictionary file (must include 'var_name' and 'table_name' columns), e.g. datadict51.xlsx.
    validators : Dict[str, List[str]]
        Mapping of {tag: [var1, var2, ...]} where variable names match the data dictionary.
    eventname : str, optional
        Event name to select (default "baseline_year_1_arm_1").
    out_dir : Path, optional
        Output directory (default Path("../output")).
    dict_sheet : Optional[str], optional
        Sheet name in the data dictionary to read (if applicable).
    dict_engine : str, optional
        Excel engine to use for reading (default "openpyxl").
    verbose : bool, optional
        Whether to print progress messages (default True).
    output_separate_dirs : bool, optional
        Whether to output each tag's baseline CSV in separate directories (default False).

    Returns
    -------
    Tuple[Dict[str, pd.DataFrame], pd.DataFrame]
        out_frames: dict mapping tag -> baseline DataFrame (contains 'src_subject_id' + requested vars)
        wide: combined wide table (validators_baseline) outer-joined on 'src_subject_id'.
    """
    print("[INFO] Reading data dictionary …")

    dict_all = pd.read_excel(dict_path, engine=dict_engine, sheet_name=None)
    # dict_all is {sheet_name: df}
    dict_df = pd.concat([
        df[["var_name", "table_name"]] for df in dict_all.values()
    ]).dropna()

        
    if not out_dir.exists():
        out_dir.mkdir(parents=True)
    def find_file(table_name: str) -> Path:
        """Recursively search under root for a file whose name contains table_name with common extensions and return the first match."""
        exts = [".txt",".tsv",".csv",".txt.gz",".tsv.gz",".csv.gz"]
        for ext in exts:
            hits = list(root.rglob(f"*{table_name}*{ext}"))
            if hits:
                return hits[0]
        raise FileNotFoundError(f"No found {table_name} tan;e file (table_name={table_name})")

    out_frames: Dict[str, pd.DataFrame] = {}

    for tag, var_list in validators.items():
        if verbose:
            print(f"\n=== {tag} ===")
        sub = dict_df[dict_df.var_name.isin(var_list)]
        if sub.empty:
            if verbose:
                print(f"  [WARN] dictionary does not contain variables {var_list}")
            continue

        frames = []
        for tbl in sub.table_name.unique():   
            fp = find_file(tbl)
            comp = "gzip" if fp.suffix.endswith(".gz") else None
            suf = fp.name.lower()
            sep = "\t" if (suf.endswith(".tsv") or suf.endswith(".tsv.gz") or suf.endswith(".txt") or suf.endswith(".txt.gz")) else ","
            want = sub.loc[sub.table_name==tbl, "var_name"].tolist()
            cols = ["src_subject_id", "eventname"] + want

            if verbose:
                print(f"  reading {fp.name} ...")
            if comp:
                df_tmp = pd.read_csv(fp, sep=sep, compression=comp, encoding="utf-8", low_memory=False)
                missing = [c for c in cols if c not in df_tmp.columns]
                if missing:
                    raise KeyError(f"[{fp.name}] missing columns: {missing}")
                df_tmp = df_tmp[cols]
            else:
                df_tmp = pd.read_csv(fp, sep=sep, usecols=cols, encoding="utf-8", low_memory=False)
            frames.append(df_tmp)

        # Align table fragments by subject ID and visit.
        df_tag = reduce(lambda l, r: l.merge(r, on=["src_subject_id", "eventname"], how="inner"), frames)
        if eventname is not None:
            df_tag = (df_tag.loc[df_tag.eventname == eventname]
                            .drop(columns="eventname"))
        else:
            df_tag = df_tag
        # Store the tag-level baseline table.
        out_frames[tag] = df_tag
        if out_dir is not None:
            tag_path = out_dir / f"{tag}_baseline.csv"
            if output_separate_dirs:
                df_tag.to_csv(tag_path, index=False, encoding="utf-8")
            if verbose:
                print(f"  -> save {tag_path.name}  ({df_tag.shape[0]} rows, {df_tag.shape[1]} columns)")
        else:
            if verbose:
                print(f"  -> generated {tag} in memory  ({df_tag.shape[0]} rows, {df_tag.shape[1]} columns)")


    # Combine all tag-level validators into a wide subject-level table.
    if out_frames:
        wide = reduce(lambda l, r: l.merge(r, on="src_subject_id", how="outer"),
                      out_frames.values())
        if out_dir is not None:
            wide.to_csv(out_dir / wide_table_name, index=False, encoding="utf-8")
            if verbose:
                print(f"[OK] generate {wide_table_name} :", wide.shape)
    else:
        wide = pd.DataFrame()
        if verbose:
            print("\n[WARN] no validators extracted!")

    return out_frames, wide

def build_10_items_validators(validators_csv: Union[str, Path] = "validators_baseline.csv") -> pd.DataFrame:
    """
    Construct the ten external-validation outcomes from the baseline validator table.

    Parameters
    ----------
    validators_csv : Union[str, Path], default="validators_baseline.csv"
        Path to a CSV containing `src_subject_id` and the source variables needed for
        each external validator.

    Returns
    -------
    pd.DataFrame
        Subject-level table with `src_subject_id` and all available derived validators.

    Raises
    ------
    FileNotFoundError
        If `validators_csv` does not exist.
    ValueError
        If none of the expected validator outcomes can be constructed.
    """
    validators_csv = Path(validators_csv)
    if not validators_csv.exists():
        raise FileNotFoundError(f"Validator CSV does not exist: {validators_csv}")

    df_val = pd.read_csv(validators_csv).drop_duplicates("src_subject_id").copy()

    def row_mean_min_count(df, cols, min_count):
        sub = df[cols].copy()
        cnt = sub.notna().sum(axis=1)
        m = sub.mean(axis=1)
        m[cnt < min_count] = np.nan
        return m

    def _safe_has(cols: List[str]) -> bool:
        return all(c in df_val.columns for c in cols)

    # Developmental delay validator.
    if "dev_delay" not in df_val.columns:
        src = ["devhx_20_p", "devhx_21_p"]
        if _safe_has(src):
            tmp = df_val[src].replace({999: np.nan}).astype(float)
            both_present = tmp.notna().sum(axis=1) >= 2
            dev_delay = tmp.mean(axis=1) * 2
            dev_delay[~both_present] = np.nan
            df_val["dev_delay"] = dev_delay

    # Family-conflict validator.
    if "fes_conflict" not in df_val.columns:
        fes_cols = [f"fes_youth_q{i}" for i in range(1, 10)]
        if _safe_has(fes_cols):
            df_val["fes_conflict"] = row_mean_min_count(df_val, fes_cols, min_count=7) * 9

    # Peer-relationship validator.
    if "n_friends" not in df_val.columns:
        rr = ["resiliency5a_y", "resiliency6a_y", "resiliency5b_y", "resiliency6b_y"]
        if _safe_has(rr):
            vals = (
                df_val[rr].apply(pd.to_numeric, errors="coerce").clip(lower=0, upper=100)
            )

            def recode_normal(x):
                x = x.copy()
                x = np.where((x >= 31) & (x <= 100), 15, x)
                x = np.where((x >= 26) & (x <= 30), 14, x)
                x = np.where((x >= 21) & (x <= 25), 13, x)
                x = np.where((x >= 16) & (x <= 20), 12, x)
                x = np.where((x >= 11) & (x <= 15), 11, x)
                return pd.to_numeric(x, errors="coerce")

            def recode_close(x):
                x = x.copy()
                x = np.where((x >= 11) & (x <= 100), 11, x)
                return pd.to_numeric(x, errors="coerce")

            recoded = pd.DataFrame({
                "resiliency5a": recode_normal(vals["resiliency5a_y"].to_numpy()),
                "resiliency6a": recode_normal(vals["resiliency6a_y"].to_numpy()),
                "resiliency5b": recode_close(vals["resiliency5b_y"].to_numpy()),
                "resiliency6b": recode_close(vals["resiliency6b_y"].to_numpy()),
            }, index=df_val.index)

            df_val["n_friends"] = row_mean_min_count(recoded, recoded.columns.tolist(), min_count=3) * 4

    # School-connectedness validator.
    if "school_conn" not in df_val.columns:
        env_cols    = [f"school_{i}_y" for i in (2,3,4,5,6,7)]
        invol_cols  = [f"school_{i}_y" for i in (8,9,10,12)]
        disgn_cols  = [f"school_{i}_y" for i in (15,17)]
        if _safe_has(env_cols) and _safe_has(invol_cols) and _safe_has(disgn_cols):
            env    = row_mean_min_count(df_val, env_cols,   min_count=5) * 6
            invol  = row_mean_min_count(df_val, invol_cols, min_count=3) * 4
            disgn  = row_mean_min_count(df_val, disgn_cols, min_count=2) * 2
            total  = env + invol + 10 - disgn
            total[env.isna() | invol.isna() | disgn.isna()] = np.nan
            df_val["school_conn"] = total

    if "avg_grades" not in df_val.columns and "kbi_p_grades_in_school" in df_val.columns:
        df_val["avg_grades"] = df_val["kbi_p_grades_in_school"].replace({6: np.nan, -1: np.nan}).astype(float)

    if "fluid_cog" not in df_val.columns and "nihtbx_fluidcomp_uncorrected" in df_val.columns:
        df_val["fluid_cog"] = pd.to_numeric(df_val["nihtbx_fluidcomp_uncorrected"], errors="coerce")
    if "cryst_cog" not in df_val.columns and "nihtbx_cryst_uncorrected" in df_val.columns:
        df_val["cryst_cog"] = pd.to_numeric(df_val["nihtbx_cryst_uncorrected"], errors="coerce")

    if "mh_service" not in df_val.columns and "kbi_p_c_mh_sa" in df_val.columns:
        df_val["mh_service"] = df_val["kbi_p_c_mh_sa"].map({1: 1.0, 2: 0.0, 3: np.nan}).astype(float)

    if "med_history" not in df_val.columns:
        mh1 = ["medhx_1a", "medhx_1b"]
        if _safe_has(mh1):
            tmp = df_val[mh1].replace({6: np.nan}).apply(pd.to_numeric, errors="coerce")
            both_present = tmp.notna().sum(axis=1) >= 2
            medhx1 = tmp.max(axis=1)
            medhx1[~both_present] = np.nan
            df_val["med_history"] = medhx1

    if "brought_meds" not in df_val.columns and "brought_medications" in df_val.columns:
        df_val["brought_meds"] = df_val["brought_medications"].map({0: 1.0, 1: 1.0, 3: 0.0, 2: np.nan}).astype(float)

    validator_info = VALIDATOR_TYPES.copy()
    needed_outcomes = [y for y in validator_info.keys() if y in df_val.columns]
    if not needed_outcomes:
        raise ValueError("No valid validators found in validators CSV.")
    df_val = df_val[["src_subject_id"] + needed_outcomes]

    return df_val


# =============================================================================
# Model-score utilities
# =============================================================================


def build_model_scores(qns: pd.DataFrame, latent_factors: np.ndarray, id_colname: str = "src_subject_id"):
    """
    Build a DataFrame containing subject IDs and latent factor scores. Must be full sample, i.e. latent factors from X_scaled.

    Parameters
    ----------
    qns : pd.DataFrame
        Questionnaire DataFrame, where the first column is subject IDs.
    latent_factors : np.ndarray
        Latent factor representation with shape (n_samples, n_factors).
    id_colname : str, default="src_subject_id"
        Name to assign to the ID column in the output.

    Returns
    -------
    AE_scores : pd.DataFrame
        DataFrame with columns [id_colname, factor_1, factor_2, ..., factor_k].
    """
    # Extract the subject ID column.
    id_series = qns.iloc[:, 0].copy()
    id_series.name = id_colname

    # Create the latent-factor score table.
    factor_cols = [f"factor_{i+1}" for i in range(latent_factors.shape[1])]
    df_factors = pd.DataFrame(latent_factors, columns=factor_cols)

    # Concatenate IDs and factor scores with aligned indices.
    AE_scores = pd.concat(
        [id_series.reset_index(drop=True), df_factors.reset_index(drop=True)],
        axis=1
    )
    return AE_scores

def get_scores_by_k(model_fn, X_scaled, subject_id, Kmax=5, prefix="factor"):
    """
    Loop over K=1..Kmax, train model and collect factor scores.

    Parameters
    ----------
    model_fn : callable
        A function that takes `k` and returns a fitted model object.
        - For AE: should have .train() and .evaluate_on_data(X) -> latent_factors, ...
        - For EFA: should have .fit(X) and .transform(X) -> latent_factors.
    X_scaled : np.ndarray
        Input data of shape (n_samples, n_features).
    subject_id : pd.Series
        Subject IDs to align with factor scores.
    Kmax : int
        Maximum number of factors to extract.
    prefix : str
        Prefix for factor column names (e.g. "factor").

    Returns
    -------
    scores_by_k : dict[int, pd.DataFrame]
        Dictionary of {k: DataFrame with [subject_id, factor_1..factor_k]}.
    """
    scores_by_k = {}

    for k in range(1, Kmax + 1):
        # Fit the K-factor model.
        model = model_fn(k)

        if hasattr(model, "train") and hasattr(model, "evaluate_on_data"):  
            # Autoencoder-style interface.
            model.train(show_plot=True)
            latent_factors, _, _, _, _, _ = model.evaluate_on_data(X_scaled)
        else:
            # FactorAnalyzer-style interface.
            model.fit(X_scaled)
            latent_factors = model.transform(X_scaled)

        # Store K-factor scores.
        factor_cols = [f"{prefix}_{i+1}" for i in range(k)]
        df_factors = pd.DataFrame(latent_factors, columns=factor_cols)
        df = pd.concat(
            [subject_id.reset_index(drop=True), df_factors.reset_index(drop=True)],
            axis=1
        )
        scores_by_k[k] = df

    return scores_by_k



# =============================================================================
# Evaluation and comparison functions
# =============================================================================


def compare_efa_poly_vs_ae_poly(
    factors_scores: pd.DataFrame,
    efa_scores_by_k: Dict[int, pd.DataFrame],
    validators_csv: Union[str, Path] = "validators_baseline.csv",
    save_dir: str = "efa_vs_model_plots_poly_compare",
    ae_scores_by_k: Optional[Dict[int, pd.DataFrame]] = None,
    degree: int = 1,
    random_seed: int = 6,
    bootstrap_B: int = 30,
    use_ddof1_zscore: bool = True,
    model_reg: str = "ols",
    model_clf: str = "logit",
    use_poly_features: bool = True,
):
    """
    Compare AE and EFA external-validation performance across factor counts.

    Parameters
    ----------
    factors_scores : pd.DataFrame
        Subject-level AE factor scores used when `ae_scores_by_k` is not provided.
    efa_scores_by_k : Dict[int, pd.DataFrame]
        EFA factor-score tables keyed by factor count K.
    validators_csv : Union[str, Path], default="validators_baseline.csv"
        CSV containing source columns or derived validator outcomes.
    save_dir : str, default="efa_vs_model_plots_poly_compare"
        Directory for summary tables and figures.
    ae_scores_by_k : Optional[Dict[int, pd.DataFrame]], default=None
        AE factor-score tables keyed by factor count K.
    degree : int, default=1
        Polynomial degree used for the target model block.
    random_seed : int, default=6
        Seed used for bootstrap sampling and stochastic estimators.
    bootstrap_B : int, default=30
        Number of bootstrap samples for final-block and gain comparisons.
    use_ddof1_zscore : bool, default=True
        Whether to standardize predictors with sample-standard-deviation scaling.
    model_reg : str, default="ols"
        Continuous-outcome estimator name.
    model_clf : str, default="logit"
        Binary-outcome estimator name.
    use_poly_features : bool, default=True
        Whether to construct polynomial features for degree greater than one.

    Returns
    -------
    Dict[str, Any]
        Summary table and output paths for the CSV and figure files.
    """

    rng = np.random.default_rng(random_seed)
    np.random.seed(random_seed)
    os.makedirs(save_dir, exist_ok=True)

    # Load the validator outcome table.
    validator_info = VALIDATOR_TYPES.copy()
    df_val = build_10_items_validators(validators_csv)

    # Helper functions for standardization and R² calculations.
    def _zscore_ddof1_inplace(df_ref: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        x = df_ref[cols].astype(float)
        mu = x.mean(axis=0); sd = x.std(axis=0, ddof=1).replace(0, np.nan)
        df_ref[cols] = (x - mu) / sd
        return df_ref

    def _standardize_inplace(df_ref: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        if use_ddof1_zscore:
            return _zscore_ddof1_inplace(df_ref, cols)
        else:
            scaler = StandardScaler()
            df_ref[cols] = scaler.fit_transform(df_ref[cols])
            return df_ref

    def loglik_null_binary(y: np.ndarray) -> float:
        y = y.astype(float)
        p = float(np.nanmean(y))
        p = np.clip(p, 1e-8, 1-1e-8)
        return float(np.nansum(y*np.log(p) + (1-y)*np.log(1-p)))

    def _bin_loglik_from_pred(y, p):
        p = np.clip(np.asarray(p, dtype=float), 1e-8, 1-1e-8)
        y = np.asarray(y, dtype=float)
        return float(np.sum(y*np.log(p) + (1-y)*np.log(1-p)))

    def r2_nagelkerke(ll_null: float, ll_model: float, n: int) -> float:
        if ll_model <= ll_null:
            return 0.0
        cs = 1 - np.exp((2.0/n) * (ll_null - ll_model))
        denom = 1 - np.exp((2.0/n) * ll_null)
        if not np.isfinite(denom) or np.isclose(denom, 0):
            return np.nan
        r2 = cs / denom
        return float(np.clip(r2, 0.0, 1.0))

    def _total_r2_cont_in(d_all: pd.DataFrame, y: str, cols: List[str]) -> float:
        """Compute in-sample Total R² for continuous outcomes using the selected estimator."""
        if not cols:
            return np.nan
        s = d_all[cols].std(numeric_only=True, ddof=1)
        used = [c for c in cols if s.get(c, 0.0) > 1e-12]
        if not used:
            return np.nan
        d = _standardize_inplace(d_all.copy(), used)
        X = d[used].astype(float).to_numpy()
        yv = d[y].astype(float).to_numpy()

        mr = model_reg.lower()
        try:
            if mr == "ols":
                m = smf.ols(f"{y} ~ " + " + ".join(used), data=d).fit()
                return float(m.rsquared)

            elif mr == "ridge":
                m = RidgeCV(alphas=np.logspace(-4, 3, 30)).fit(X, yv)
                yhat = m.predict(X)
                return float(r2_score(yv, yhat))

            elif mr == "kernel_ridge_rbf":
                m = KernelRidge(alpha=1.0, kernel="rbf")
                m.fit(X, yv)
                yhat = m.predict(X)
                return float(r2_score(yv, yhat))

            elif mr == "rf":
                m = RandomForestRegressor(
                    n_estimators=400, max_depth=None, random_state=random_seed, n_jobs=-1
                ).fit(X, yv)
                yhat = m.predict(X)
                return float(r2_score(yv, yhat))
            
            elif mr in ("xgb", "xgboost"):
                # XGBoost regressor.
                m = XGBRegressor(
                    n_estimators=400,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_lambda=1.0,
                    random_state=random_seed,
                    n_jobs=-1,
                    tree_method="hist",
                )
                m.fit(X, yv)
                yhat = m.predict(X)
                return float(r2_score(yv, yhat))

            elif mr == "gbrt":
                m = HistGradientBoostingRegressor(random_state=random_seed).fit(X, yv)
                yhat = m.predict(X)
                return float(r2_score(yv, yhat))

            elif mr == "svm_rbf":
                m = SVR(kernel="rbf").fit(X, yv)
                yhat = m.predict(X)
                return float(r2_score(yv, yhat))

            elif mr == "mlp":
                m = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=800,
                                random_state=random_seed).fit(X, yv)
                yhat = m.predict(X)
                return float(r2_score(yv, yhat))

            else:
                return np.nan
        except Exception:
            return np.nan


    def _total_r2_bin_in(d_all: pd.DataFrame, y: str, cols: List[str]) -> float:
        """Compute in-sample Nagelkerke R² for binary outcomes from predicted probabilities."""
        if not cols:
            return np.nan
        s = d_all[cols].std(numeric_only=True, ddof=1)
        used = [c for c in cols if s.get(c, 0.0) > 1e-12]
        if not used:
            return np.nan
        d = _standardize_inplace(d_all.copy(), used)
        X = d[used].astype(float).to_numpy()
        yv = d[y].astype(float).to_numpy()
        if pd.Series(yv).nunique(dropna=True) < 2:
            return np.nan
        ll0 = loglik_null_binary(yv)

        mc = model_clf.lower()
        try:
            if mc == "logit":
                m = smf.logit(f"{y} ~ " + " + ".join(used), data=d).fit(disp=0, method="newton", maxiter=100)
                p_hat = np.asarray(m.predict(d))

            elif mc == "rf":
                m = RandomForestClassifier(
                    n_estimators=600, max_depth=None, class_weight="balanced",
                    random_state=random_seed, n_jobs=-1
                ).fit(X, yv)
                p_hat = m.predict_proba(X)[:, 1]

            elif mc == "gbrt":
                m = HistGradientBoostingClassifier(random_state=random_seed).fit(X, yv)
                p_hat = m.predict_proba(X)[:, 1]

            elif mc == "svm_rbf":
                m = SVC(kernel="rbf", probability=True, random_state=random_seed).fit(X, yv)
                p_hat = m.predict_proba(X)[:, 1]

            elif mc == "mlp":
                m = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=800,
                                random_state=random_seed).fit(X, yv)
                p_hat = m.predict_proba(X)[:, 1]

            else:
                return np.nan

            ll1 = _bin_loglik_from_pred(yv, p_hat)
            return r2_nagelkerke(ll0, ll1, len(d))
        except Exception:
            return np.nan
        
    def _total_r2_by_type(d_all, y, cols, ytype):
        return _total_r2_cont_in(d_all, y, cols) if ytype == "cont" else _total_r2_bin_in(d_all, y, cols)

    # Bootstrap p-value comparing AE vs EFA total R² at final block
    def _pvalue_bootstrap_final_block(panel_in: pd.DataFrame, y: str, ytype: str,
                                      ae_cols: List[str], efa_cols: List[str],
                                      B: int, seed: int) -> float:
        use_cols = sorted(set(ae_cols) | set(efa_cols))
        subset_cols = [y] + use_cols
        if len(subset_cols) <= 1:
            return np.nan
        d_all = panel_in.dropna(subset=subset_cols, how="any").copy()
        if d_all.empty:
            return np.nan
        r2_ae  = _total_r2_by_type(d_all, y, ae_cols,  ytype)
        r2_efa = _total_r2_by_type(d_all, y, efa_cols, ytype)
        if not (np.isfinite(r2_ae) and np.isfinite(r2_efa)):
            return np.nan
        obs_diff = r2_ae - r2_efa

        rng_local = np.random.default_rng(seed)
        diffs = []
        n = len(d_all)
        for _ in range(int(B)):
            idx = rng_local.integers(0, n, size=n)
            db = d_all.iloc[idx].copy()
            try:
                r2a = _total_r2_by_type(db, y, ae_cols,  ytype)
                r2e = _total_r2_by_type(db, y, efa_cols, ytype)
            except Exception:
                continue
            if np.isfinite(r2a) and np.isfinite(r2e):
                diffs.append(r2a - r2e)
        if len(diffs) < 20:
            return np.nan
        diffs = np.array(diffs, dtype=float)
        p = 2 * min((diffs >= obs_diff).mean(), (diffs <= obs_diff).mean())
        return float(np.clip(p, 0.0, 1.0))

    # Build cumulative linear predictor blocks for K = 1..Kmax.
    def _build_linear_blocks(bases: List[str]) -> List[List[str]]:
        # Cumulative inclusion of base factors.
        return [bases[:i] for i in range(1, len(bases)+1)]

    # Bootstrap p-value for nonlinear gain within a model block.
    def _p_boot_gain(panel_in: pd.DataFrame, y: str, ytype: str,
                     cols_lin: List[str], cols_deg: List[str],
                     B: int, seed: int) -> float:
        # Same-sample comparison: ΔR² = R²(degree) - R²(linear).
        subset_cols = [y] + cols_lin + cols_deg
        d_all = panel_in.dropna(subset=subset_cols, how="any").copy()
        if d_all.empty:
            return np.nan
        g_obs = _total_r2_by_type(d_all, y, cols_deg, ytype) - _total_r2_by_type(d_all, y, cols_lin, ytype)
        if not np.isfinite(g_obs):
            return np.nan

        rng_local = np.random.default_rng(seed)
        diffs = []
        n = len(d_all)
        for _ in range(int(B)):
            idx = rng_local.integers(0, n, size=n)
            db = d_all.iloc[idx].copy()
            try:
                gd = _total_r2_by_type(db, y, cols_deg, ytype)
                gl = _total_r2_by_type(db, y, cols_lin, ytype)
                g  = gd - gl
            except Exception:
                continue
            if np.isfinite(g):
                diffs.append(g)
        if len(diffs) < 20:
            return np.nan
        diffs = np.array(diffs, dtype=float)
        p = 2 * min((diffs >= g_obs).mean(), (diffs <= g_obs).mean())
        return float(np.clip(p, 0.0, 1.0))

    # Prepare factor-score tables across K.
    def _sorted_cols(df, prefix):
        return sorted(
            [c for c in df.columns if c.startswith(prefix)],
            key=lambda s: int(re.search(r"_(\d+)$", s).group(1)) if re.search(r"_(\d+)$", s) else 10**9
        )

    ae_df_base = factors_scores.drop_duplicates("src_subject_id").copy()
    ae_base_all = _sorted_cols(ae_df_base, "factor_")

    K_list = sorted(efa_scores_by_k.keys())
    if not K_list:
        raise ValueError("efa_scores_by_k is empty; you must provide EFA factor scores keyed by K.")

    # Compare AE and EFA performance for each factor count.
    name_map = VALIDATOR_LABELS.copy()

    rows_final = []  # per outcome × K summary rows
    p_rows = []      # p-values (final-block AE vs EFA) and nonlinear gain p-values

    for K in K_list:
        # AE_K
        if ae_scores_by_k is not None and K in ae_scores_by_k:
            ae_k = ae_scores_by_k[K].drop_duplicates("src_subject_id").copy()
            ae_cols = _sorted_cols(ae_k, "factor_")[:K]
            ae_k = ae_k[["src_subject_id"] + ae_cols]
        else:
            if len(ae_base_all) < K:
                raise ValueError(f"AE factor columns are insufficient: need at least {K}, found {len(ae_base_all)}.")
            ae_cols = ae_base_all[:K]
            ae_k = ae_df_base[["src_subject_id"] + ae_cols].copy()

        # EFA_K
        efa_k_raw = efa_scores_by_k[K].drop_duplicates("src_subject_id").copy()
        efa_cols = _sorted_cols(efa_k_raw, "efa_")[:K]
        if len(efa_cols) < K:
            raise ValueError(f"EFA K={K} has insufficient efa_* columns: need {K}.")
        efa_k = efa_k_raw[["src_subject_id"] + efa_cols].copy()

        # Merge panel
        panel_k = df_val.merge(ae_k,  on="src_subject_id", how="inner") \
                        .merge(efa_k, on="src_subject_id", how="inner")
        if panel_k.empty:
            continue

        # ---- Build polynomial features for degree (>=1) ----
        def build_poly_blocks(df_ref: pd.DataFrame, bases: List[str], degree: int, tag: str) -> List[List[str]]:
            if degree <= 1:
                # linear-only blocks (no features added)
                return [bases[:i] for i in range(1, len(bases)+1)]
            X = df_ref[bases].astype(float).copy()
            mu = X.mean(0); sd = X.std(0, ddof=1).replace(0, np.nan)
            Xz = (X - mu) / sd
            poly = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=False)
            Z = poly.fit_transform(Xz[bases])
            names = poly.get_feature_names_out(bases)  # e.g., "factor_1", "factor_1 factor_2", "factor_1^2"
            out_cols = []
            for j, nm in enumerate(names):
                col = f"{tag}__{nm.replace(' ', '__x__').replace('^', '__p')}"
                df_ref[col] = Z[:, j]
                out_cols.append(col)

            token_re = re.compile(r"[A-Za-z_]\w*")
            def uses_within_first_i(nm: str, i: int) -> bool:
                vars_in = set(token_re.findall(nm))
                return vars_in.issubset(set(bases[:i]))

            blocks = []
            for i in range(1, len(bases)+1):
                cols_i = [c for c, nm in zip(out_cols, names) if uses_within_first_i(nm, i)]
                blocks.append(cols_i)
            return blocks

        panel_work = panel_k.copy()

        # blocks for "target degree"
        # ae_blocks_deg  = build_poly_blocks(panel_work, ae_cols,  degree, tag=f"AEK{K}")
        # efa_blocks_deg = build_poly_blocks(panel_work, efa_cols, degree, tag=f"EFAK{K}")
        ae_blocks_deg  = build_poly_blocks(panel_work, ae_cols,  degree if use_poly_features else 1, tag=f"AEK{K}")
        efa_blocks_deg = build_poly_blocks(panel_work, efa_cols, degree if use_poly_features else 1, tag=f"EFAK{K}")
        # Linear blocks provide the reference for nonlinear gain.
        ae_blocks_lin  = _build_linear_blocks(ae_cols)
        efa_blocks_lin = _build_linear_blocks(efa_cols)

        # ---- For each outcome: compute linear R², degree R², and nonlinear gain ----
        for y, ytype in validator_info.items():
            if y not in panel_work.columns:
                continue

            # final-block columns
            ae_cols_final_deg  = [c for c in ae_blocks_deg[K-1]  if c in panel_work.columns]
            efa_cols_final_deg = [c for c in efa_blocks_deg[K-1] if c in panel_work.columns]
            ae_cols_final_lin  = ae_blocks_lin[K-1]   # base columns only
            efa_cols_final_lin = efa_blocks_lin[K-1]

            # --- compute R² on their own listwise samples (final block)
            def r2_on_common_sample(cols):
                d = panel_work.dropna(subset=[y] + cols, how="any").copy()
                if d.empty: 
                    return np.nan
                return _total_r2_by_type(d, y, cols, ytype)

            r2_lin_ae  = r2_on_common_sample(ae_cols_final_lin)
            r2_lin_efa = r2_on_common_sample(efa_cols_final_lin)
            r2_deg_ae  = r2_on_common_sample(ae_cols_final_deg)
            r2_deg_efa = r2_on_common_sample(efa_cols_final_deg)

            # nonlinear gain (degree - linear)
            ae_gain  = (r2_deg_ae  - r2_lin_ae)  if (np.isfinite(r2_deg_ae)  and np.isfinite(r2_lin_ae))  else np.nan
            efa_gain = (r2_deg_efa - r2_lin_efa) if (np.isfinite(r2_deg_efa) and np.isfinite(r2_lin_efa)) else np.nan

            rows_final.append({
                "outcome": y, "pretty": name_map.get(y, y),
                "K": K,
                "AE_lin":  r2_lin_ae,
                "EFA_lin": r2_lin_efa,
                "AE":      r2_deg_ae,
                "EFA":     r2_deg_efa,
                "AE_nonlin_gain":  ae_gain,
                "EFA_nonlin_gain": efa_gain,
            })

            # p-values
            pval_total = _pvalue_bootstrap_final_block(
                panel_work, y, ytype,
                ae_cols_final_deg, efa_cols_final_deg,
                B=bootstrap_B, seed=int(random_seed + 13*K)
            )
            p_gain_ae  = _p_boot_gain(panel_work, y, ytype,
                                      ae_cols_final_lin, ae_cols_final_deg,
                                      B=bootstrap_B, seed=int(random_seed + 31*K))
            p_gain_efa = _p_boot_gain(panel_work, y, ytype,
                                      efa_cols_final_lin, efa_cols_final_deg,
                                      B=bootstrap_B, seed=int(random_seed + 37*K))

            p_rows.append({
                "outcome": y, "K": K,
                "p_value_compare_final": pval_total,
                "p_value_AE_nonlin_gain": p_gain_ae,
                "p_value_EFA_nonlin_gain": p_gain_efa,
            })

    # Export summary tables.
    res_final = pd.DataFrame(rows_final)
    p_df = pd.DataFrame(p_rows)

    # Summary table for each outcome and factor count.
    if not res_final.empty:
        res_final["diff (AE - EFA)"] = res_final["AE"] - res_final["EFA"]
        with np.errstate(divide='ignore', invalid='ignore'):
            res_final["improve % (AE/EFA - 1)"] = np.where(
                res_final["EFA"].abs() > 1e-8, res_final["AE"] / res_final["EFA"] - 1, np.nan
            )
        res_out = res_final.merge(p_df, on=["outcome","K"], how="left")
    else:
        res_out = pd.DataFrame(columns=[
            "outcome","pretty","K",
            "AE_lin","EFA_lin","AE","EFA",
            "AE_nonlin_gain","EFA_nonlin_gain",
            "diff (AE - EFA)","improve % (AE/EFA - 1)",
            "p_value_compare_final","p_value_AE_nonlin_gain","p_value_EFA_nonlin_gain"
        ])

    csv_path = os.path.join(save_dir, "efa_vs_ae_perK_summary.csv")
    res_out.sort_values(["outcome","K"]).to_csv(csv_path, index=False, float_format="%.6f")

    # Plot total R² and nonlinear gain across K.
    # Total R² / Nagelkerke R² figure.
    if not res_final.empty:
        outcomes = [o for o in PLOT_OUTCOME_ORDER if o in res_final["outcome"].unique()]

        n = len(outcomes)
        n_col, n_row = (1,1) if n == 1 else (2, int(np.ceil(n/2)))
        fig, axes = plt.subplots(n_row, n_col, figsize=(n_col*4.8, n_row*3.3), sharex=False, sharey=False)
        axes = np.atleast_1d(axes)

        p_map_total = {}
        if not p_df.empty and "p_value_compare_final" in p_df.columns:
            for _, r in p_df.iterrows():
                p_map_total[(r["outcome"], int(r["K"]))] = r["p_value_compare_final"]

        for idx, outcome in enumerate(outcomes):
            ax = axes.flat[idx]
            sub = res_out[res_out["outcome"] == outcome].sort_values("K")
            ax.plot(sub["K"], sub["AE"],  marker="o", lw=1.8, label="AE")
            ax.plot(sub["K"], sub["EFA"], marker="s", lw=1.8, label="EFA")

            # Significance markers for final-block AE versus EFA comparisons.
            for K in sub["K"].unique():
                p = float(p_map_total.get((outcome, int(K)), np.nan))
                if np.isfinite(p) and p < .05:
                    ae_val = float(sub.loc[sub["K"] == K, "AE"].iloc[0])
                    efa_val = float(sub.loc[sub["K"] == K, "EFA"].iloc[0])
                    y_mid = np.nanmean([ae_val, efa_val])
                    stars = "***" if p < .001 else "**" if p < .01 else "*"
                    ax.annotate(stars, (int(K), y_mid), xytext=(0, 6),
                                textcoords="offset points", ha="center", fontsize=8, weight="bold")

            nice = VALIDATOR_LABELS.get(outcome, outcome)
            ax.set_title(nice, fontsize=9)
            ax.set_xticks(sorted(sub["K"].unique()))
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=1))
            if idx == 0:
                ax.legend(frameon=False, fontsize=8)

        total_axes = axes.size if hasattr(axes, 'size') else len(axes.flat)
        for j in range(len(outcomes), total_axes):
            fig.delaxes(axes.flat[j])

        fig.text(0.5, 0.04, "Number of factors (K)", ha="center", fontsize=12)
        fig.text(0.06, 0.5, "Total R² / Nagelkerke R²", va="center", rotation="vertical", fontsize=12)
        plt.tight_layout(rect=[0.06, 0.04, 1, 1])
        png_total = os.path.join(save_dir, "efa_vs_ae_R2_vs_K.png")
        plt.savefig(png_total, dpi=300)
        plt.close(fig)
    else:
        png_total = os.path.join(save_dir, "efa_vs_ae_R2_vs_K.png")
        plt.figure(figsize=(6, 4))
        plt.title("No valid outcomes")
        plt.savefig(png_total, dpi=300)
        plt.close()

    # Nonlinear-gain figure.
    if not res_final.empty:
        outcomes = [o for o in PLOT_OUTCOME_ORDER if o in res_final["outcome"].unique()]

        n = len(outcomes)
        n_col, n_row = (1,1) if n == 1 else (2, int(np.ceil(n/2)))
        fig, axes = plt.subplots(n_row, n_col, figsize=(n_col*4.8, n_row*3.3), sharex=False, sharey=False)
        axes = np.atleast_1d(axes)

        p_map_gain_ae = {}
        p_map_gain_efa = {}
        if not p_df.empty:
            if "p_value_AE_nonlin_gain" in p_df.columns:
                for _, r in p_df.iterrows():
                    p_map_gain_ae[(r["outcome"], int(r["K"]))] = r["p_value_AE_nonlin_gain"]
            if "p_value_EFA_nonlin_gain" in p_df.columns:
                for _, r in p_df.iterrows():
                    p_map_gain_efa[(r["outcome"], int(r["K"]))] = r["p_value_EFA_nonlin_gain"]

        for idx, outcome in enumerate(outcomes):
            ax = axes.flat[idx]
            sub = res_out[res_out["outcome"] == outcome].sort_values("K")
            ax.plot(sub["K"], sub["AE_nonlin_gain"],  marker="o", lw=1.8, label="AE ΔR²")
            ax.plot(sub["K"], sub["EFA_nonlin_gain"], marker="s", lw=1.8, label="EFA ΔR²")

            # Significance markers for AE and EFA nonlinear-gain tests.
            for K in sub["K"].unique():
                # AE stars
                p_ae  = float(p_map_gain_ae.get((outcome, int(K)), np.nan))
                if np.isfinite(p_ae) and p_ae < .05:
                    y_a = float(sub.loc[sub["K"] == K, "AE_nonlin_gain"].iloc[0])
                    stars = "***" if p_ae < .001 else "**" if p_ae < .01 else "*"
                    ax.annotate(stars, (int(K), y_a), xytext=(0, 6),
                                textcoords="offset points", ha="center", fontsize=8, weight="bold")
                # EFA stars
                p_efa = float(p_map_gain_efa.get((outcome, int(K)), np.nan))
                if np.isfinite(p_efa) and p_efa < .05:
                    y_e = float(sub.loc[sub["K"] == K, "EFA_nonlin_gain"].iloc[0])
                    stars = "***" if p_efa < .001 else "**" if p_efa < .01 else "*"
                    ax.annotate(
                        stars,
                        (int(K), y_e),
                        xytext=(0, -10),
                        textcoords="offset points",
                        ha="center",
                        fontsize=7,
                        weight="bold",
                    )

            nice = VALIDATOR_LABELS.get(outcome, outcome)
            ax.set_title(nice + " — Nonlinear gain (ΔR²)", fontsize=9)
            ax.set_xticks(sorted(sub["K"].unique()))
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=1))
            if idx == 0:
                ax.legend(frameon=False, fontsize=8)

        for j in range(len(outcomes), axes.size if hasattr(axes, 'size') else len(axes.flat)):
            fig.delaxes(axes.flat[j])

        fig.text(0.5, 0.04, "Number of factors (K)", ha="center", fontsize=12)
        fig.text(0.06, 0.5, "Nonlinear gain ΔR²", va="center", rotation="vertical", fontsize=12)
        plt.tight_layout(rect=[0.06, 0.04, 1, 1])
        png_gain = os.path.join(save_dir, "efa_vs_ae_NONLINEAR_GAIN_vs_K.png")
        plt.savefig(png_gain, dpi=300)
        plt.close(fig)
    else:
        png_gain = os.path.join(save_dir, "efa_vs_ae_NONLINEAR_GAIN_vs_K.png")
        plt.figure(figsize=(6, 4))
        plt.title("No valid outcomes")
        plt.savefig(png_gain, dpi=300)
        plt.close()

    # Print a compact preview of the summary table.
    print(res_out.head(20))

    return {
        "summary_perK": res_out,
        "paths": {
            "summary_csv": csv_path,
            "grid_png": png_total,
            "gain_png": png_gain
        }
    }


def predict_one_validator(
    validator: str,
    X: pd.DataFrame,
    panel: pd.DataFrame,
    model: str = "rf",
    ytype: Optional[str] = None,
    cov_cont: Optional[List[str]] = None,
    cov_cat: Optional[List[str]] = None,
    cv_folds: int = 5,
    random_state: int = 6,
    compute_importance: bool = False,
    id_col: str = "src_subject_id",
    groups_col: Optional[str] = None,
    group_strategy: str = "logo",
) -> Dict[str, Any]:
    """
    Predict one external validator from latent factors and optional covariates.

    Parameters
    ----------
    validator : str
        Outcome column in `panel`.
    X : pd.DataFrame
        Latent-factor table. If `id_col` is absent, the index is converted into
        an ID column before merging.
    panel : pd.DataFrame
        Table containing the validator outcome and optional covariates.
    model : str, default="rf"
        Estimator name. Continuous outcomes support rf, gbrt, ols, ridge, krbf,
        svm_rbf, and xgb. Binary outcomes support rf, gbrt, logit, svm_rbf, and xgb.
    ytype : Optional[str], default=None
        Outcome type: "cont" or "bin". If None, the type is inferred from values.
    cov_cont : Optional[List[str]], default=None
        Continuous covariate columns.
    cov_cat : Optional[List[str]], default=None
        Categorical covariate columns to one-hot encode.
    cv_folds : int, default=5
        Number of folds for non-grouped or grouped K-fold cross-validation.
    random_state : int, default=6
        Random seed for stochastic estimators and splitters.
    compute_importance : bool, default=False
        Whether to compute permutation importance on the full aligned sample.
    id_col : str, default="src_subject_id"
        Subject identifier used to align factor scores and outcomes.
    groups_col : Optional[str], default=None
        Optional grouping column for grouped cross-validation.
    group_strategy : str, default="logo"
        Grouped CV strategy: "logo", "groupkfold", or "auto".

    Returns
    -------
    Dict[str, Any]
        Model-performance summary, optional permutation importance table, feature
        dimensions, and the feature columns used in the analysis.
    """

    cov_cont = cov_cont or []
    cov_cat = cov_cat or []

    # Ensure both factor and outcome tables contain an explicit subject ID column.
    Xf = X.copy()
    P = panel.copy()

    if id_col not in Xf.columns:
        Xf = Xf.reset_index().rename(columns={Xf.index.name or "index": id_col})
    if id_col not in P.columns:
        P = P.reset_index().rename(columns={P.index.name or "index": id_col})

    Xf[id_col] = Xf[id_col].astype(str)
    P[id_col]  = P[id_col].astype(str)

    # Select latent-factor columns, with a fallback to all non-ID columns.
    factor_cols = [c for c in Xf.columns if c.startswith("factor_")]
    if not factor_cols:
        factor_cols = [c for c in Xf.columns if c != id_col]

    fac_tbl = Xf[[id_col] + factor_cols].drop_duplicates(subset=[id_col])

    # Build the outcome and covariate table, including one-hot categorical covariates.
    needed_cols = [validator] + cov_cont + cov_cat
    missing = [c for c in needed_cols if c not in P.columns]
    if missing:
        raise KeyError(f"Panel is missing required columns: {missing}")

    X_cov = (
        pd.get_dummies(P[cov_cont + cov_cat], columns=cov_cat, drop_first=True)
        if (len(cov_cont) + len(cov_cat)) > 0 else pd.DataFrame(index=P.index)
    )
    panel_tbl = pd.concat([P[[id_col, validator]], X_cov], axis=1)

    # Align factor scores and validator outcomes by subject ID.
    merged = fac_tbl.merge(panel_tbl, on=id_col, how="inner")

    # Apply listwise deletion to all variables used by the model.
    cov_cols = list(X_cov.columns) if not X_cov.empty else []
    drop_cols = [validator] + factor_cols + cov_cols
    merged = merged.dropna(subset=drop_cols, how="any")
    if merged.empty:
        raise ValueError("No valid samples remain after merging and listwise deletion.")

    # Separate the target, full feature matrix, and covariate-only matrix.
    y = merged[validator]
    X_all = merged[factor_cols + cov_cols]
    X_cov_aligned = merged[cov_cols] if cov_cols else pd.DataFrame(index=merged.index)

    # Infer outcome type when it is not supplied.
    if ytype is None:
        uniq = pd.Series(pd.unique(y.dropna()))
        # Numeric two-level outcomes are treated as binary.
        uniq_num = pd.to_numeric(uniq, errors="coerce").dropna().unique()
        if len(uniq_num) <= 2 and set(np.unique(uniq_num)).issubset({0.0, 1.0}):
            ytype = "bin"
        else:
            ytype = "cont"

    # Construct the estimator requested for the outcome type.
    def _get_estimator(model_name: str, _ytype: str, rs: int):
        m = model_name.lower()
        if _ytype == "cont":
            if m == "rf":
                return RandomForestRegressor(
                    n_estimators=600, max_depth=None,
                    random_state=rs, n_jobs=-1)
            elif m == "gbrt":
                return HistGradientBoostingRegressor(random_state=rs)
            elif m == "ols":
                return LinearRegression()
            elif m == "ridge":
                return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-4, 3, 30)))
            elif m == "krbf":
                return KernelRidge(alpha=1.0, kernel="rbf")
            elif m == "svm_rbf":
                return make_pipeline(StandardScaler(), SVR(kernel="rbf"))
            elif m in ("xgb", "xgboost"):
                try:
                    from xgboost import XGBRegressor
                except Exception as e:
                    raise ImportError("xgboost is required for the xgb regressor.") from e
                return XGBRegressor(
                    n_estimators=500, max_depth=6, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                    random_state=rs, n_jobs=-1, tree_method="hist"
                )
            else:
                raise ValueError(f"Unknown regression model: {model_name}")
        else:
            if m == "rf":
                return RandomForestClassifier(n_estimators=800, class_weight="balanced",
                                              random_state=rs, n_jobs=-1)
            elif m == "gbrt":
                return HistGradientBoostingClassifier(random_state=rs)
            elif m == "logit":
                return make_pipeline(StandardScaler(),
                                     LogisticRegression(max_iter=2000, class_weight="balanced"))
            elif m == "svm_rbf":
                return make_pipeline(StandardScaler(),
                                     SVC(kernel="rbf", probability=True, class_weight="balanced",
                                         random_state=rs))
            elif m in ("xgb", "xgboost"):
                try:
                    from xgboost import XGBClassifier
                except Exception as e:
                    raise ImportError("xgboost is required for the xgb classifier.") from e
                return XGBClassifier(
                    n_estimators=700, max_depth=6, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                    random_state=rs, n_jobs=-1, tree_method="hist",
                    eval_metric="auc"
                )
            else:
                raise ValueError(f"Unknown classification model: {model_name}")

    est_full = _get_estimator(model, ytype, random_state)

    # Select standard, stratified, or grouped cross-validation.
    def _safe_k_for_stratified(y_ser: pd.Series, desired: int) -> int:
        # Avoid using more folds than the smallest class can support.
        vc_min = int(y_ser.value_counts().min())
        return max(2, min(desired, vc_min))

    groups = None
    if groups_col and (groups_col in P.columns):
        # Map group labels to the aligned modelling sample.
        grp_map = P[[id_col, groups_col]].drop_duplicates(subset=[id_col])
        grp_map[groups_col] = grp_map[groups_col].astype("category")
        groups = merged[[id_col]].merge(grp_map, on=id_col, how="left")[groups_col]

    if groups is None:
        if ytype == "bin":
            n_splits = _safe_k_for_stratified(y, cv_folds)
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        else:
            n_splits = max(2, min(cv_folds, len(y)))
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    else:
        n_groups = int(groups.nunique())
        n_splits = max(2, min(cv_folds, n_groups))
        if group_strategy.lower() == "logo":
            cv = LeaveOneGroupOut()
        elif ytype == "bin" and _HAS_SGF and n_groups >= 2:
            cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        else:
            cv = GroupKFold(n_splits=n_splits)

    # Evaluate full models and covariate-only baselines by cross-validation.
    if ytype == "cont":
        r2_full  = cross_val_score(est_full, X_all, y, cv=cv, groups=groups, scoring="r2")
        mae_full = -cross_val_score(est_full, X_all, y, cv=cv, groups=groups, scoring="neg_mean_absolute_error")

        if X_cov_aligned.shape[1] > 0:
            est_base = _get_estimator(model, ytype, random_state)
            r2_base  = cross_val_score(est_base, X_cov_aligned, y, cv=cv, groups=groups, scoring="r2")
            mae_base = -cross_val_score(est_base, X_cov_aligned, y, cv=cv, groups=groups, scoring="neg_mean_absolute_error")
            delta = float(np.mean(r2_full) - np.mean(r2_base))
        else:
            r2_base = mae_base = None
            delta = None

        summary = {
            "ytype": ytype,
            "model": model,
            "n_samples": int(len(y)),
            "n_features_full": int(X_all.shape[1]),
            "n_features_cov": int(X_cov_aligned.shape[1]),
            "CV_strategy": f"{type(cv).__name__}" + (f" (groups={groups_col})" if groups is not None else ""),
            "R2_full_mean": float(np.mean(r2_full)), "R2_full_std": float(np.std(r2_full)),
            "MAE_full_mean": float(np.mean(mae_full)), "MAE_full_std": float(np.std(mae_full)),
            "R2_cov_mean": float(np.mean(r2_base)) if r2_base is not None else None,
            "R2_cov_std": float(np.std(r2_base)) if r2_base is not None else None,
            "MAE_cov_mean": float(np.mean(mae_base)) if mae_base is not None else None,
            "MAE_cov_std": float(np.std(mae_base)) if mae_base is not None else None,
            "delta_R2_full_minus_cov": delta,
        }

    else:
        auc_full  = cross_val_score(est_full, X_all, y, cv=cv, groups=groups, scoring="roc_auc")
        bacc_full = cross_val_score(est_full, X_all, y, cv=cv, groups=groups, scoring="balanced_accuracy")

        if X_cov_aligned.shape[1] > 0:
            est_base = _get_estimator(model, ytype, random_state)
            auc_base  = cross_val_score(est_base, X_cov_aligned, y, cv=cv, groups=groups, scoring="roc_auc")
            bacc_base = cross_val_score(est_base, X_cov_aligned, y, cv=cv, groups=groups, scoring="balanced_accuracy")
            delta_auc  = float(np.mean(auc_full) - np.mean(auc_base))
            delta_bacc = float(np.mean(bacc_full) - np.mean(bacc_base))
        else:
            auc_base = bacc_base = None
            delta_auc = delta_bacc = None

        summary = {
            "ytype": ytype,
            "model": model,
            "n_samples": int(len(y)),
            "n_features_full": int(X_all.shape[1]),
            "n_features_cov": int(X_cov_aligned.shape[1]),
            "CV_strategy": f"{type(cv).__name__}" + (f" (groups={groups_col})" if groups is not None else ""),
            "AUC_full_mean": float(np.mean(auc_full)), "AUC_full_std": float(np.std(auc_full)),
            "BACC_full_mean": float(np.mean(bacc_full)), "BACC_full_std": float(np.std(bacc_full)),
            "AUC_cov_mean": float(np.mean(auc_base)) if auc_base is not None else None,
            "AUC_cov_std": float(np.std(auc_base)) if auc_base is not None else None,
            "BACC_cov_mean": float(np.mean(bacc_base)) if bacc_base is not None else None,
            "BACC_cov_std": float(np.std(bacc_base)) if bacc_base is not None else None,
            "delta_AUC_full_minus_cov": delta_auc,
            "delta_BACC_full_minus_cov": delta_bacc,
        }

    # Estimate permutation importance on the aligned sample when requested.
    importance = None
    if compute_importance and X_all.shape[1] > 0:
        est_full.fit(X_all, y)
        scoring = "r2" if ytype == "cont" else "roc_auc"
        imp = permutation_importance(
            est_full, X_all, y, scoring=scoring, n_repeats=5,
            random_state=random_state, n_jobs=-1
        )
        importance = (pd.DataFrame({
            "feature": X_all.columns,
            "import_mean": imp.importances_mean,
            "import_std":  imp.importances_std
        })
        .sort_values("import_mean", ascending=False)
        .reset_index(drop=True))

    return {
        "summary": summary,
        "importance": importance,
        "X_shape": X_all.shape,
        "cov_shape": X_cov_aligned.shape,
        "used_factor_cols": factor_cols,
        "used_cov_cols": cov_cols,
    }

# =============================================================================
# Main workflow
# =============================================================================


def main() -> None:
    """
    Provide a safe command-line entry point for the validation utility module.

    The functions in this file require project-specific data paths and fitted
    factor-score tables. Import this module in an analysis script or notebook and
    call the required functions with explicit inputs.
    """
    available = [
        "build_validators_baseline",
        "build_10_items_validators",
        "build_model_scores",
        "get_scores_by_k",
        "compare_efa_poly_vs_ae_poly",
        "predict_one_validator",
    ]
    print("Available validation utilities:")
    for name in available:
        print(f"- {name}")


if __name__ == "__main__":
    main()
