import itertools
import os
import re
import warnings
from functools import reduce
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.api as sm
import statsmodels.formula.api as smf
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter
from numpy.polynomial.legendre import legval
from sklearn.ensemble import (HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.inspection import permutation_importance
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression, LogisticRegression, RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVC, SVR
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.families.links import logit as logit_link
from statsmodels.genmod.families.links import probit as probit_link
from xgboost import XGBRegressor
from sklearn.metrics import (balanced_accuracy_score, make_scorer,
                            roc_auc_score)
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold, StratifiedGroupKFold


def build_validators_baseline(
    root: Path,
    dict_path: Path,
    validators: Dict[str, List[str]],
    eventname: str = "baseline_year_1_arm_1",
    out_dir: Path = Path("../output"),
    dict_sheet: Optional[str] = None,
    dict_engine: str = "openpyxl",
    verbose: bool = True,
    wide_table_name: str = "validators_baseline.csv"
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

        # align as src_subject_id, eventname 
        df_tag = reduce(lambda l, r: l.merge(r, on=["src_subject_id", "eventname"], how="inner"), frames)
        df_tag = (df_tag.loc[df_tag.eventname == eventname]
                         .drop(columns="eventname"))
        # drop each tag
        out_frames[tag] = df_tag
        if out_dir is not None:
            tag_path = out_dir / f"{tag}_baseline.csv"
            df_tag.to_csv(tag_path, index=False, encoding="utf-8")
            if verbose:
                print(f"  -> save {tag_path.name}  ({df_tag.shape[0]} rows, {df_tag.shape[1]} colums)")
        else:
            if verbose:
                print(f"  -> generated {tag} in memory  ({df_tag.shape[0]} rows, {df_tag.shape[1]} columns)")


    # summary as src_subject_id
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
    validators_csv = str(validators_csv)
    df_val = pd.read_csv(validators_csv).drop_duplicates("src_subject_id").copy()

    def row_mean_min_count(df, cols, min_count):
        sub = df[cols].copy()
        cnt = sub.notna().sum(axis=1)
        m = sub.mean(axis=1)
        m[cnt < min_count] = np.nan
        return m

    def _safe_has(cols: List[str]) -> bool:
        return all(c in df_val.columns for c in cols)

    # dev_delay
    if "dev_delay" not in df_val.columns:
        src = ["devhx_20_p", "devhx_21_p"]
        if _safe_has(src):
            tmp = df_val[src].replace({999: np.nan}).astype(float)
            both_present = tmp.notna().sum(axis=1) >= 2
            dev_delay = tmp.mean(axis=1) * 2
            dev_delay[~both_present] = np.nan
            df_val["dev_delay"] = dev_delay

    # fes_conflict
    if "fes_conflict" not in df_val.columns:
        fes_cols = [f"fes_youth_q{i}" for i in range(1, 10)]
        if _safe_has(fes_cols):
            df_val["fes_conflict"] = row_mean_min_count(df_val, fes_cols, min_count=7) * 9

    # n_friends
    if "n_friends" not in df_val.columns:
        rr = ["resiliency5a_y","resiliency6a_y","resiliency5b_y","resiliency6b_y"]
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

    # school_conn
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

    validator_info = {
        "dev_delay"   : "cont",
        "fes_conflict": "cont",
        "n_friends"   : "cont",
        "school_conn" : "cont",
        "avg_grades"  : "cont",
        "fluid_cog"   : "cont",
        "cryst_cog"   : "cont",
        "mh_service"  : "bin",
        "med_history" : "bin",
        "brought_meds": "bin",
    }
    needed_outcomes = [y for y in validator_info.keys() if y in df_val.columns]
    if not needed_outcomes:
        raise ValueError("No valid validators found in validators CSV.")
    df_val = df_val[["src_subject_id"] + needed_outcomes]

    return df_val

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
    # Extract ID column
    id_series = qns.iloc[:, 0].copy()
    id_series.name = id_colname

    # Create factor DataFrame
    factor_cols = [f"factor_{i+1}" for i in range(latent_factors.shape[1])]
    df_factors = pd.DataFrame(latent_factors, columns=factor_cols)

    # Concatenate ID and factor scores, reset index to avoid mismatch
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
        # Build and fit model
        model = model_fn(k)

        if hasattr(model, "train") and hasattr(model, "evaluate_on_data"):  
            # Autoencoder style
            model.train(show_plot=True)
            latent_factors, _, _, _, _, _ = model.evaluate_on_data(X_scaled)
        else:
            # FactorAnalyzer style
            model.fit(X_scaled)
            latent_factors = model.transform(X_scaled)

        # Build DataFrame
        factor_cols = [f"{prefix}_{i+1}" for i in range(k)]
        df_factors = pd.DataFrame(latent_factors, columns=factor_cols)
        df = pd.concat(
            [subject_id.reset_index(drop=True), df_factors.reset_index(drop=True)],
            axis=1
        )
        scores_by_k[k] = df

    return scores_by_k


warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def compare_efa_poly_vs_ae_poly(
    factors_scores: pd.DataFrame,                  # AE（若未提供 ae_scores_by_k 时使用，包含 factor_* 列）
    efa_scores_by_k: Dict[int, pd.DataFrame],      # 必填：EFA 按 K 的因子分数，形如 {K: df}，df 含 src_subject_id 与 efa_1..efa_K
    validators_csv: Union[str, Path] = "validators_baseline.csv",
    save_dir: str = "efa_vs_model_plots_poly_compare",
    ae_scores_by_k: Optional[Dict[int, pd.DataFrame]] = None,   # 可选：AE 按 K 的因子分数，形如 {K: df}，df 含 src_subject_id 与 factor_1..factor_K
    degree: int = 1,                                # 多项式阶数（1=线性；2/3/4 可选）
    random_seed: int = 6,
    bootstrap_B: int = 30,                          # 自助法重抽样次数（用于最终块 AE vs EFA 的差异 p 值和增益 p 值）
    use_ddof1_zscore: bool = True,                  # ddof=1 标准化（贴近 SPSS）
    model_reg: str = "ols",
    model_clf: str = "logit",
    use_poly_features: bool = True,
):
    """
    精简版 + 方案A：按 K 比较 AE 与 EFA 的外部验证预测力，并显式度量“非线性增益”。
    - 对每个 K：合并 validators + AE_K + EFA_K。
    - 对每个 outcome：
        * 计算线性最终块R²（degree=1）
        * 计算目标阶最终块R²（degree=degree）
        * 非线性增益 ΔR² = R²(degree) - R²(linear)
        * 用自助法分别对 AE/EFA 的“非线性增益”做显著性检验（p 值）
        * 原有 AE vs EFA 最终块 R² 的差异 p 值仍保留
    - 连续结局：OLS in-sample R²；二分类：Logit + Nagelkerke R²。
    - 输出两张图：
        * Total R² / Nagelkerke R² vs K（AE 与 EFA）
        * Nonlinear gain (ΔR²) vs K（AE 与 EFA）
    model_reg: 连续型结局的模型；model_clf: 二分类结局的模型
    use_poly_features: 是否仍然构造多项式特征（非线性树/核方法常可设为 False）
    """

    rng = np.random.default_rng(random_seed)
    np.random.seed(random_seed)
    os.makedirs(save_dir, exist_ok=True)

    # ---------------- 1) Load validators + SPSS-equivalent constructors ----------------
    validator_info = {
        "dev_delay"   : "cont",
        "fes_conflict": "cont",
        "n_friends"   : "cont",
        "school_conn" : "cont",
        "avg_grades"  : "cont",
        "fluid_cog"   : "cont",
        "cryst_cog"   : "cont",
        "mh_service"  : "bin",
        "med_history" : "bin",
        "brought_meds": "bin",
    }
    df_val = build_10_items_validators(validators_csv)

    # ---------------- 2) Helpers ----------------
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

    # Continuous outcome: in-sample OLS R²; Binary: in-sample Logit + Nagelkerke
    # def _total_r2_cont_in(d_all: pd.DataFrame, y: str, cols: List[str]) -> float:
    #     if not cols:
    #         return np.nan
    #     s = d_all[cols].std(numeric_only=True, ddof=1)
    #     used = [c for c in cols if s.get(c, 0.0) > 1e-12]
    #     if not used:
    #         return np.nan
    #     d = _standardize_inplace(d_all.copy(), used)
    #     model = smf.ols(f"{y} ~ " + " + ".join(used), data=d).fit()
    #     return float(model.rsquared)

    # def _total_r2_bin_in(d_all: pd.DataFrame, y: str, cols: List[str]) -> float:
    #     if not cols:
    #         return np.nan
    #     s = d_all[cols].std(numeric_only=True, ddof=1)
    #     used = [c for c in cols if s.get(c, 0.0) > 1e-12]
    #     if not used:
    #         return np.nan
    #     d = _standardize_inplace(d_all.copy(), used)
    #     yv = d[y].astype(float).values
    #     if pd.Series(yv).nunique(dropna=True) < 2:
    #         return np.nan
    #     ll0 = loglik_null_binary(yv)
    #     try:
    #         m = smf.logit(f"{y} ~ " + " + ".join(used), data=d).fit(disp=0, method="newton", maxiter=100)
    #         p_hat = np.asarray(m.predict(d))
    #         ll1 = _bin_loglik_from_pred(yv, p_hat)
    #         return r2_nagelkerke(ll0, ll1, len(d))
    #     except Exception:
    #         return np.nan
    # ====== 3) 把你函数内部这两个 helper 用下面版本“整段替换” =======

    def _total_r2_cont_in(d_all: pd.DataFrame, y: str, cols: List[str]) -> float:
        """连续结局：按 model_reg 计算样本内 Total R²。仍做 ddof=1 标准化。"""
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
                m = KernelRidge(alpha=1.0, kernel="rbf")  # 如需更强拟合可调 alpha/gamma
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
                # XGBoost 
                m = XGBRegressor(
                    n_estimators=400,
                    max_depth=6,          # 典型浅树；可按需调 4~10
                    learning_rate=0.05,   # 学习率；与 n_estimators 联动
                    subsample=0.8,        # 行采样
                    colsample_bytree=0.8, # 列采样
                    reg_lambda=1.0,       # L2 正则
                    random_state=random_seed,
                    n_jobs=-1,
                    tree_method="hist",   # 更快的直方图算法（CPU）
                    # 如果 xgboost>=1.6 且有 GPU，可用：tree_method="gpu_hist"
                    # 或 xgboost>=2.0 可用：device="cuda"
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
        """二分类结局：按 model_clf 计算样本内 Nagelkerke R²（基于预测概率的对数似然）。"""
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

    # >>> NEW: build linear-only blocks (no polynomial terms)
    def _build_linear_blocks(bases: List[str]) -> List[List[str]]:
        # cumulative inclusion of base factors (1..K)
        return [bases[:i] for i in range(1, len(bases)+1)]

    # >>> NEW: bootstrap p-value for "nonlinear gain" within a model (degree - linear)
    def _p_boot_gain(panel_in: pd.DataFrame, y: str, ytype: str,
                     cols_lin: List[str], cols_deg: List[str],
                     B: int, seed: int) -> float:
        # same-sample comparison: ΔR² = R²(deg) - R²(lin)
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

    # ---------------- 3) Prepare factor scores by K ----------------
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

    # ---------------- 4) Main loop per K ----------------
    name_map = {
        "avg_grades"  : "School grades",
        "fluid_cog"   : "Fluid intelligence",
        "cryst_cog"   : "Crystallized intelligence",
        "dev_delay"   : "Developmental delays",
        "fes_conflict": "Family conflict",
        "n_friends"   : "Number of friends",
        "school_conn" : "School connectedness",
        "mh_service"  : "Mental health services",
        "med_history" : "Medical history",
        "brought_meds": "Medication use",
    }

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
                raise ValueError(f"AE（fallback）因子数不足：需要至少 {K} 列 factor_*，实际只有 {len(ae_base_all)}。")
            ae_cols = ae_base_all[:K]
            ae_k = ae_df_base[["src_subject_id"] + ae_cols].copy()

        # EFA_K
        efa_k_raw = efa_scores_by_k[K].drop_duplicates("src_subject_id").copy()
        efa_cols = _sorted_cols(efa_k_raw, "efa_")[:K]
        if len(efa_cols) < K:
            raise ValueError(f"EFA K={K} 的列不足（需要 {K} 列，以 efa_ 前缀命名）。")
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
        # >>> NEW: linear-only blocks (degree=1 backbone)
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

    # ---------------- 5) 汇总 & 导出 ----------------
    res_final = pd.DataFrame(rows_final)
    p_df = pd.DataFrame(p_rows)

    # 汇总表：每个 outcome × K 的 AE/EFA（线性、目标阶）R² 与非线性增益
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

    # ---------------- 6) 绘图：Total R² vs K + Nonlinear gain vs K ----------------
    # (A) total R²
    if not res_final.empty:
        outcomes = [o for o in [
            "mh_service", "med_history", "brought_meds", "fes_conflict",
            "school_conn", "fluid_cog", "cryst_cog", "avg_grades",
            "dev_delay", "n_friends",
        ] if o in res_final["outcome"].unique()]

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

            # significance stars for total R² (final-block AE vs EFA)
            for K in sub["K"].unique():
                p = float(p_map_total.get((outcome, int(K)), np.nan))
                if np.isfinite(p) and p < .05:
                    y_mid = np.nanmean([float(sub.loc[sub["K"]==K, "AE"]), float(sub.loc[sub["K"]==K, "EFA"])])
                    stars = "***" if p < .001 else "**" if p < .01 else "*"
                    ax.annotate(stars, (int(K), y_mid), xytext=(0, 6),
                                textcoords="offset points", ha="center", fontsize=8, weight="bold")

            nice = dict(
                avg_grades="School grades", fluid_cog="Fluid intelligence", cryst_cog="Crystallized intelligence",
                dev_delay="Developmental delays", fes_conflict="Family conflict", n_friends="Number of friends",
                school_conn="School connectedness", mh_service="Mental health services",
                med_history="Medical history", brought_meds="Medication use"
            ).get(outcome, outcome)
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

    # (B) nonlinear gain (ΔR²)
    if not res_final.empty:
        outcomes = [o for o in [
            "mh_service", "med_history", "brought_meds", "fes_conflict",
            "school_conn", "fluid_cog", "cryst_cog", "avg_grades",
            "dev_delay", "n_friends",
        ] if o in res_final["outcome"].unique()]

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

            # significance stars for AE / EFA nonlinear gain respectively
            for K in sub["K"].unique():
                # AE stars
                p_ae  = float(p_map_gain_ae.get((outcome, int(K)), np.nan))
                if np.isfinite(p_ae) and p_ae < .05:
                    y_a = float(sub.loc[sub["K"]==K, "AE_nonlin_gain"])
                    stars = "***" if p_ae < .001 else "**" if p_ae < .01 else "*"
                    ax.annotate(stars, (int(K), y_a), xytext=(0, 6),
                                textcoords="offset points", ha="center", fontsize=8, weight="bold")
                # EFA stars
                p_efa = float(p_map_gain_efa.get((outcome, int(K)), np.nan))
                if np.isfinite(p_efa) and p_efa < .05:
                    y_e = float(sub.loc[sub["K"]==K, "EFA_nonlin_gain"])
                    ax.annotate("o" if p_ae < .05 else "***" if p_efa < .001 else "**" if p_efa < .01 else "*",
                                (int(K), y_e), xytext=(0, -10), textcoords="offset points",
                                ha="center", fontsize=7, weight="bold")

            nice = dict(
                avg_grades="School grades", fluid_cog="Fluid intelligence", cryst_cog="Crystallized intelligence",
                dev_delay="Developmental delays", fes_conflict="Family conflict", n_friends="Number of friends",
                school_conn="School connectedness", mh_service="Mental health services",
                med_history="Medical history", brought_meds="Medication use"
            ).get(outcome, outcome)
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

    # 友好打印（前 20 行）
    print(res_out.head(20))

    return {
        "summary_perK": res_out,     # outcome × K 的 AE/EFA（线性、目标阶）R²、非线性增益、p 值
        "paths": {
            "summary_csv": csv_path,
            "grid_png": png_total,
            "gain_png": png_gain
        }
    }

def predict_one_validator(
    validator: str,
    X: pd.DataFrame,                     # latent factors，可包含或不包含 id_col
    panel: pd.DataFrame,                 # 含 validator + 协变量 +（可含 id_col）
    model: str = "rf",                   # 回归(cont)：rf/gbrt/ols/ridge/krbf/svm_rbf/xgb；分类(bin)：rf/gbrt/logit/svm_rbf/xgb
    ytype: Optional[str] = None,         # "cont" / "bin"；None 时自动判断
    cov_cont: Optional[List[str]] = None,# 连续协变量
    cov_cat: Optional[List[str]] = None, # 分类型协变量（one-hot）
    cv_folds: int = 5,
    random_state: int = 6,
    compute_importance: bool = False,    # 是否计算 permutation importance
    # 新增：用 id 列做对齐（索引不一致也没关系）
    id_col: str = "src_subject_id",
    # 新增：分组 CV（例如 site）
    groups_col: Optional[str] = None,
    group_strategy: str = "logo",        # "logo" / "groupkfold" / "auto"
) -> Dict[str, Any]:
    """
    用指定模型从 latent factors 预测单个 validator，并“正确地”考虑协变量与对齐问题。
    - 先用 id_col 对齐 X 与 panel（inner merge，确保样本交集）。
    - 协变量：分类 one-hot；对比 cov-only 与 full（factors+cov）并报告增量（ΔR²/ΔAUC）。
    - 交叉验证：可选按 groups_col 分组（Leave-One-Group-Out 或 GroupKFold / StratifiedGroupKFold）。
    - 缺失：对参与建模的列做 listwise deletion。
    """
    cov_cont = cov_cont or []
    cov_cat  = cov_cat  or []

    # ---------------- A) 让 X 与 panel 都“显式持有 id_col” ----------------
    Xf = X.copy()
    P  = panel.copy()

    if id_col not in Xf.columns:
        Xf = Xf.reset_index().rename(columns={Xf.index.name or "index": id_col})
    if id_col not in P.columns:
        P = P.reset_index().rename(columns={P.index.name or "index": id_col})

    Xf[id_col] = Xf[id_col].astype(str)
    P[id_col]  = P[id_col].astype(str)

    # ---------------- B) 取 factors 列 ----------------
    # 仅取 factor_*，若没有此前缀则退化为除 id_col 外的所有列
    factor_cols = [c for c in Xf.columns if c.startswith("factor_")]
    if not factor_cols:
        factor_cols = [c for c in Xf.columns if c != id_col]

    fac_tbl = Xf[[id_col] + factor_cols].drop_duplicates(subset=[id_col])

    # ---------------- C) 构造 y 与协变量（one-hot）----------------
    needed_cols = [validator] + cov_cont + cov_cat
    missing = [c for c in needed_cols if c not in P.columns]
    if missing:
        raise KeyError(f"panel 缺少列: {missing}")

    # 分类协变量 one-hot
    X_cov = (
        pd.get_dummies(P[cov_cont + cov_cat], columns=cov_cat, drop_first=True)
        if (len(cov_cont) + len(cov_cat)) > 0 else pd.DataFrame(index=P.index)
    )
    # 面板表（携带 y 与协变量）
    panel_tbl = pd.concat([P[[id_col, validator]], X_cov], axis=1)

    # ---------------- D) 按 id_col 合并，完成“样本对齐” ----------------
    merged = fac_tbl.merge(panel_tbl, on=id_col, how="inner")

    # ---------------- E) 缺失处理：listwise deletion ----------------
    cov_cols = list(X_cov.columns) if not X_cov.empty else []
    drop_cols = [validator] + factor_cols + cov_cols
    merged = merged.dropna(subset=drop_cols, how="any")
    if merged.empty:
        raise ValueError("合并后无有效样本（可能是 id 不重叠或缺失过多）。")

    # y 与特征矩阵
    y = merged[validator]
    X_all = merged[factor_cols + cov_cols]
    X_cov_aligned = merged[cov_cols] if cov_cols else pd.DataFrame(index=merged.index)

    # ---------------- F) 自动判断 ytype ----------------
    if ytype is None:
        uniq = pd.Series(pd.unique(y.dropna()))
        # 尝试转成数值再判断是否 {0,1}
        uniq_num = pd.to_numeric(uniq, errors="coerce").dropna().unique()
        if len(uniq_num) <= 2 and set(np.unique(uniq_num)).issubset({0.0, 1.0}):
            ytype = "bin"
        else:
            ytype = "cont"

    # ---------------- G) 构造估计器 ----------------
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
                    raise ImportError("需要安装 xgboost 才能使用 'xgb' 回归器") from e
                return XGBRegressor(
                    n_estimators=500, max_depth=6, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                    random_state=rs, n_jobs=-1, tree_method="hist"
                )
            else:
                raise ValueError(f"未知回归模型: {model_name}")
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
                    raise ImportError("需要安装 xgboost 才能使用 'xgb' 分类器") from e
                return XGBClassifier(
                    n_estimators=700, max_depth=6, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                    random_state=rs, n_jobs=-1, tree_method="hist",
                    eval_metric="auc"
                )
            else:
                raise ValueError(f"未知分类模型: {model_name}")

    est_full = _get_estimator(model, ytype, random_state)

    # ---------------- H) 选择合适的 CV（含分组） ----------------
    def _safe_k_for_stratified(y_ser: pd.Series, desired: int) -> int:
        # 避免某一类样本不足导致报错
        vc_min = int(y_ser.value_counts().min())
        return max(2, min(desired, vc_min))

    groups = None
    if groups_col and (groups_col in P.columns):
        # 将组标签映射回 merged 的样本顺序
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

    # ---------------- I) 交叉验证评估：cov-only vs full ----------------
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

    # ---------------- J) （可选）Permutation importance ----------------
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
        "importance": importance,      # DataFrame 或 None
        "X_shape": X_all.shape,
        "cov_shape": X_cov_aligned.shape,
        "used_factor_cols": factor_cols,
        "used_cov_cols": cov_cols,
    }
