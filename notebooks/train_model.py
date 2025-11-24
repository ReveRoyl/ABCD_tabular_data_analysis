# ==== Step 0: 全局环境 & 随机种子（最顶层 cell）====
import argparse
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=8)
parser.add_argument("--n_components", type=int, default=5)
args = parser.parse_args()
n_components = args.n_components
seed = args.seed

print(f"Using seed = {seed}")

# 先设环境变量，再 import 任何 torch 相关的东西
os.environ["PYTHONHASHSEED"] = str(seed)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # CUDA 确定性
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import random
import numpy as np
import torch

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

print(">>> Global env + basic seeds set, seed =", seed)

from pathlib import Path
from functools import reduce
from scipy.linalg import orthogonal_procrustes

from validators import (
    build_validators_baseline,
    build_model_scores,
    get_scores_by_k,
    compare_efa_poly_vs_ae_poly,
    build_10_items_validators,
)

from utils import (
    compute_autoencoder_loadings_with_plot,
    check_reconstruction,
    translate_text,
    get_cbcl_details,
    compute_shap_loadings_decoder_only,  # 你原来有用
)

import copy
import shap
import matplotlib.pyplot as plt
from factor_analyzer import FactorAnalyzer
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import NMF
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, KFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from torch.utils.data import DataLoader, Dataset
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F

import optuna
import io
from PIL import Image
import netron

from model_code import AE, set_deterministic   # 注意，这里在 torch 后面没问题
from IPython.display import Image as IPyImage

import pandas as pd

# 这里再正式设一次（会更新 model_code 里的全局 device）
device = set_deterministic(seed)
code_dir = Path(os.getcwd())
data_path = code_dir.parent / "data"
assert os.path.exists(
    data_path
), "Data directory not found. Make sure you're running this code from the root directory of the project."

with open(data_path / "cbcl_data_remove_low_frequency.csv", "r", encoding="utf-8") as f:
    qns = pd.read_csv(f)

X = qns.iloc[:, 1:].values

# Standardize the data
scaler = StandardScaler()
# scaler = MinMaxScaler()
# X_scaled = scaler.fit_transform(X)

items = [get_cbcl_details(col) for col in qns.iloc[:, 1:].columns]
items = np.array(items)


def _to_2d(a):
    arr = np.asarray(a)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def _standardize(X, ddof=1):
    X = np.asarray(X, dtype=float)
    mean = X.mean(axis=0)
    std = X.std(axis=0, ddof=ddof)
    std[std == 0] = 1.0
    return (X - mean) / std


def _corr_matrix(A, B):
    """
    Column-wise correlation matrix corr(A[:, i], B[:, j])
    Returns shape (A_dim, B_dim)
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


def run_cca_interpret(
    Z,
    X,
    n_components=n_components,
    impute=False,
    standardize=True,
    random_state=seed,
    n_perm=0,
    top_k=10,
    item_names_X=None,
    factor_names_Z=None,
):
    Z = _to_2d(Z)
    X = _to_2d(X)

    if standardize:
        Xz = _standardize(X, ddof=1)
        Zz = _standardize(Z, ddof=1)
    else:
        Xz, Zz = X.astype(float), Z.astype(float)

    n, p = Xz.shape
    _, k = Zz.shape
    n_components = int(min(n_components, p, k))

    cca = CCA(n_components=n_components, max_iter=5000, scale=False)
    cca.fit(Xz, Zz)
    U, V = cca.transform(Xz, Zz)

    can_corr = np.array([np.corrcoef(U[:, i], V[:, i])[0, 1] for i in range(n_components)])

    if item_names_X is None:
        item_names_X = [f"X{j}" for j in range(p)]
    if factor_names_Z is None:
        factor_names_Z = [f"z{m}" for m in range(k)]

    X_loadings = _corr_matrix(Xz, U)
    Z_loadings = _corr_matrix(Zz, V)

    X_top_items = {}
    for i in range(n_components):
        contrib = pd.Series(np.abs(X_loadings[:, i]), index=item_names_X)
        X_top_items[i] = contrib.sort_values(ascending=False).head(top_k)

    p_values = None
    if n_perm and n_perm > 0:
        rng = np.random.default_rng(random_state)
        perm_corrs = np.zeros((n_perm, n_components), dtype=float)
        for b in range(n_perm):
            perm_idx = rng.permutation(n)
            Z_perm = Zz[perm_idx, :]
            cca_b = CCA(n_components=n_components, max_iter=2000, scale=False)
            cca_b.fit(Xz, Z_perm)
            U_b, V_b = cca_b.transform(Xz, Z_perm)
            perm_corrs[b, :] = [np.corrcoef(U_b[:, i], V_b[:, i])[0, 1] for i in range(n_components)]
        p_values = np.mean(perm_corrs >= can_corr[None, :], axis=0)

    results = {
        "can_corr": can_corr,
        "U": U,
        "V": V,
        "X_loadings": pd.DataFrame(
            X_loadings,
            index=item_names_X,
            columns=[f"CC{i+1}" for i in range(n_components)],
        ),
        "Z_loadings": pd.DataFrame(
            Z_loadings,
            index=factor_names_Z,
            columns=[f"CC{i+1}" for i in range(n_components)],
        ),
        "X_top_items": X_top_items,
        "p_values": p_values,
    }
    return results


def run_folds_cv(X, items, seed=seed, artifact_root=None, cca_dataset=None):

    if artifact_root is not None:
        artifact_root = Path(artifact_root)
        models_dir = artifact_root / "models"
        results_dir = artifact_root / "results"
        models_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)
    else:
        models_dir = None
        results_dir = None

    cca_data = cca_dataset if cca_dataset is not None else X
    if items is None:
        item_names_for_cca = None
    elif isinstance(items, np.ndarray):
        item_names_for_cca = items.tolist()
    else:
        item_names_for_cca = list(items)

    kf = KFold(n_splits=5, shuffle=True, random_state=seed)

    all_results = []
    fold_metadata = []
    model_config = {
        "encoding_dim": n_components,
        "h1": 192,
        "h2": 64,
        "seed": seed,
        "lr": 0.003,
        "lambda_decorr": 0.005,
        "weight_decay": 8e-05,
    }

    fold_id = 1
    for train_val_idx, test_idx in kf.split(X):

        print(f"\n================ Fold {fold_id} ================")

        X_train_val_raw = X[train_val_idx]
        X_test_raw = X[test_idx]

        # ---- Standardize: fit on train_val only ----
        scaler = StandardScaler()
        X_train_val = scaler.fit_transform(X_train_val_raw)
        X_test = scaler.transform(X_test_raw)

        # ---- Then split train_val into train/val (same ratio你现在是 80/10/10，我们保持一样) ----
        X_train, X_val = train_test_split(
            X_train_val, 
            test_size=0.10,   # 0.1 / (0.9) ≈ 0.11
            random_state=seed
        )

        # ---- Train AE ----
        autoencoder = AE(X_train, X_val, **model_config)
        autoencoder.train(show_plot=False)

        latent_factors, rec_errors, evr, ev_total, reconstructed, r2 = \
            autoencoder.evaluate_on_data(X_test)

        rec_mean = rec_errors.mean()
        print(f"Fold {fold_id} | EV_total={ev_total:.4f}, RecErr={rec_mean:.4f}, R²={r2:.4f}")

        latent_full = full_rec_errors = full_evr = full_ev_total = full_recon = full_r2 = None
        if cca_data is not None:
            latent_full, full_rec_errors, full_evr, full_ev_total, full_recon, full_r2 = \
                autoencoder.evaluate_on_data(cca_data)

        model_artifact = None
        if models_dir is not None:
            model_artifact = models_dir / f"seed_{seed}_fold_{fold_id}.pt"
            torch.save(
                {
                    "seed": seed,
                    "fold": fold_id,
                    "model_state_dict": autoencoder.model.state_dict(),
                    "model_config": {**model_config, "input_dim": X_train.shape[1]},
                },
                model_artifact,
            )

        result_artifact = None
        if results_dir is not None:
            result_artifact = results_dir / f"seed_{seed}_fold_{fold_id}.npz"
            np.savez_compressed(
                result_artifact,
                latent=latent_factors,
                rec_errors=rec_errors,
                evr=evr,
                ev_total=np.array(ev_total),
                r2=np.array(r2),
                reconstructed=reconstructed,
            )

        all_results.append({
            "fold": fold_id,
            "latent": latent_factors,
            "latent_full": latent_full,
            "rec_error": rec_errors,
            "full_rec_error": full_rec_errors,
            "ev_total": ev_total,
            "full_ev_total": full_ev_total,
            "r2": r2,
            "full_r2": full_r2,
            "reconstructed": reconstructed,
            "full_reconstructed": full_recon,
            "model_artifact": str(model_artifact) if model_artifact else None,
            "results_artifact": str(result_artifact) if result_artifact else None,
            "full_evr": full_evr,
        })

        fold_metadata.append({
            "fold": fold_id,
            "ev_total": float(ev_total),
            "rec_error_mean": float(rec_mean),
            "r2": float(r2),
            **({"model_path": str(model_artifact)} if model_artifact else {}),
            **({"results_path": str(result_artifact)} if result_artifact else {}),
        })

        fold_id += 1

    # ---- 汇总结果 ----
    ev_list = [r["ev_total"] for r in all_results]
    r2_list = [r["r2"] for r in all_results]
    rec_list = [r["rec_error"].mean() for r in all_results]

    print("\n================ CV Summary ================")
    print(f"Explained Variance Total: mean={np.mean(ev_list):.4f}, std={np.std(ev_list):.4f}")
    print(f"Reconstruction Error:     mean={np.mean(rec_list):.4f}, std={np.std(rec_list):.4f}")
    print(f"R²:                       mean={np.mean(r2_list):.4f}, std={np.std(r2_list):.4f}")

    median_rec = float(np.median(rec_list))
    diffs_to_median = [abs(m - median_rec) for m in rec_list]
    best_idx = int(np.argmin(diffs_to_median))
    best_fold_info = all_results[best_idx]
    print(
        f"Selected fold {best_fold_info['fold']} for CCA (rec_error_mean={rec_list[best_idx]:.4f}, "
        f"median target={median_rec:.4f})"
    )

    cca_payload = None
    cca_artifact_path = None
    if best_fold_info.get("latent_full") is not None and cca_data is not None:
        factor_names = [f"z{i+1}" for i in range(best_fold_info["latent_full"].shape[1])]
        cca_result = run_cca_interpret(
            Z=best_fold_info["latent_full"],
            X=cca_data,
            n_components=min(5, best_fold_info["latent_full"].shape[1], cca_data.shape[1]),
            item_names_X=item_names_for_cca,
            factor_names_Z=factor_names,
            random_state=seed,
            top_k=10,
        )

        def _df_to_dict(df):
            return {
                "index": list(df.index),
                "columns": list(df.columns),
                "data": [[float(v) for v in row] for row in df.values],
            }

        top_items_serialized = {
            f"CC{i+1}": [
                {"item": idx, "abs_loading": float(val)}
                for idx, val in series.items()
            ]
            for i, series in cca_result["X_top_items"].items()
        }

        cca_payload = {
            "fold": best_fold_info["fold"],
            "canonical_correlations": cca_result["can_corr"].tolist(),
            "X_loadings": _df_to_dict(cca_result["X_loadings"]),
            "Z_loadings": _df_to_dict(cca_result["Z_loadings"]),
            "top_items": top_items_serialized,
            "p_values": (
                cca_result["p_values"].tolist()
                if cca_result["p_values"] is not None
                else None
            ),
        }

        if artifact_root is not None:
            cca_artifact_path = artifact_root / f"cca_seed_{seed}.json"
            with open(cca_artifact_path, "w", encoding="utf-8") as f:
                json.dump(cca_payload, f, ensure_ascii=False, indent=2)
            print(f"CCA summary saved to {cca_artifact_path}")

    summary = {
        "seed": seed,
        "cv_summary": {
            "explained_variance_mean": float(np.mean(ev_list)),
            "explained_variance_std": float(np.std(ev_list)),
            "reconstruction_error_mean": float(np.mean(rec_list)),
            "reconstruction_error_std": float(np.std(rec_list)),
            "r2_mean": float(np.mean(r2_list)),
            "r2_std": float(np.std(r2_list)),
        },
        "folds": fold_metadata,
        "median_fold": {
            "target_metric": "rec_error_mean",
            "median_value": median_rec,
            "selected_fold": int(best_fold_info["fold"]),
            "selected_fold_metrics": {
                "rec_error_mean": float(rec_list[best_idx]),
                "ev_total": float(best_fold_info["ev_total"]),
                "r2": float(best_fold_info["r2"]),
            },
        },
        "cca": cca_payload,
    }

    if artifact_root is not None:
        summary_path = artifact_root / f"summary_seed_{seed}.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    return all_results, summary


artifact_root = code_dir.parent / "output" / f"seed_runs_dim{n_components}" / f"seed_{seed}"
results, summary = run_folds_cv(
    X,
    items,
    seed=seed,
    artifact_root=artifact_root,
    cca_dataset=X,
)
print(f"Saved models and fold results with dim {n_components} for seed {seed} in {artifact_root}")