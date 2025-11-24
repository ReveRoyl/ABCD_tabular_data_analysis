# tune_model.py
import os
import random
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold

import optuna
import argparse

from utils.model_code import (
    set_deterministic,
    _total_evr_from_recon,
    _total_r2_from_recon,
    QuestionnaireDataset,
    decorrelation_loss,
    batch_swap_noise,
)

# ========= Command-line arguments =========
parser = argparse.ArgumentParser()
parser.add_argument(
    "--outer_fold",
    type=int,
    default=0,
    help="Outer fold index to run (1-5). 0 runs all 5 folds (long runtime).",
)
parser.add_argument(
    "--tpe_trials",
    type=int,
    default=50,
    help="Number of TPE trials (default 50; lower it if needed).",
)
parser.add_argument(
    "--cma_trials",
    type=int,
    default=100,
    help="Number of CMA-ES trials (default 100; lower it if needed).",
)
args = parser.parse_args()

# ========= Global seed & environment =========
seed = 520
os.environ["PYTHONHASHSEED"] = str(seed)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

print(">>> Global env + basic seeds set, seed =", seed)

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

device = set_deterministic(seed)

# ========= Paths & data =========
code_dir = Path.cwd()
data_path = code_dir.parent / "data"
data_file = data_path / "cbcl_data_remove_low_frequency.csv"
if not data_file.exists():
    raise FileNotFoundError(f"Could not find {data_file}")

qns = pd.read_csv(data_file, encoding="utf-8")
X = qns.iloc[:, 1:].values

# ======== Hyperparameter ranges (global) ========
H1_MIN, H1_MAX, H1_STEP = 64, 256, 32
H2_MIN, H2_MAX, H2_STEP = 32, 128, 16
LR_MIN, LR_MAX = 1e-4, 5e-3
LDEC_MIN, LDEC_MAX = 0.0, 2.0
WD_MIN, WD_MAX = 1e-6, 1e-3

encoding_dim = 5


class AEModel(nn.Module):
    def __init__(self, input_dim, latent_dim, h1, h2):
        super(AEModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.GELU(),
            nn.Linear(h1, h2),
            nn.GELU(),
            nn.Linear(h2, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, h2),
            nn.GELU(),
            nn.Linear(h2, h1),
            nn.GELU(),
            nn.Linear(h1, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


class AE:
    """
    Wrapper for the autoencoder.
    """

    def __init__(
        self,
        X_train,
        X_val,
        encoding_dim,
        h1=0,
        h2=0,
        *,
        lr,
        seed,
        lambda_decorr: float = 0.0,
        weight_decay: float = 0.0,
    ):
        train_ds = QuestionnaireDataset(X_train)
        val_ds = QuestionnaireDataset(X_val)

        g_loader = torch.Generator().manual_seed(seed)
        self.train_loader = DataLoader(
            train_ds, batch_size=32, shuffle=True, generator=g_loader
        )
        self.val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

        # decorrelation weight and noise generator
        self.lambda_decorr = float(lambda_decorr)
        self.g_noise = torch.Generator().manual_seed(seed + 1)

        input_dim = X_train.shape[1]
        self.model = AEModel(input_dim, encoding_dim, h1, h2).to(device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.1, patience=5
        )

    def train(self, max_epochs=1000, patience=30, show_plot=False):
        best_val_loss = float("inf")
        epochs_no_improve = 0
        train_losses, val_losses = [], []

        for epoch in range(max_epochs):
            # ---------- train ----------
            self.model.train()
            total_train_loss = 0.0
            for batch_x, _ in self.train_loader:
                batch_x = batch_x.to(device)
                x_noisy = batch_swap_noise(
                    batch_x, swap_prob=0.1, generator=self.g_noise
                )
                self.optimizer.zero_grad()
                reconstructed = self.model(x_noisy)
                rec_loss = self.criterion(reconstructed, batch_x)

                latent = self.model.encoder(batch_x)
                if self.lambda_decorr > 0.0:
                    dec_loss = self.lambda_decorr * decorrelation_loss(latent)
                else:
                    dec_loss = 0.0

                loss = rec_loss + dec_loss
                loss.backward()
                self.optimizer.step()
                total_train_loss += loss.item() * batch_x.size(0)

            train_avg = total_train_loss / len(self.train_loader.dataset)
            train_losses.append(train_avg)

            # ---------- val ----------
            self.model.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for batch_x, _ in self.val_loader:
                    batch_x = batch_x.to(device)
                    reconstructed = self.model(batch_x)
                    rec_loss = self.criterion(reconstructed, batch_x)
                    latent = self.model.encoder(batch_x)
                    if self.lambda_decorr > 0.0:
                        dec_loss = self.lambda_decorr * decorrelation_loss(latent)
                    else:
                        dec_loss = 0.0
                    total_val_loss += (rec_loss + dec_loss) * batch_x.size(0)

            val_avg = total_val_loss.item() / len(self.val_loader.dataset)
            val_losses.append(val_avg)
            self.scheduler.step(val_avg)

            if val_avg < best_val_loss:
                best_val_loss = val_avg
                epochs_no_improve = 0
                # If you want to save the best model later, back up state_dict here.
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        if show_plot:
            plt.plot(train_losses, label="Train Loss")
            plt.plot(val_losses, label="Validation Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("AE Loss Curves")
            plt.legend()
            plt.show()

    def evaluate_on_data(self, X):
        self.model.eval()
        X_np = np.asarray(X)
        with torch.no_grad():
            X_t = torch.tensor(X_np, dtype=torch.float32).to(device)
            recon = self.model(X_t)
            latent = self.model.encoder(X_t).cpu().numpy()
            latent_vars = np.var(latent, axis=0, ddof=1)
            total_var = np.var(X_np, axis=0, ddof=1).sum()
            evr = latent_vars / total_var
            recon_np = recon.cpu().numpy()
            rec_errors = np.mean((X_np - recon_np) ** 2, axis=1)
            total_evr = _total_evr_from_recon(X_np, recon_np, ddof=1)
            total_r2 = _total_r2_from_recon(X_np, recon_np)
        return latent, rec_errors, evr, total_evr, recon_np, total_r2


# ========= KFold & Sampler =========
outer_kf = KFold(n_splits=5, shuffle=True, random_state=seed)

tpe_sampler = optuna.samplers.TPESampler(seed=seed)
cma_sampler = optuna.samplers.CmaEsSampler(
    seed=seed,
    sigma0=0.5,
    warn_independent_sampling=False,
)

all_results = []
fold_metadata = []
outer_fold_id = 1


def _clip_int_range(center, lo, hi, step):
    """Given a center value, choose a narrower interval within [lo, hi]."""
    low = max(lo, center - step)
    high = min(hi, center + step)
    if low >= high:
        low = lo
        high = hi
    return low, high


# ========= Result output directory =========
if args.outer_fold in [1, 2, 3, 4, 5]:
    nested_root = code_dir / "output" / "nested_cv_ae_optuna" / f"fold_{args.outer_fold}"
else:
    nested_root = code_dir / "output" / "nested_cv_ae_optuna"

nested_root.mkdir(parents=True, exist_ok=True)


for train_val_idx, test_idx in outer_kf.split(X):
    # If a single outer_fold is specified, run only that fold.
    if args.outer_fold in [1, 2, 3, 4, 5] and outer_fold_id != args.outer_fold:
        outer_fold_id += 1
        continue

    print(f"\n================ Outer Fold {outer_fold_id} ================")

    # ---------- Outer Train/Val/Test split ----------
    X_train_val_raw = X[train_val_idx]
    X_test_raw = X[test_idx]

    scaler = StandardScaler()
    X_train_val = scaler.fit_transform(X_train_val_raw)
    X_test = scaler.transform(X_test_raw)

    # ---------- inner objective ----------
    def make_objective(X_outer_train_val, search_space_mode="wide", base_params=None):
        def objective(trial):
            # 1) Define search space
            if search_space_mode == "wide":
                h1 = trial.suggest_int("h1", H1_MIN, H1_MAX, step=H1_STEP)
                h2 = trial.suggest_int("h2", H2_MIN, H2_MAX, step=H2_STEP)
                lr = trial.suggest_float("lr", LR_MIN, LR_MAX, log=True)
                lambda_decorr = trial.suggest_float("lambda_decorr", LDEC_MIN, LDEC_MAX)
                weight_decay = trial.suggest_float(
                    "weight_decay", WD_MIN, WD_MAX, log=True
                )

            elif search_space_mode == "narrow":
                assert base_params is not None, "base_params cannot be None (narrow mode)"

                h1_center = int(base_params["h1"])
                h2_center = int(base_params["h2"])
                lr_center = float(base_params["lr"])
                ldec_center = float(base_params["lambda_decorr"])
                wd_center = float(base_params["weight_decay"])

                h1_low, h1_high = _clip_int_range(
                    h1_center, H1_MIN, H1_MAX, H1_STEP
                )
                h2_low, h2_high = _clip_int_range(
                    h2_center, H2_MIN, H2_MAX, H2_STEP
                )

                h1 = trial.suggest_int("h1", h1_low, h1_high, step=H1_STEP)
                h2 = trial.suggest_int("h2", h2_low, h2_high, step=H2_STEP)

                lr_low = max(LR_MIN, lr_center / 3.0)
                lr_high = min(LR_MAX, lr_center * 3.0)
                lr = trial.suggest_float("lr", lr_low, lr_high, log=True)

                ldec_low = max(LDEC_MIN, ldec_center - 0.3)
                ldec_high = min(LDEC_MAX, ldec_center + 0.3)
                if ldec_low >= ldec_high:
                    ldec_low, ldec_high = LDEC_MIN, LDEC_MAX
                lambda_decorr = trial.suggest_float(
                    "lambda_decorr", ldec_low, ldec_high
                )

                wd_low = max(WD_MIN, wd_center / 5.0)
                wd_high = min(WD_MAX, wd_center * 5.0)
                weight_decay = trial.suggest_float(
                    "weight_decay", wd_low, wd_high, log=True
                )
            else:
                raise ValueError(f"Unknown search_space_mode: {search_space_mode}")

            # 2) inner KFold
            inner_kf = KFold(n_splits=5, shuffle=True, random_state=seed)
            ev_totals = []

            inner_fold_id = 1
            for inner_train_idx, inner_val_idx in inner_kf.split(X_outer_train_val):
                print(
                    f"-------- Inner Fold {inner_fold_id} ({search_space_mode}) --------"
                )

                X_inner_train = X_outer_train_val[inner_train_idx]
                X_inner_val = X_outer_train_val[inner_val_idx]

                set_deterministic(seed + inner_fold_id)

                ae = AE(
                    X_inner_train,
                    X_inner_val,
                    encoding_dim=encoding_dim,
                    h1=h1,
                    h2=h2,
                    lr=lr,
                    seed=seed + inner_fold_id,
                    lambda_decorr=lambda_decorr,
                    weight_decay=weight_decay,
                )

                print(f"[Inner Fold {inner_fold_id}] Training AE...")
                ae.train(show_plot=False)

                _, _, _, ev_total_val, _, _ = ae.evaluate_on_data(X_inner_val)
                ev_totals.append(ev_total_val)
                print(f"[Inner Fold {inner_fold_id}] EV_total = {ev_total_val:.4f}")

                inner_fold_id += 1

            mean_ev = float(np.mean(ev_totals))
            print(
                f"[Inner Summary | mode={search_space_mode}] Mean EV_total = {mean_ev:.4f}"
            )
            return mean_ev

        return objective

    # ---------- Stage 1: TPE coarse search ----------
    objective_wide = make_objective(X_train_val, search_space_mode="wide")

    study_tpe = optuna.create_study(
        direction="maximize",
        sampler=tpe_sampler,
        study_name=f"ae_outer{outer_fold_id}_tpe",
    )
    study_tpe.optimize(objective_wide, n_trials=args.tpe_trials)

    best_params_tpe = study_tpe.best_params
    best_value_tpe = study_tpe.best_value
    print(f"[Outer Fold {outer_fold_id}] TPE Best Params: {best_params_tpe}")
    print(f"[Outer Fold {outer_fold_id}] TPE Best EV_total = {best_value_tpe:.4f}")

    # ---------- Stage 2: CMA-ES fine search ----------
    objective_narrow = make_objective(
        X_train_val, search_space_mode="narrow", base_params=best_params_tpe
    )

    study_cma = optuna.create_study(
        direction="maximize",
        sampler=cma_sampler,
        study_name=f"ae_outer{outer_fold_id}_cma",
    )
    study_cma.optimize(objective_narrow, n_trials=args.cma_trials)

    best_params_cma = study_cma.best_params
    best_value_cma = study_cma.best_value
    print(f"[Outer Fold {outer_fold_id}] CMA-ES Best Params: {best_params_cma}")
    print(f"[Outer Fold {outer_fold_id}] CMA-ES Best EV_total = {best_value_cma:.4f}")

    # Use the CMA-ES result
    final_best_params = best_params_cma
    final_best_params["stage1_best_ev"] = best_value_tpe
    final_best_params["stage2_best_ev"] = best_value_cma

    best_h1 = final_best_params["h1"]
    best_h2 = final_best_params["h2"]
    best_lr = final_best_params["lr"]
    best_lambda_decorr = final_best_params["lambda_decorr"]
    best_weight_decay = final_best_params["weight_decay"]

    # ---------- Train on outer fold using best hyperparams ----------
    X_train, X_val = train_test_split(
        X_train_val,
        test_size=0.1,  
        random_state=seed,
    )

    autoencoder = AE(
        X_train,
        X_val,
        encoding_dim=encoding_dim,
        h1=best_h1,
        h2=best_h2,
        lr=best_lr,
        seed=seed,
        lambda_decorr=best_lambda_decorr,
        weight_decay=best_weight_decay,
    )

    print(f"[Outer Fold {outer_fold_id}] Training Final AE (with best params)...")
    autoencoder.train(show_plot=False)

    # Save model parameters
    model_path = nested_root / f"outer{outer_fold_id}_ae_model.pth"
    torch.save(autoencoder.model.state_dict(), model_path)
    print(f"[Outer Fold {outer_fold_id}] Model saved to: {model_path}")

    # ---------- Test set evaluation ----------
    latent_factors, rec_errors, evr, ev_total, reconstructed, r2 = autoencoder.evaluate_on_data(
        X_test
    )

    rec_mean = rec_errors.mean()
    print(
        f"[Outer Fold {outer_fold_id}] Final Test: "
        f"EV_total={ev_total:.4f}, RecErr={rec_mean:.4f}, R^2={r2:.4f}"
    )

    all_results.append(
        {
            "fold": outer_fold_id,
            "latent": latent_factors,
            "evr": evr,
            "ev_total": ev_total,
            "reconstructed": reconstructed,
            "r2": r2,
            "rec_error": rec_errors,
        }
    )

    fold_metadata.append(
        {
            "fold": outer_fold_id,
            "ev_total": float(ev_total),
            "rec_error_mean": float(rec_mean),
            "r2": float(r2),
            "best_h1": int(best_h1),
            "best_h2": int(best_h2),
            "best_lr": float(best_lr),
            "best_lambda_decorr": float(best_lambda_decorr),
            "best_weight_decay": float(best_weight_decay),
            "tpe_best_ev": float(best_value_tpe),
            "cma_best_ev": float(best_value_cma),
        }
    )

    # Save trials
    study_tpe.trials_dataframe().to_csv(
        nested_root / f"outer{outer_fold_id}_tpe_trials.csv",
        index=False,
    )
    study_cma.trials_dataframe().to_csv(
        nested_root / f"outer{outer_fold_id}_cma_trials.csv",
        index=False,
    )

    outer_fold_id += 1

# ---------- Final summary ----------
if len(all_results) == 0:
    raise RuntimeError(
        "No outer folds were run; please check that --outer_fold is 1-5 or 0."
    )

ev_list = [r["ev_total"] for r in all_results]
r2_list = [r["r2"] for r in all_results]
rec_list = [r["rec_error"].mean() for r in all_results]

print("\n================ Nested CV Summary ================")
print(f"EV_total mean={np.mean(ev_list):.4f}, std={np.std(ev_list):.4f}")
print(f"RecErr mean={np.mean(rec_list):.4f}, std={np.std(rec_list):.4f}")
print(f"R^2 mean={np.mean(r2_list):.4f}, std={np.std(r2_list):.4f}")

summary = {
    "seed": int(seed),
    "encoding_dim": int(encoding_dim),
    "outer_folds": fold_metadata,
    "summary_metrics": {
        "ev_total_mean": float(np.mean(ev_list)),
        "ev_total_std": float(np.std(ev_list)),
        "rec_err_mean": float(np.mean(rec_list)),
        "rec_err_std": float(np.std(rec_list)),
        "r2_mean": float(np.mean(r2_list)),
        "r2_std": float(np.std(r2_list)),
    },
}

with open(nested_root / "nested_cv_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

pd.DataFrame(fold_metadata).to_csv(
    nested_root / "nested_cv_folds.csv", index=False
)

print(f"Nested CV summary saved to: {nested_root}")
