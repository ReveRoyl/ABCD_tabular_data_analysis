# nested_cv.py

import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import optuna
import sys

sys.path.append('.')
from model_code import AE, SparseAE, VAE, BetaVAE, COAE, FactorVAE


# =======================
# 通用：嵌套 CV + Optuna（使用 TPE）
# =======================

def NestedCVWithOptuna(
    ModelWrapper,            # 签名：ModelWrapper(XTrain, XVal, EncodingDim, **hparams)
    X: np.ndarray,
    EncodingDim: int,
    DefineSearchSpace,       # 函数：DefineSearchSpace(trial) -> dict of hyperparams
    OuterFolds: int = 5,
    InnerFolds: int = 5,
    NTrials: int = 20,
    MaxEpochs: int = 100,
    NJobs: int = 1,          # 并行作业数
    UseHyperband: bool = True  # 默认开启 HyperbandPruner
):
    """
    对“ModelWrapper”做嵌套交叉验证：
      - OuterFolds 折：留一折做最终测试
      - InnerFolds 折：用 Optuna (TPE + HyperbandPruner) 搜索超参
      - EncodingDim 固定不参与搜索
      - 每个内层 trial 在 InnerFolds 上计算平均验证 MSE
      - 并行作业数 NJobs
      - 返回：Outer MSE 列表、每个外折对应的最优超参列表
    """
    outer_cv = KFold(n_splits=OuterFolds, shuffle=True, random_state=42)
    OuterScores = []
    BestParamsList = []

    for foldIdx, (trainIdx, testIdx) in enumerate(outer_cv.split(X)):
        print(f"\n===== Outer Fold {foldIdx+1}/{OuterFolds} =====")
        XTrainOuter, XTestOuter = X[trainIdx], X[testIdx]

        # ---- 定义 Optuna 目标函数 ----
        def Objective(trial):
            inner_cv = KFold(n_splits=InnerFolds, shuffle=True, random_state=foldIdx)
            valMSEs = []

            for innerTrainIdx, innerValIdx in inner_cv.split(XTrainOuter):
                XTrainInner = XTrainOuter[innerTrainIdx]
                XValInner   = XTrainOuter[innerValIdx]

                params = DefineSearchSpace(trial)
                model = ModelWrapper(XTrainInner, XValInner, EncodingDim, **params)
                model.train(max_epochs=MaxEpochs, show_plot=False)

                _, _, _, _, reconVal = model.evaluate_on_data(XValInner)
                mse = mean_squared_error(XValInner, reconVal)
                valMSEs.append(mse)

            return np.mean(valMSEs)

        # ---- 用 TPE + HyperbandPruner 搜索最优超参 ----
        sampler = optuna.samplers.TPESampler(seed=foldIdx, multivariate=True)
        pruner = optuna.pruners.HyperbandPruner() if UseHyperband else optuna.pruners.NopPruner()
        study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
        study.optimize(Objective, n_trials=NTrials, n_jobs=NJobs, show_progress_bar=True)

        bestParams = study.best_trial.params
        BestParamsList.append(bestParams)
        print(f"  >> Best inner params: {bestParams}, inner MSE={study.best_value:.5f}")

        # ---- 用最优超参在整个 XTrainOuter 上重训，并评估在 XTestOuter 上 ----
        finalModel = ModelWrapper(XTrainOuter, XTestOuter, EncodingDim, **bestParams)
        finalModel.train(max_epochs=MaxEpochs, show_plot=False)
        _, _, _, _, reconTest = finalModel.evaluate_on_data(XTestOuter)
        testMSE = mean_squared_error(XTestOuter, reconTest)
        print(f"  >> Outer test MSE = {testMSE:.5f}")
        OuterScores.append(testMSE)

    return OuterScores, BestParamsList


# =======================
# 各模型的 Wrapper + 搜索空间（DefineSearchSpace）
# =======================

# ---------- AE ----------
def SearchAE(trial):
    return {
        "h1": trial.suggest_int("h1", 32, 256),
        "h2": trial.suggest_int("h2", 16, 128),
        "h3": trial.suggest_int("h3", 8, 64),
    }

def WrapAE(XTrain, XVal, EncodingDim, h1, h2, h3):
    return AE(XTrain, XVal, EncodingDim, h1, h2, h3)

def OptimizeAE(X: np.ndarray, EncodingDim: int = 5):
    """
    嵌套 CV + Optuna 搜索 AE 的超参 (不调整 EncodingDim)。
    返回 (outer_test_mses, best_params_per_outer_fold)
    """
    print(">>> Optimize AE (nested CV + TPE) ...")
    return NestedCVWithOptuna(
        ModelWrapper=WrapAE,
        X=X,
        EncodingDim=EncodingDim,
        DefineSearchSpace=SearchAE,
        OuterFolds=5,
        InnerFolds=5,
        NTrials=20,
        MaxEpochs=50
    )


# ---------- SparseAE ----------
def SearchSparseAE(trial):
    return {
        "h1": trial.suggest_int("h1", 32, 256),
        "h2": trial.suggest_int("h2", 16, 128),
        "h3": trial.suggest_int("h3", 8, 64),
        "sparsity_target": trial.suggest_uniform("sparsity_target", 0.01, 0.3),
        "beta": trial.suggest_loguniform("beta", 0.1, 10.0),
    }

def WrapSparseAE(XTrain, XVal, EncodingDim, h1, h2, h3, sparsity_target, beta):
    return SparseAE(XTrain, XVal, EncodingDim, h1, h2, h3, sparsity_target, beta)

def OptimizeSparseAE(X: np.ndarray, EncodingDim: int = 5):
    print(">>> Optimize SparseAE (nested CV + TPE) ...")
    return NestedCVWithOptuna(
        ModelWrapper=WrapSparseAE,
        X=X,
        EncodingDim=EncodingDim,
        DefineSearchSpace=SearchSparseAE,
        OuterFolds=5,
        InnerFolds=5,
        NTrials=20,
        MaxEpochs=50
    )


# ---------- VAE ----------
def SearchVAE(trial):
    return {
        "h1": trial.suggest_int("h1", 32, 256),
        "h2": trial.suggest_int("h2", 16, 128),
        "h3": trial.suggest_int("h3", 8, 64),
        "beta_kl": trial.suggest_loguniform("beta_kl", 0.01, 5.0),
    }

def WrapVAE(XTrain, XVal, EncodingDim, h1, h2, h3, beta_kl):
    return VAE(XTrain, XVal, EncodingDim, h1, h2, h3, beta_kl)

def OptimizeVAE(X: np.ndarray, EncodingDim: int = 5):
    print(">>> Optimize VAE (nested CV + TPE) ...")
    return NestedCVWithOptuna(
        ModelWrapper=WrapVAE,
        X=X,
        EncodingDim=EncodingDim,
        DefineSearchSpace=SearchVAE,
        OuterFolds=5,
        InnerFolds=5,
        NTrials=20,
        MaxEpochs=50
    )


# ---------- BetaVAE ----------
class BetaVAEWrapper(BetaVAE):
    def __init__(self, XTrain, XVal, EncodingDim, h1, h2, h3, beta_max, kl_anneal_epochs, recon_weight):
        super().__init__(XTrain, XVal, EncodingDim, h1, h2, h3)
        self.beta_max = beta_max
        self.kl_anneal_epochs = kl_anneal_epochs
        self.recon_weight = recon_weight

    def train(self, max_epochs=2000, patience=20, show_plot=False):
        super().train(
            max_epochs=max_epochs,
            patience=patience,
            beta_max=self.beta_max,
            kl_anneal_epochs=self.kl_anneal_epochs,
            recon_weight=self.recon_weight,
            show_plot=show_plot
        )

def SearchBetaVAE(trial):
    return {
        "h1": trial.suggest_int("h1", 32, 256),
        "h2": trial.suggest_int("h2", 16, 128),
        "h3": trial.suggest_int("h3", 8, 64),
        "beta_max": trial.suggest_uniform("beta_max", 0.1, 2.0),
        "kl_anneal_epochs": trial.suggest_int("kl_anneal_epochs", 10, 200),
        "recon_weight": trial.suggest_loguniform("recon_weight", 10.0, 500.0),
    }

def WrapBetaVAE(XTrain, XVal, EncodingDim, h1, h2, h3, beta_max, kl_anneal_epochs, recon_weight):
    return BetaVAEWrapper(
        XTrain, XVal, EncodingDim, h1, h2, h3, beta_max, kl_anneal_epochs, recon_weight
    )

def OptimizeBetaVAE(X: np.ndarray, EncodingDim: int = 5):
    print(">>> Optimize BetaVAE (nested CV + TPE) ...")
    return NestedCVWithOptuna(
        ModelWrapper=WrapBetaVAE,
        X=X,
        EncodingDim=EncodingDim,
        DefineSearchSpace=SearchBetaVAE,
        OuterFolds=5,
        InnerFolds=5,
        NTrials=20,
        MaxEpochs=50
    )


# ---------- COAE ----------
class COAEWrapper(COAE):
    def __init__(self, XTrain, XVal, EncodingDim, h1, h2, h3, n_clusters,
                 lambda_orth, mu_clust, lambda_sparsity, sparsity_target):
        super().__init__(XTrain, XVal, EncodingDim, h1, h2, h3, n_clusters,
                         lambda_orth=lambda_orth, mu_clust=mu_clust,
                         lambda_sparsity=lambda_sparsity, sparsity_target=sparsity_target)

    def train(self, max_epochs=200, patience=20, show_plot=False):
        super().train(max_epochs=max_epochs, patience=patience, show_plot=show_plot)

def SearchCOAE(trial):
    return {
        "h1": trial.suggest_int("h1", 32, 256),
        "h2": trial.suggest_int("h2", 16, 128),
        "h3": trial.suggest_int("h3", 8, 64),
        "n_clusters": trial.suggest_int("n_clusters", 2, 20),
        "lambda_orth": trial.suggest_loguniform("lambda_orth", 1e-3, 1.0),
        "mu_clust": trial.suggest_loguniform("mu_clust", 0.1, 5.0),
        "lambda_sparsity": trial.suggest_loguniform("lambda_sparsity", 1e-5, 1e-2),
        "sparsity_target": trial.suggest_uniform("sparsity_target", 0.01, 0.3),
    }

def WrapCOAE(XTrain, XVal, EncodingDim, h1, h2, h3, n_clusters,
             lambda_orth, mu_clust, lambda_sparsity, sparsity_target):
    return COAEWrapper(
        XTrain, XVal, EncodingDim, h1, h2, h3, n_clusters,
        lambda_orth, mu_clust, lambda_sparsity, sparsity_target
    )

def OptimizeCOAE(X: np.ndarray, EncodingDim: int = 5):
    print(">>> Optimize COAE (nested CV + TPE) ...")
    return NestedCVWithOptuna(
        ModelWrapper=WrapCOAE,
        X=X,
        EncodingDim=EncodingDim,
        DefineSearchSpace=SearchCOAE,
        OuterFolds=5,
        InnerFolds=5,
        NTrials=20,
        MaxEpochs=50
    )

# =======================
# FactorVAE 的 Wrapper + 搜索空间 + 调用
# =======================
class FactorVAEWrapper(FactorVAE):
    def __init__(
        self,
        XTrain,
        XVal,
        EncodingDim,
        layer1_neurons,
        layer2_neurons,
        layer3_neurons,
        n_clusters,
        lr_vae,
        lr_d,
        beta_max,
        warmup_epochs,
        verbose=True
    ):
        # 把 verbose 直接传给父类的 Trainer，让它决定是否打印
        super().__init__(
            XTrain,
            XVal,
            EncodingDim,
            layer1_neurons,
            layer2_neurons,
            layer3_neurons,
            n_clusters=n_clusters,
            batch_size=32,
            lr_vae=lr_vae,
            lr_d=lr_d,
            beta_max=beta_max,
            warmup_epochs=warmup_epochs
        )
        # 父类初始化里默认没有 verbose 参数，需要手动赋值给其 trainer
        self.trainer.verbose = verbose

    def train(self, max_epochs=200, show_plot=False):
        # 直接调用父类的 train，这里把 max_epochs 传给 num_epochs
        super().train(num_epochs=max_epochs, show_plot=show_plot)


def SearchFactorVAE(trial):
    return {
        "layer1_neurons": trial.suggest_int("layer1_neurons", 32, 256),
        "layer2_neurons": trial.suggest_int("layer2_neurons", 16, 128),
        "layer3_neurons": trial.suggest_int("layer3_neurons", 8, 64),
        "n_clusters": trial.suggest_int("n_clusters", 2, 20),
        "lr_vae": trial.suggest_float("lr_vae", 1e-4, 1e-2, log=True),
        "lr_d": trial.suggest_float("lr_d", 1e-4, 1e-2, log=True),
        "beta_max": trial.suggest_float("beta_max", 0.1, 2.0),
        "warmup_epochs": trial.suggest_int("warmup_epochs", 5, 50),
    }

def WrapFactorVAE(
    XTrain,
    XVal,
    EncodingDim,
    layer1_neurons,
    layer2_neurons,
    layer3_neurons,
    n_clusters,
    lr_vae,
    lr_d,
    beta_max,
    warmup_epochs
):
    # 这里一定要返回 FactorVAEWrapper，且把 verbose=False 传进去
    return FactorVAEWrapper(
        XTrain,
        XVal,
        EncodingDim,
        layer1_neurons,
        layer2_neurons,
        layer3_neurons,
        n_clusters,
        lr_vae,
        lr_d,
        beta_max,
        warmup_epochs,
        verbose=False  # TPE 内部搜索时不让模型打印 epoch
    )

def OptimizeFactorVAE(X: np.ndarray, EncodingDim: int = 5):
    print(">>> Optimize FactorVAE (nested CV + TPE) ...")
    return NestedCVWithOptuna(
        ModelWrapper=WrapFactorVAE,
        X=X,
        EncodingDim=EncodingDim,
        DefineSearchSpace=SearchFactorVAE,
        OuterFolds=5,
        InnerFolds=5,
        NTrials=20,
        MaxEpochs=50
        # 这里 show_progress_bar 默认就 True，不需要改
    )