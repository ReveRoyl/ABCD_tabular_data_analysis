"""
AE Latent Factors ROI Explanation Pipeline (for CBCL/KSADS & brain FC)
=====================================================================

功能概览
--------
1) 对每个潜在维度 (factor_1..factor_K) 进行 1D GMM 拟合（BIC 选择成分数，BIC 并列时用 silhouette 评分作优先），
   依据相邻高斯成分的后验相等点作为边界，得到**互不重叠**的 ROI 区间 [LT, UT]。
2) 计算各潜在维度的一维分布之间的**对称 KL 散度** (DKL_S = KL(P||Q)+KL(Q||P))，基于该距离做层次聚类 (AHC, complete linkage)。
   簇数 Kc 通过最大化 silhouette score (基于 DKL_S 转换距离矩阵) 自动选择。
3) 在每个簇内，组合各维度的**主成分 ROI**（默认每维最多取权重 top_m 个成分），交集形成候选“组别”。
   过滤样本数过少的组别，输出每个“组”的成员、占比等。
4) 与外部 validators 绑定：连续变量给出均值/效应量 (Cohen's d)，二值变量给出比值比/相对风险，并做**控制协变量**后的 OLS / Logit 事后回归检验。
5) 可在多个 AE 配置（如 d=10/20/30、不同噪声）之间跑稳定性评估（ARI/Jaccard）。

使用方式（示例）
----------------
from roi_pipeline import run_roi_pipeline, auto_detect_validators

# factors_df: 必须包含 'src_subject_id' 与 'factor_1'..'factor_K'
# validators_df: 必须包含 'src_subject_id' 与外部验证器列及协变量

res = run_roi_pipeline(
    factors_df=factors_df,
    validators_df=validators_df,
    save_dir="roi_outputs",
    factor_prefix="factor_",
    subject_col="src_subject_id",
    # validators 设置（可自动探测，也可手动指定）
    continuous_validators=["cbcl_total","gpa","n_friends","fes_conflict"],
    binary_validators=["ksads_dx_mdd","ksads_dx_anx"],
    covariates=["age","sex","site","field_strength"],
    top_m=2,
    min_group_size=100,
    random_state=7,
)

# 结果对象 res 中包含：
# - rois_df: 每一维每个高斯成分的 ROI 定义（LT/UT/weight/mean/std）
# - dkls_matrix: 对称 KL 距离矩阵 (numpy.ndarray)
# - clusters: dict[int, List[str]] -> 聚类簇到维度名的映射
# - groups_df: 每个 ROI 组合组别的规模、占比等
# - membership: 每个样本所属的组（可多选）与 one-hot 矩阵
# - validators_summary: 分组的描述性统计（连续/二值）
# - regression_results: 事后 OLS/Logit 的回归系数与 p 值（协变量已控制）

依赖
----
- numpy, pandas, scipy, scikit-learn, statsmodels, matplotlib (仅保存简单图)

"""
from __future__ import annotations
import os
import math
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Iterable

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, adjusted_rand_score, jaccard_score
from sklearn.cluster import AgglomerativeClustering
import statsmodels.api as sm
import matplotlib.pyplot as plt

# -------------------------
# 工具函数
# -------------------------

def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def _is_binary_series(s: pd.Series, tol_unique: int = 3) -> bool:
    vals = s.dropna().unique()
    return len(vals) <= tol_unique and set(np.unique(vals)).issubset({0, 1})


def _minmax(x: np.ndarray) -> Tuple[float, float]:
    return float(np.nanmin(x)), float(np.nanmax(x))


def _grid_from_data(x: np.ndarray, n: int = 512, pad: float = 0.05) -> np.ndarray:
    lo, hi = _minmax(x)
    rng = hi - lo
    if rng <= 0:
        lo, hi = lo - 1.0, hi + 1.0
        rng = 2.0
    lo -= pad * rng
    hi += pad * rng
    return np.linspace(lo, hi, n)


# -------------------------
# 1D GMM with BIC & Silhouette tie-break, ROI boundaries
# -------------------------
@dataclass
class ROIComponent:
    dim: str
    k: int              # component index (0..Kc-1)
    weight: float
    mean: float
    std: float
    LT: float           # lower threshold (boundary)
    UT: float           # upper threshold (boundary)


def fit_gmm_1d_with_bic(x: np.ndarray, max_components: int = 6, random_state: int = 7) -> GaussianMixture:
    x = np.asarray(x).reshape(-1, 1)
    finite_mask = np.isfinite(x.ravel())
    x = x[finite_mask]
    if len(x) < 10:
        # fallback: 1 component
        gm = GaussianMixture(n_components=1, random_state=random_state)
        gm.fit(x)
        return gm

    models = []
    bics = []
    sils = []
    for n in range(1, max_components + 1):
        gm = GaussianMixture(n_components=n, covariance_type='full', random_state=random_state)
        gm.fit(x)
        labels = gm.predict(x)
        # silhouette on 1D euclidean space
        sil = -np.inf
        if n > 1 and len(np.unique(labels)) > 1:
            try:
                sil = silhouette_score(x, labels, metric='euclidean')
            except Exception:
                sil = -np.inf
        models.append(gm)
        bics.append(gm.bic(x))
        sils.append(sil)

    bics = np.array(bics)
    sils = np.array(sils)
    # pick min BIC; if ties within epsilon, pick highest silhouette
    eps = 1e-6
    idx_min = np.where(np.isclose(bics, bics.min(), rtol=0, atol=eps))[0]
    if len(idx_min) == 1:
        best = idx_min[0]
    else:
        # choose with max silhouette among tied
        best = idx_min[np.argmax(sils[idx_min])]
    return models[best]


def _equal_posterior_boundary_1d(weights, means, stds, i, j) -> Optional[float]:
    """Solve for x where wi*N(x|mi,si^2) == wj*N(x|mj,sj^2) in 1D.
    Return a boundary near between means; if none, return midpoint.
    """
    wi, wj = weights[i], weights[j]
    mi, mj = means[i], means[j]
    si, sj = stds[i], stds[j]
    # Solve: log(wi) - 0.5*log(2π) - log(si) - (x-mi)^2/(2si^2) = log(wj) - log(sj) - (x-mj)^2/(2sj^2)
    # Rearrange to ax^2 + bx + c = 0
    ai = 1.0/(2*si*si)
    aj = 1.0/(2*sj*sj)
    a = -ai + aj
    b = 2*mi*ai - 2*mj*aj
    c = - (mi*mi)*ai + (mj*mj)*aj + math.log((wj/sj)/(wi/si))
    xs: List[float] = []
    if abs(a) < 1e-12:
        # linear
        if abs(b) < 1e-12:
            return float(mi + mj)/2.0
        x = -c/b
        xs = [x]
    else:
        disc = b*b - 4*a*c
        if disc < 0:
            return float(mi + mj)/2.0
        sqrt_disc = math.sqrt(disc)
        x1 = (-b + sqrt_disc)/(2*a)
        x2 = (-b - sqrt_disc)/(2*a)
        xs = [x1, x2]
    # choose the root between means if exists; else choose the closest to segment [mi, mj]
    lo, hi = sorted([mi, mj])
    inside = [x for x in xs if lo <= x <= hi]
    if inside:
        # for robustness, pick the one closest to midpoint
        mid = 0.5*(mi + mj)
        return min(inside, key=lambda z: abs(z - mid))
    # fallback: choose the root closest to [lo,hi]
    return min(xs, key=lambda z: 0 if lo <= z <= hi else min(abs(z - lo), abs(z - hi)))


def derive_roi_intervals_from_gmm(gm: GaussianMixture, dim_name: str) -> List[ROIComponent]:
    means = gm.means_.ravel()
    stds = np.sqrt(gm.covariances_.ravel()) if gm.covariance_type == 'full' else np.sqrt(gm.covariances_)
    weights = gm.weights_.ravel()
    order = np.argsort(means)
    means, stds, weights = means[order], stds[order], weights[order]

    # boundaries between adjacent components where posteriors are equal
    bounds = [-np.inf]
    for a, b in zip(range(len(means)-1), range(1, len(means))):
        xb = _equal_posterior_boundary_1d(weights, means, stds, a, b)
        bounds.append(float(xb))
    bounds.append(np.inf)

    rois: List[ROIComponent] = []
    for k in range(len(means)):
        rois.append(ROIComponent(
            dim=dim_name,
            k=k,
            weight=float(weights[k]),
            mean=float(means[k]),
            std=float(stds[k]),
            LT=float(bounds[k]),
            UT=float(bounds[k+1])
        ))
    return rois


# -------------------------
# Symmetric KL distance between 1D marginals
# -------------------------

def pdf_from_gmm_1d(gm: GaussianMixture, grid: np.ndarray) -> np.ndarray:
    grid = grid.reshape(-1, 1)
    # sklearn's score_samples returns log p(x)
    lp = gm.score_samples(grid)
    p = np.exp(lp)
    # normalize numerically
    dx = (grid[1] - grid[0])[0]
    s = np.trapz(p, dx=float(dx))
    if s <= 0:
        return p
    return p / s


def symmetric_kl_1d(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = p + eps
    q = q + eps
    p = p / p.sum()
    q = q / q.sum()
    d1 = np.sum(p * np.log(p / q))
    d2 = np.sum(q * np.log(q / p))
    return float(d1 + d2)


def dkls_matrix_for_dimensions(factors_df: pd.DataFrame, factor_cols: List[str], random_state: int = 7) -> Tuple[np.ndarray, Dict[str, GaussianMixture], Dict[str, List[ROIComponent]]]:
    gmms: Dict[str, GaussianMixture] = {}
    rois: Dict[str, List[ROIComponent]] = {}
    grids: Dict[str, np.ndarray] = {}
    pdfs: Dict[str, np.ndarray] = {}

    for c in factor_cols:
        x = factors_df[c].values
        gm = fit_gmm_1d_with_bic(x, random_state=random_state)
        gmms[c] = gm
        rois[c] = derive_roi_intervals_from_gmm(gm, c)
        grids[c] = _grid_from_data(x)
        pdfs[c] = pdf_from_gmm_1d(gm, grids[c])

    K = len(factor_cols)
    D = np.zeros((K, K), dtype=float)
    for i, ci in enumerate(factor_cols):
        for j, cj in enumerate(factor_cols):
            if j <= i:
                continue
            # interpolate to common grid (use finer of two)
            g = grids[ci] if len(grids[ci]) >= len(grids[cj]) else grids[cj]
            pi = np.interp(g, grids[ci], pdfs[ci])
            pj = np.interp(g, grids[cj], pdfs[cj])
            D[i, j] = D[j, i] = symmetric_kl_1d(pi, pj)
    return D, gmms, rois


# -------------------------
# Clustering dimensions using DKLS distance
# -------------------------

def _agglomerative_on_distance(D: np.ndarray, labels: List[str], random_state: int = 7) -> Dict[int, List[str]]:
    # Choose cluster count by maximizing silhouette on distances
    K = len(labels)
    # Convert distance to similarity for silhouette? sklearn silhouette accepts distances with metric='precomputed'.
    best_k, best_s, best_assign = None, -np.inf, None
    # ensure symmetry and zero diagonal
    D = (D + D.T) / 2.0
    np.fill_diagonal(D, 0.0)
    for k in range(2, min(K, 8) + 1):
        try:
            try:
                ac = AgglomerativeClustering(n_clusters=k, linkage='complete', metric='precomputed')
            except TypeError:
                # older sklearn uses affinity
                ac = AgglomerativeClustering(n_clusters=k, linkage='complete', affinity='precomputed')
            lab = ac.fit_predict(D)
            s = silhouette_score(D, lab, metric='precomputed')
            if s > best_s:
                best_s, best_k, best_assign = s, k, lab
        except Exception:
            continue
    if best_assign is None:
        # fallback: k=2
        try:
            ac = AgglomerativeClustering(n_clusters=2, linkage='complete', metric='precomputed')
        except TypeError:
            ac = AgglomerativeClustering(n_clusters=2, linkage='complete', affinity='precomputed')
        best_assign = ac.fit_predict(D)
    clusters: Dict[int, List[str]] = {}
    for idx, lab in enumerate(best_assign):
        clusters.setdefault(int(lab), []).append(labels[idx])
    return clusters


# -------------------------
# Compose ROI combos inside each cluster
# -------------------------
@dataclass
class ROICombo:
    cluster_id: int
    dims: List[str]
    combo: Dict[str, Tuple[float, float]]  # dim -> (LT, UT)
    size: int
    proportion: float


def _top_m_components(rois_per_dim: List[ROIComponent], m: int) -> List[ROIComponent]:
    rois_sorted = sorted(rois_per_dim, key=lambda r: r.weight, reverse=True)
    return rois_sorted[:max(1, m)]


def build_roi_combos(
    factors_df: pd.DataFrame,
    clusters: Dict[int, List[str]],
    rois: Dict[str, List[ROIComponent]],
    top_m: int = 2,
    min_group_size: int = 100,
) -> Tuple[List[ROICombo], pd.DataFrame, pd.DataFrame]:
    """
    返回：
      - combos: List[ROICombo]
      - membership_long: (subject_id, cluster_id, combo_key, in_group)
      - membership_wide: one-hot DataFrame (rows: subject, cols: combo_key)
    """
    subject_col = 'src_subject_id' if 'src_subject_id' in factors_df.columns else factors_df.columns[0]
    N = len(factors_df)

    combos: List[ROICombo] = []
    membership_records = []

    def _combo_key(cluster_id: int, dims: List[str], intervals: Dict[str, Tuple[float, float]]):
        parts = [f"{d}[{intervals[d][0]:.4g},{intervals[d][1]:.4g}]" for d in dims]
        return f"C{cluster_id}|" + "&".join(parts)

    for cid, dims in clusters.items():
        if len(dims) == 0:
            continue
        # collect top-m ROIs per dim
        top_rois_list: List[List[ROIComponent]] = [
            _top_m_components(rois[d], top_m) for d in dims
        ]
        # cartesian product of ROI choices across dims
        import itertools
        for choices in itertools.product(*top_rois_list):
            intervals = {r.dim: (r.LT, r.UT) for r in choices}
            dims_order = list(intervals.keys())
            # compute membership
            mask = np.ones(N, dtype=bool)
            for d in dims_order:
                LT, UT = intervals[d]
                x = factors_df[d].values
                mask &= (x >= LT) & (x < UT)
            size = int(mask.sum())
            if size < min_group_size:
                continue
            prop = size / N
            combos.append(ROICombo(cluster_id=int(cid), dims=dims_order, combo=intervals, size=size, proportion=prop))
            key = _combo_key(int(cid), dims_order, intervals)
            for sid, in_g in zip(factors_df[subject_col].values, mask):
                membership_records.append((sid, int(cid), key, bool(in_g)))

    membership_long = pd.DataFrame(membership_records, columns=["src_subject_id", "cluster_id", "combo_key", "in_group"]) if membership_records else pd.DataFrame(columns=["src_subject_id","cluster_id","combo_key","in_group"])
    if not membership_long.empty:
        membership_wide = membership_long.pivot_table(index="src_subject_id", columns="combo_key", values="in_group", fill_value=False, aggfunc='max')
        membership_wide = membership_wide.astype(bool)
    else:
        membership_wide = pd.DataFrame(index=factors_df[subject_col].values)

    return combos, membership_long, membership_wide


# -------------------------
# Validators: descriptive stats & post-hoc regression
# -------------------------
@dataclass
class ValidatorResult:
    combo_key: str
    outcome: str
    type: str  # 'continuous' or 'binary'
    n_in: int
    n_out: int
    mean_in: Optional[float]
    mean_out: Optional[float]
    effect_size: Optional[float]  # Cohen's d or log(OR)
    p_value: Optional[float]


def auto_detect_validators(validators_df: pd.DataFrame, exclude: Iterable[str]) -> Tuple[List[str], List[str]]:
    cont_cols, bin_cols = [], []
    for c in validators_df.columns:
        if c in exclude:
            continue
        s = validators_df[c]
        if pd.api.types.is_numeric_dtype(s):
            if _is_binary_series(s):
                bin_cols.append(c)
            else:
                cont_cols.append(c)
    return cont_cols, bin_cols


def _prepare_design(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    X = df[cols].copy()
    # one-hot encode object/categorical
    obj_cols = [c for c in cols if (df[c].dtype == 'object' or pd.api.types.is_categorical_dtype(df[c]))]
    if obj_cols:
        X = pd.get_dummies(X, columns=obj_cols, drop_first=True)
    X = sm.add_constant(X, has_constant='add')
    return X


def _cohens_d(x_in: np.ndarray, x_out: np.ndarray) -> float:
    n1, n2 = len(x_in), len(x_out)
    if n1 < 2 or n2 < 2:
        return np.nan
    m1, m2 = np.nanmean(x_in), np.nanmean(x_out)
    s1, s2 = np.nanstd(x_in, ddof=1), np.nanstd(x_out, ddof=1)
    sp = np.sqrt(((n1-1)*s1*s1 + (n2-1)*s2*s2) / (n1+n2-2))
    if sp == 0:
        return 0.0
    return (m1 - m2) / sp


def _log_odds_ratio(a: int, b: int, c: int, d: int, eps: float = 0.5) -> float:
    # 2x2: in_group yes/no vs outcome 1/0
    return math.log(((a + eps) * (d + eps)) / ((b + eps) * (c + eps)))


def validators_analysis(
    membership_wide: pd.DataFrame,
    validators_df: pd.DataFrame,
    continuous_validators: List[str],
    binary_validators: List[str],
    covariates: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    subject_col = 'src_subject_id' if 'src_subject_id' in validators_df.columns else validators_df.columns[0]
    df = validators_df.set_index(subject_col)
    common_idx = df.index.intersection(membership_wide.index)

    df = df.loc[common_idx]
    M = membership_wide.loc[common_idx]

    rows_desc = []
    rows_reg = []

    for combo_key in M.columns:
        in_mask = M[combo_key].astype(bool).values
        out_mask = ~in_mask
        n_in = int(in_mask.sum())
        n_out = int(out_mask.sum())
        if n_in < 10:
            continue
        # continuous
        for y in continuous_validators:
            yv = df[y].to_numpy(dtype=float)
            mean_in = float(np.nanmean(yv[in_mask])) if n_in > 0 else np.nan
            mean_out = float(np.nanmean(yv[out_mask])) if n_out > 0 else np.nan
            d = _cohens_d(yv[in_mask], yv[out_mask])
            rows_desc.append([combo_key, y, 'continuous', n_in, n_out, mean_in, mean_out, d, np.nan])

            # OLS with covariates
            try:
                X = pd.DataFrame({'in_group': M[combo_key].astype(int)})
                if covariates:
                    X = pd.concat([X, df[covariates]], axis=1)
                X = _prepare_design(pd.concat([X, df[covariates] if covariates else df.iloc[:, :0]], axis=1), X.columns.tolist())
                model = sm.OLS(yv, X, missing='drop').fit(cov_type='HC3')
                p = model.pvalues.get('in_group', np.nan)
                beta = model.params.get('in_group', np.nan)
            except Exception as e:
                p, beta = np.nan, np.nan
            rows_reg.append([combo_key, y, 'OLS', n_in, n_out, beta, p])

        # binary
        for y in binary_validators:
            yv = df[y].astype(float).to_numpy()
            a = int(((yv == 1) & in_mask).sum())
            b = int(((yv == 0) & in_mask).sum())
            c = int(((yv == 1) & out_mask).sum())
            d0 = int(((yv == 0) & out_mask).sum())
            log_or = _log_odds_ratio(a, b, c, d0)
            rows_desc.append([combo_key, y, 'binary', n_in, n_out, a/n_in if n_in else np.nan, c/n_out if n_out else np.nan, log_or, np.nan])

            # Logit with covariates
            try:
                X = pd.DataFrame({'in_group': M[combo_key].astype(int)})
                if covariates:
                    X = pd.concat([X, df[covariates]], axis=1)
                X = _prepare_design(pd.concat([X, df[covariates] if covariates else df.iloc[:, :0]], axis=1), X.columns.tolist())
                model = sm.Logit(yv, X, missing='drop').fit(disp=False, maxiter=200)
                p = model.pvalues.get('in_group', np.nan)
                beta = model.params.get('in_group', np.nan)
            except Exception:
                p, beta = np.nan, np.nan
            rows_reg.append([combo_key, y, 'Logit', n_in, n_out, beta, p])

    desc = pd.DataFrame(rows_desc, columns=["combo_key","outcome","type","n_in","n_out","mean_in","mean_out","effect_size","p_value"])
    reg = pd.DataFrame(rows_reg, columns=["combo_key","outcome","model","n_in","n_out","beta_in_group","p_value"])
    return desc, reg


# -------------------------
# Main pipeline for one AE config
# -------------------------
@dataclass
class ROIPipelineResult:
    rois_df: pd.DataFrame
    dkls_matrix: np.ndarray
    clusters: Dict[int, List[str]]
    groups_df: pd.DataFrame
    membership_long: pd.DataFrame
    membership_wide: pd.DataFrame
    validators_summary: pd.DataFrame
    regression_results: pd.DataFrame


def run_roi_pipeline(
    factors_df: pd.DataFrame,
    validators_df: Optional[pd.DataFrame] = None,
    save_dir: Optional[str] = None,
    factor_prefix: str = "factor_",
    subject_col: str = "src_subject_id",
    max_gmm_components: int = 6,
    top_m: int = 2,
    min_group_size: int = 100,
    continuous_validators: Optional[List[str]] = None,
    binary_validators: Optional[List[str]] = None,
    covariates: Optional[List[str]] = None,
    random_state: int = 7,
) -> ROIPipelineResult:
    if subject_col not in factors_df.columns:
        raise ValueError(f"factors_df 必须包含标识列 '{subject_col}'")
    factor_cols = [c for c in factors_df.columns if c.startswith(factor_prefix)]
    if not factor_cols:
        raise ValueError(f"未找到以 '{factor_prefix}' 开头的潜在因子列")

    # 1) DKLS + per-dim GMM + ROI
    D, gmms, rois_map = dkls_matrix_for_dimensions(factors_df, factor_cols, random_state=random_state)
    rois_records: List[Tuple] = []
    for dname, rois_list in rois_map.items():
        for r in rois_list:
            rois_records.append((dname, r.k, r.weight, r.mean, r.std, r.LT, r.UT))
    rois_df = pd.DataFrame(rois_records, columns=["dim","component","weight","mean","std","LT","UT"]).sort_values(["dim","component"]).reset_index(drop=True)

    # 2) AHC on DKLS
    clusters = _agglomerative_on_distance(D, factor_cols, random_state=random_state)

    # 3) ROI 组合 + 会员矩阵
    combos, membership_long, membership_wide = build_roi_combos(
        factors_df=factors_df[[subject_col] + factor_cols],
        clusters=clusters,
        rois=rois_map,
        top_m=top_m,
        min_group_size=min_group_size,
    )

    groups_df = pd.DataFrame([
        {
            "cluster_id": c.cluster_id,
            "combo_key": f"C{c.cluster_id}|" + "&".join([f"{d}[{c.combo[d][0]:.6g},{c.combo[d][1]:.6g}]" for d in c.dims]),
            "dims": ",".join(c.dims),
            "size": c.size,
            "proportion": c.proportion,
        }
        for c in combos
    ])

    # 4) Validators 统计与回归
    if validators_df is not None and not membership_wide.empty:
        # 自动探测（若未提供）
        if continuous_validators is None or binary_validators is None:
            exclude = set([subject_col]) | set([c for c in validators_df.columns if c.startswith(factor_prefix)])
            cont_auto, bin_auto = auto_detect_validators(validators_df, exclude=exclude)
            continuous_validators = cont_auto if continuous_validators is None else continuous_validators
            binary_validators = bin_auto if binary_validators is None else binary_validators
        desc, reg = validators_analysis(
            membership_wide=membership_wide,
            validators_df=validators_df,
            continuous_validators=continuous_validators,
            binary_validators=binary_validators,
            covariates=covariates,
        )
    else:
        desc = pd.DataFrame()
        reg = pd.DataFrame()

    # 5) 保存输出
    if save_dir is not None:
        _ensure_dir(save_dir)
        rois_df.to_csv(os.path.join(save_dir, "rois_definitions.csv"), index=False)
        pd.DataFrame(D, index=factor_cols, columns=factor_cols).to_csv(os.path.join(save_dir, "dkls_matrix.csv"))
        # simple heatmap
        plt.figure(figsize=(max(5, 0.3*len(factor_cols)), max(4, 0.3*len(factor_cols))))
        plt.imshow(D, interpolation='nearest')
        plt.colorbar()
        plt.xticks(range(len(factor_cols)), factor_cols, rotation=90)
        plt.yticks(range(len(factor_cols)), factor_cols)
        plt.title("Symmetric KL Distance (factors)")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "dkls_heatmap.png"), dpi=200)
        plt.close()

        # clusters
        with open(os.path.join(save_dir, "clusters.txt"), 'w', encoding='utf-8') as f:
            for cid, dims in clusters.items():
                f.write(f"Cluster {cid}: {', '.join(dims)}\n")

        groups_df.to_csv(os.path.join(save_dir, "roi_groups.csv"), index=False)
        membership_long.to_csv(os.path.join(save_dir, "membership_long.csv"), index=False)
        membership_wide.astype(int).to_csv(os.path.join(save_dir, "membership_wide.csv"))
        desc.to_csv(os.path.join(save_dir, "validators_summary.csv"), index=False)
        reg.to_csv(os.path.join(save_dir, "regression_results.csv"), index=False)

    return ROIPipelineResult(
        rois_df=rois_df,
        dkls_matrix=D,
        clusters=clusters,
        groups_df=groups_df,
        membership_long=membership_long,
        membership_wide=membership_wide,
        validators_summary=desc,
        regression_results=reg,
    )


# -------------------------
# Stability across multiple AE configs (optional)
# -------------------------
@dataclass
class StabilityResult:
    pairwise_ari: pd.DataFrame
    pairwise_jaccard: pd.DataFrame


def evaluate_stability_across_configs(
    memberships: Dict[str, pd.DataFrame]
) -> StabilityResult:
    """Compare group assignments across configs.
    memberships: dict[label -> membership_wide DataFrame]
    """
    labels = list(memberships.keys())
    # union of subjects
    subjects = None
    for m in memberships.values():
        subjects = m.index if subjects is None else subjects.union(m.index)
    subjects = subjects if subjects is not None else pd.Index([])

    # turn multi-label group membership into a single label via argmax (fallback: all False -> label 0)
    single_labels: Dict[str, np.ndarray] = {}
    for k, M in memberships.items():
        M = M.reindex(subjects).fillna(False)
        # choose the most frequent group per subject
        counts = M.astype(int).values
        lab = np.argmax(counts, axis=1) if counts.size else np.zeros(len(subjects), dtype=int)
        single_labels[k] = lab

    n = len(labels)
    ARI = np.zeros((n, n))
    J = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if j <= i:
                continue
            ari = adjusted_rand_score(single_labels[labels[i]], single_labels[labels[j]])
            ARI[i, j] = ARI[j, i] = ari
            # jaccard on binary group union (rough heuristic)
            a = memberships[labels[i]].astype(int).sum(axis=1) > 0
            b = memberships[labels[j]].astype(int).sum(axis=1) > 0
            J[i, j] = J[j, i] = jaccard_score(a, b)
    df_ari = pd.DataFrame(ARI, index=labels, columns=labels)
    df_j = pd.DataFrame(J, index=labels, columns=labels)
    return StabilityResult(pairwise_ari=df_ari, pairwise_jaccard=df_j)


# -------------------------
# 小工具：将 AE 多配置统一跑一遍（d=10/20/30 等）
# -------------------------

def run_multiple_configs(
    factors_by_cfg: Dict[str, pd.DataFrame],
    validators_df: Optional[pd.DataFrame] = None,
    **kwargs
) -> Tuple[Dict[str, ROIPipelineResult], Optional[StabilityResult]]:
    results: Dict[str, ROIPipelineResult] = {}
    memberships: Dict[str, pd.DataFrame] = {}
    for cfg_name, fdf in factors_by_cfg.items():
        out_dir = None
        if 'save_dir' in kwargs and kwargs['save_dir'] is not None:
            out_dir = os.path.join(kwargs['save_dir'], cfg_name)
            kwargs_cfg = {**kwargs, 'save_dir': out_dir}
        else:
            kwargs_cfg = kwargs
        res = run_roi_pipeline(factors_df=fdf, validators_df=validators_df, **kwargs_cfg)
        results[cfg_name] = res
        memberships[cfg_name] = res.membership_wide
    stab = evaluate_stability_across_configs(memberships) if len(results) >= 2 else None
    # 保存稳定性
    if stab is not None and 'save_dir' in kwargs and kwargs['save_dir'] is not None:
        _ensure_dir(kwargs['save_dir'])
        stab.pairwise_ari.to_csv(os.path.join(kwargs['save_dir'], "stability_ARI.csv"))
        stab.pairwise_jaccard.to_csv(os.path.join(kwargs['save_dir'], "stability_Jaccard.csv"))
    return results, stab


# -------------------------
# 入口（可选演示，实际使用请在外部脚本中调用）
# -------------------------
if __name__ == "__main__":
    print("This module provides functions to run the ROI explanation pipeline.\n" \
          "Import and call run_roi_pipeline() in your analysis script.")
