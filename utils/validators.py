from pathlib import Path
from functools import reduce
import pandas as pd
from typing import Dict, List, Tuple, Optional
import os, warnings, itertools
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import scipy.stats as st
import os, itertools, re, warnings
from pathlib import Path
from typing import Optional, Union, Dict, List, Tuple
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler
import os, re, itertools, warnings
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
import statsmodels.api as sm
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.families.links import probit as probit_link, logit as logit_link
from matplotlib.ticker import PercentFormatter
from numpy.polynomial.legendre import legval
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures
import re
def build_validators_baseline(
    root: Path,
    dict_path: Path,
    validators: Dict[str, List[str]],
    eventname: str = "baseline_year_1_arm_1",
    out_dir: Path = Path("../output"),
    dict_sheet: Optional[str] = None,
    dict_engine: str = "openpyxl",
    verbose: bool = True
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
    # dict_df = pd.read_excel(
    #     dict_path, engine=dict_engine, sheet_name=dict_sheet
    # )[[ "var_name", "table_name" ]].dropna()
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
        tag_path = out_dir / f"{tag}_baseline.csv"
        df_tag.to_csv(tag_path, index=False, encoding="utf-8")
        out_frames[tag] = df_tag
        if verbose:
            print(f"  -> save {tag_path.name}  ({df_tag.shape[0]} rows, {df_tag.shape[1]} colums)")

    # summary as src_subject_id
    if out_frames:
        wide = reduce(lambda l, r: l.merge(r, on="src_subject_id", how="outer"),
                      out_frames.values())
        wide.to_csv(out_dir / "validators_baseline.csv", index=False, encoding="utf-8")
        if verbose:
            print("[OK] generate validators_baseline.csv :", wide.shape)
    else:
        wide = pd.DataFrame()
        if verbose:
            print("\n[WARN] no validators extracted!")

    return out_frames, wide


warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def compare_efa_poly_vs_ae_poly(
    factors_scores: pd.DataFrame,        # contains src_subject_id and factor_* (AE/NMF)
    efa_scores: pd.DataFrame,            # contains src_subject_id and efa_* (EFA)
    validators_csv: Union[str, Path] = "validators_baseline.csv",
    save_dir: str = "efa_vs_model_plots_poly_compare",
    model_label_ae: str = "AE",
    model_label_efa: str = "EFA",
    ae_poly_degree: int = 4,             # AE polynomial degree: 1/2/3/4
    efa_poly_degree: int = 4,            # EFA polynomial degree: 1/2/3/4
    include_interactions: bool = False,  # whether to include pairwise linear interactions (inside model)
    max_interactions_per_block: int = 200,
    random_seed: int = 6,
    compare_bootstrap_B: int = 30,       # bootstrap resamples for AE vs EFA differences
    baseline_map: Optional[dict] = None,    # {validator: baseline_R2}
    name_map: Optional[dict] = None,        # friendly display names
    validator_info: Optional[dict] = None,  # {validator: "cont" (continuous) or "bin" (binary)}
    oos_cv_folds: int = 5,              # 5 folds OOS-CV to estimate AE/EFA factor scores, if 1 then no CV
    rotation_invariant_full_poly: bool = True,
    strict_common_sample: bool = False,          # True = use a single unified listwise sample; False = listwise per-block (like SPSS/Mplus)
    use_ddof1_zscore: bool = True,              # True = z-score using sample std (ddof=1, like SPSS); False = use StandardScaler
    use_probit_for_binary: bool = True,        # True = use probit link for binary outcomes (closer to Mplus); False = use logit (like SPSS)
    allow_logit_ridge_fallback: bool = True,    # True = if logit fails, fallback to L2-regularized fit; False = do not fallback
    binary_r2_metric: str = "mplus_latent",
):
    """
    Compare hierarchical regression performance of AE (NMF) factors vs EFA factors on external validators
    using polynomial expansion (degree up to 4), optional interactions, and per-block incremental tests.

    Returns
    -------
    result : dict
        {
          "res_ae": DataFrame,
          "res_efa": DataFrame,
          "summary": DataFrame,
          "n_table": DataFrame,
          "paths": {
              "res_ae_csv": str,
              "res_efa_csv": str,
              "summary_csv": str,
              "grid_png": str
          }
        }
    """
    # ---------------- Defaults ----------------
    if baseline_map is None:
        baseline_map = {}

    if name_map is None:
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

    if validator_info is None:
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

    rng = np.random.default_rng(random_seed)
    np.random.seed(random_seed)
    os.makedirs(save_dir, exist_ok=True)

    # ---------------- 1) Load validators ----------------
    validators_csv = str(validators_csv)
    df_val = pd.read_csv(validators_csv).drop_duplicates("src_subject_id").copy()

    def _safe_has(cols: List[str]) -> bool:
        return all(c in df_val.columns for c in cols)

    # ---------- construct validators ----------
    if "dev_delay" not in df_val.columns:
        if _safe_has(["devhx_20_p", "devhx_21_p"]):
            vals = df_val[["devhx_20_p", "devhx_21_p"]].replace({999: np.nan})
            df_val["dev_delay"] = vals.mean(axis=1)


    if "fes_conflict" not in df_val.columns:
        fes_cols = [f"fes_youth_q{i}" for i in range(1,10)]
        if _safe_has(fes_cols):
            df_val["fes_conflict"] = df_val[fes_cols].mean(axis=1)

    if "n_friends" not in df_val.columns:
        rr = ["resiliency5a_y","resiliency6a_y","resiliency5b_y","resiliency6b_y"]

        vals = (
            df_val[rr]
            .apply(pd.to_numeric, errors="coerce")
            .clip(lower=0, upper=100)   #capping at 100 to avoid outliers
        )

        def recode_friend_count(series, kind="normal"):
            """friends number recode"""
            s = series.copy()
            if kind == "normal":  # normal (5a/6a)
                s = pd.cut(
                    s,
                    bins=[-np.inf,10,15,20,25,30,100],
                    labels=[10,11,12,13,14,15]
                ).astype(float)
            elif kind == "close":  # close (5b/6b)
                s = s.where(s <= 10, 11)
            return s

        recoded = pd.DataFrame({
            "resiliency5a": recode_friend_count(vals["resiliency5a_y"], kind="normal"),
            "resiliency6a": recode_friend_count(vals["resiliency6a_y"], kind="normal"),
            "resiliency5b": recode_friend_count(vals["resiliency5b_y"], kind="close"),
            "resiliency6b": recode_friend_count(vals["resiliency6b_y"], kind="close"),
        })

        df_val["n_friends"] = recoded.mean(axis=1, skipna=True) * 4

    if "school_conn" not in df_val.columns:
        env    = [f"school_{i}_y" for i in (2,3,4,5,6)]
        part   = [f"school_{i}_y" for i in (7,8,9,10)]
        detach = [f"school_{i}_y" for i in (12,15,17)]
        if _safe_has(env+part+detach):
            df_val["school_conn"] = (df_val[env].mean(axis=1) + df_val[part].mean(axis=1) - df_val[detach].mean(axis=1)) / 3

    if "avg_grades" not in df_val.columns and "kbi_p_grades_in_school" in df_val.columns:
        df_val["avg_grades"] = df_val["kbi_p_grades_in_school"].replace({6: np.nan, -1: np.nan})
    if "fluid_cog" not in df_val.columns and "nihtbx_fluidcomp_uncorrected" in df_val.columns:
        df_val["fluid_cog"] = df_val["nihtbx_fluidcomp_uncorrected"]
    if "cryst_cog" not in df_val.columns and "nihtbx_cryst_uncorrected" in df_val.columns:
        df_val["cryst_cog"] = df_val["nihtbx_cryst_uncorrected"]

    if "mh_service" not in df_val.columns and "kbi_p_c_mh_sa" in df_val.columns:
        df_val["mh_service"] = df_val["kbi_p_c_mh_sa"].map({1: 1.0, 2: 0.0, 3: np.nan}).astype(float)

    if "med_history" not in df_val.columns:
        med_cols = [f"medhx_2{c}" for c in "abcdefghijklmnopqrs"]
        if any(c in df_val.columns for c in med_cols):
            exist = [c for c in med_cols if c in df_val.columns]
            tmp = df_val[exist].replace({0:0,3:0,6:np.nan}).fillna(0).astype(float)
            df_val["med_history"] = tmp.max(axis=1)

    if "brought_meds" not in df_val.columns and "brought_medications" in df_val.columns:
        map_bm = {0: 1.0, 1: 1.0, 3: 0.0, 2: np.nan}
        df_val["brought_meds"] = df_val["brought_medications"].map(map_bm).astype(float)


    needed_outcomes = [y for y in validator_info.keys() if y in df_val.columns]
    if not needed_outcomes:
        raise ValueError("No validator columns found/constructed in validators CSV that match validator_info keys.")
    df_val = df_val[["src_subject_id"] + needed_outcomes].copy()

    # ---------------- 2) Merge AE/EFA with validators ----------------
    ae_df  = factors_scores.drop_duplicates("src_subject_id").copy()
    efa_df = efa_scores.drop_duplicates("src_subject_id").copy()

    ae_base  = sorted([c for c in ae_df.columns  if c.startswith("factor_")],
                      key=lambda s: int(re.search(r"_(\d+)$", s).group(1)) if re.search(r"_(\d+)$", s) else 10**9)
    efa_base = sorted([c for c in efa_df.columns if c.startswith("efa_")],
                      key=lambda s: int(re.search(r"_(\d+)$", s).group(1)) if re.search(r"_(\d+)$", s) else 10**9)
    if not ae_base or not efa_base:
        raise ValueError("AE base columns (factor_*) or EFA base columns (efa_*) not found.")

    K = min(len(ae_base), len(efa_base))
    ae_base  = ae_base[:K]
    efa_base = efa_base[:K]

    panel = df_val.merge(ae_df[["src_subject_id"] + ae_base], on="src_subject_id", how="inner")
    panel = panel.merge(efa_df[["src_subject_id"] + efa_base], on="src_subject_id", how="inner")

    # ---------------- 3) Polynomial expansion & blocks ----------------

    def expand_polynomials(df_ref: pd.DataFrame, bases: List[str], degree: int,
                        prefix_sq="__P2", prefix_cu="__P3", prefix_qu="__P4") -> Dict[str, List[str]]:
        """
        Use Legendre orthogonal polynomials (computed after column-wise z-scoring).
        This is more numerically stable than using raw power terms.
        """
        colmap = {c: [c] for c in bases}
        if degree <= 1:
            return colmap

        # 先对 bases 做样本内 z-score（ddof=1）
        X = df_ref[bases].astype(float)
        mu = X.mean(0)
        sd = X.std(0, ddof=1).replace(0, np.nan)
        Xz = (X - mu) / sd

        # 逐列构造 P2,P3,P4...
        for c in bases:
            xc = Xz[c].to_numpy(dtype=float)
            # P2
            if degree >= 2:
                P2 = legval(xc, [0, 0, 1])   # 系数向量 [0,0,1] 表示 P2
                c2 = f"{c}{prefix_sq}"
                df_ref[c2] = P2
                colmap[c].append(c2)
            # P3
            if degree >= 3:
                P3 = legval(xc, [0, 0, 0, 1])
                c3 = f"{c}{prefix_cu}"
                df_ref[c3] = P3
                colmap[c].append(c3)
            # P4
            if degree >= 4:
                P4 = legval(xc, [0, 0, 0, 0, 1])
                c4 = f"{c}{prefix_qu}"
                df_ref[c4] = P4
                colmap[c].append(c4)

        return colmap

    def build_blocks_with_poly(df_ref: pd.DataFrame, bases: List[str], poly_map: Dict[str, List[str]],
                               include_inter: bool = False, max_inter: int = 200) -> List[List[str]]:
        blocks = []
        for i in range(1, len(bases)+1):
            main_terms = []
            for c in bases[:i]:
                main_terms.extend(poly_map[c])
            inter_terms = []
            if include_inter and i >= 2:
                for a, b in itertools.combinations(bases[:i], 2):
                    name = f"{a}__X__{b}"
                    if name not in df_ref.columns:
                        df_ref[name] = df_ref[a] * df_ref[b]
                    inter_terms.append(name)
                if len(inter_terms) > max_inter:
                    inter_terms = inter_terms[:max_inter]
            blocks.append(main_terms + inter_terms)
        return blocks


    def build_full_poly_blocks(df_ref: pd.DataFrame, bases: List[str], degree: int, model_tag: str) -> List[List[str]]:
        """
        Rotation-invariant full polynomial basis:
        - Z-score the `bases` columns in-sample (ddof=1).
        - Use sklearn.preprocessing.PolynomialFeatures to generate all monomials up to `degree`
          (including interaction terms and powers).
        - Create new columns on df_ref prefixed with f"{model_tag}__" to avoid name collisions.
        - For each block i=1..K, return the list of generated columns that involve only the first
          i base variables (so blocks are cumulative by the original variable order).
        """
        if degree <= 1:
            return [bases[:i] for i in range(1, len(bases)+1)]

        X = df_ref[bases].astype(float)
        mu = X.mean(0); sd = X.std(0, ddof=1).replace(0, np.nan)
        Xz = (X - mu) / sd

        poly = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=False)
        Z = poly.fit_transform(Xz[bases])
        names = poly.get_feature_names_out(bases)  # e.g. "factor_1", "factor_1 factor_2", "factor_1^2"

        out_cols = []
        for j, nm in enumerate(names):
            col = f"{model_tag}__{nm.replace(' ', '__x__').replace('^', '__p')}"
            df_ref[col] = Z[:, j]
            out_cols.append(col)

        token_re = re.compile(r"[A-Za-z_]\w*")
        def uses_within_first_i(nm: str, i: int) -> bool:
            # nm 用的是原始 feature name（没加前缀、没替换空格/幂），便于判断涉及哪些变量
            vars_in = set(token_re.findall(nm))
            return vars_in.issubset(set(bases[:i]))

        blocks = []
        for i in range(1, len(bases)+1):
            cols_i = [c for c, nm in zip(out_cols, names) if uses_within_first_i(nm, i)]
            blocks.append(cols_i)
        return blocks


    panel_all = panel.copy()
    # ae_poly_map  = expand_polynomials(panel_all, ae_base,  ae_poly_degree)
    # efa_poly_map = expand_polynomials(panel_all, efa_base, efa_poly_degree)

    # ae_blocks  = build_blocks_with_poly(panel_all, ae_base,  ae_poly_map,
    #                                     include_interactions, max_interactions_per_block)
    # efa_blocks = build_blocks_with_poly(panel_all, efa_base,  efa_poly_map,
    #                                     include_interactions, max_interactions_per_block)
    panel_all = panel.copy()

    # ✅ 新：旋转不敏感完整多项式；如需回退老方式，把开关设为 False
    if rotation_invariant_full_poly:
        ae_blocks  = build_full_poly_blocks(panel_all, ae_base,  ae_poly_degree,  model_tag="AE")
        efa_blocks = build_full_poly_blocks(panel_all, efa_base,  efa_poly_degree, model_tag="EFA")
    else:
        ae_poly_map  = expand_polynomials(panel_all, ae_base,  ae_poly_degree)
        efa_poly_map = expand_polynomials(panel_all, efa_base,  efa_poly_degree)
        ae_blocks  = build_blocks_with_poly(panel_all, ae_base,  ae_poly_map,
                                            include_interactions, max_interactions_per_block)
        efa_blocks = build_blocks_with_poly(panel_all, efa_base,  efa_poly_map,
                                            include_interactions, max_interactions_per_block)

    need_cols_ae  = sorted({t for blk in ae_blocks  for t in blk})
    need_cols_efa = sorted({t for blk in efa_blocks for t in blk})
    need_common   = sorted(set(need_cols_ae) | set(need_cols_efa))


    need_cols_ae  = sorted({t for blk in ae_blocks  for t in blk})
    need_cols_efa = sorted({t for blk in efa_blocks for t in blk})
    need_common   = sorted(set(need_cols_ae) | set(need_cols_efa))

    # ---------------- 4) Helpers ----------------
    def loglik_null_binary(y: np.ndarray) -> float:
        y = y.astype(float)
        p = float(np.nanmean(y))
        p = np.clip(p, 1e-8, 1-1e-8)
        return float(np.nansum(y*np.log(p) + (1-y)*np.log(1-p)))

    def _zscore_ddof1_inplace(df_ref: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        x = df_ref[cols].astype(float)
        mu = x.mean(axis=0)
        sd = x.std(axis=0, ddof=1).replace(0, np.nan)
        df_ref[cols] = (x - mu) / sd
        return df_ref

    def _standardize_inplace(df_ref: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        if use_ddof1_zscore:
            return _zscore_ddof1_inplace(df_ref, cols)
        else:
            scaler = StandardScaler()
            df_ref[cols] = scaler.fit_transform(df_ref[cols])
            return df_ref


    def _total_r2_cont(df_ref: pd.DataFrame, y: str, cols: List[str]) -> float:
        d = df_ref.copy()

        # 样本外（K 折）
        if oos_cv_folds and oos_cv_folds > 1:
            return _total_r2_cont_oos(d, y, cols, folds=oos_cv_folds)

        # 样本内
        if not cols:
            return np.nan

        # 先剔除零/近零方差列
        s = d[cols].std(numeric_only=True, ddof=1)
        cols2 = [c for c in cols if s.get(c, 0.0) > 1e-12]
        if not cols2:
            return np.nan

        x = d[cols2].astype(float)
        mu = x.mean(0)
        sd = x.std(0, ddof=1).replace(0, np.nan)
        Xz = ((x - mu) / sd).to_numpy()
        if np.isnan(Xz).any():
            return np.nan  # 理论上上游已 dropna；这里兜底

        yv = d[y].to_numpy(dtype=float)

        alphas = np.logspace(-4, 3, 20)
        m = RidgeCV(alphas=alphas, store_cv_values=False)
        m.fit(Xz, yv)

        yhat = m.predict(Xz)
        ssr = np.sum((yv - yhat)**2)
        sst = np.sum((yv - np.mean(yv))**2)
        if sst <= 0:
            return np.nan
        return float(max(0.0, 1 - ssr/sst))


    def _total_r2_cont_oos(d: pd.DataFrame, y: str, cols: List[str], folds: int = 5) -> float:
        X = d[cols].astype(float).to_numpy()
        yv = d[y].astype(float).to_numpy()
        if np.nanstd(yv) <= 0 or X.shape[1] == 0:
            return np.nan
        kf = KFold(n_splits=folds, shuffle=True, random_state=random_seed)
        y_pred = np.full_like(yv, np.nan, dtype=float)
        alphas = np.logspace(-4, 3, 20)
        for tr, te in kf.split(X):
            Xtr, Xte = X[tr], X[te]
            ytr = yv[tr]
            # 折内样本 z-score（ddof=1）
            mu = Xtr.mean(0); sd = Xtr.std(0, ddof=1)
            sd[sd == 0] = np.nan
            Xtrz = (Xtr - mu) / sd
            Xtez = (Xte - mu) / sd
            m = RidgeCV(alphas=alphas, store_cv_values=False)
            m.fit(Xtrz, ytr)
            y_pred[te] = m.predict(Xtez)
        mask = ~np.isnan(y_pred) & ~np.isnan(yv)
        if mask.sum() < 3:
            return np.nan
        ssr = np.sum((yv[mask] - y_pred[mask])**2)
        sst = np.sum((yv[mask] - np.mean(yv[mask]))**2)
        return float(max(0.0, 1 - ssr/sst))

    def _total_r2_bin_oos(d: pd.DataFrame, y: str, cols: List[str], folds: int = 5, metric: str = "nagelkerke") -> float:
        X = d[cols].astype(float)
        yv = d[y].astype(float).to_numpy()
        if X.shape[1] == 0 or np.unique(yv[~np.isnan(yv)]).size < 2:
            return np.nan
        kf = KFold(n_splits=folds, shuffle=True, random_state=random_seed)
        p_hat_all = np.full_like(yv, np.nan, dtype=float)

        for tr, te in kf.split(X):
            dtr = d.iloc[tr].copy(); dte = d.iloc[te].copy()
            # 折内标准化
            s = dtr[cols].std(numeric_only=True)
            cols2 = [c for c in cols if s.get(c, 0.0) > 1e-12]
            if not cols2:
                continue
            dtr = _standardize_inplace(dtr, cols2)
            dte = _standardize_inplace(dte, cols2)
            yname = y
            formula = f"{yname} ~ " + " + ".join(cols2)
            try:
                m, used_reg, p_trash, _ = _fit_binary_general(formula, dtr, use_probit_for_binary)
                p_hat = np.asarray(m.predict(dte))
                p_hat_all[te] = p_hat
            except Exception:
                continue

        mask = ~np.isnan(p_hat_all) & ~np.isnan(yv)
        if mask.sum() < 3:
            return np.nan

        y_test = yv[mask]; p_test = np.clip(p_hat_all[mask], 1e-8, 1-1e-8)
        if metric == "tjur":
            return float(np.nanmean(p_test[y_test == 1]) - np.nanmean(p_test[y_test == 0]))
        elif metric == "mplus_latent":
            if use_probit_for_binary:
                from scipy.stats import norm
                eta = norm.ppf(p_test)
            else:
                eta = np.log(p_test/(1-p_test))
            var_eta = float(np.nanvar(eta, ddof=1))
            resid_var = 1.0 if use_probit_for_binary else (np.pi**2 / 3.0)
            denom = var_eta + resid_var
            if denom <= 0 or not np.isfinite(denom):
                return np.nan
            return float(np.clip(var_eta / denom, 0.0, 1.0))
        else:
            # Nagelkerke 样本外：用测试折各自的基线对数似然
            ll0 = loglik_null_binary(y_test)
            ll1 = _bin_loglik_from_pred(y_test, p_test)
            return r2_nagelkerke(ll0, ll1, len(y_test))

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

    def _fit_logit_with_optional_ridge(formula: str, data: pd.DataFrame,
                                       maxiter: int = 100,
                                       alpha: float = 1e-4) -> Tuple[object, bool, np.ndarray, float]:
        """
        Return: (model, used_regularization, p_hat, ll_model)
        """
        # Try plain logit first
        try:
            m = smf.logit(formula, data=data).fit(disp=0, method="newton", maxiter=maxiter)
            p_hat = np.asarray(m.predict(data))
            ll1 = _bin_loglik_from_pred(data[formula.split("~")[0].strip()].values.astype(float), p_hat)
            return m, False, p_hat, ll1
        except Exception:
            if not allow_logit_ridge_fallback:
                raise
            # Ridge regularization fallback
            for a in (alpha, 1e-3, 1e-2):
                try:
                    m = smf.logit(formula, data=data).fit_regularized(
                        alpha=a, L1_wt=0.0, disp=0, maxiter=500
                    )
                    p_hat = np.asarray(m.predict(data))
                    ll1 = _bin_loglik_from_pred(data[formula.split("~")[0].strip()].values.astype(float), p_hat)
                    return m, True, p_hat, ll1
                except Exception:
                    continue
            raise

    def _fit_binary_general(formula: str, data: pd.DataFrame,
                            use_probit: bool) -> Tuple[object, bool, np.ndarray, float]:
        """
        统一接口：probit（GLM）或 logit（Logit/正则）
        Return: (model, used_regularization, p_hat, ll_model)
        """
        yname = formula.split("~")[0].strip()
        if use_probit:
            # GLM Binomial with probit link
            try:
                model = sm.GLM.from_formula(formula, data=data,
                                            family=Binomial(link=probit_link()))
                m = model.fit()
                p_hat = np.asarray(m.predict(data))
                ll1 = _bin_loglik_from_pred(data[yname].values.astype(float), p_hat)
                return m, False, p_hat, ll1
            except Exception:
                # 若 GLM 失败，返回 NaN
                raise
        else:
            return _fit_logit_with_optional_ridge(formula, data)

    def _total_r2_bin(df_ref: pd.DataFrame, y: str, cols: List[str], metric: str = "nagelkerke") -> float:
        # 先复制，避免 NameError
        d = df_ref.copy()

        # 样本外：直接用 d，并把 metric 传下去
        if oos_cv_folds and oos_cv_folds > 1:
            return _total_r2_bin_oos(d, y, cols, folds=oos_cv_folds, metric=metric)

        # 样本内（你原来的分支）
        if not cols:
            return np.nan

        s = d[cols].std(numeric_only=True)
        cols = [c for c in cols if s.get(c, 0.0) > 1e-12]
        if not cols:
            return np.nan

        d = _standardize_inplace(d, cols)
        ybin = d[y].astype(float).values
        ll0  = loglik_null_binary(ybin)

        formula = f"{y} ~ " + " + ".join(cols)
        try:
            m, used_reg, p_hat, ll1 = _fit_binary_general(formula, d, use_probit_for_binary)
        except Exception:
            return np.nan

        # 指标选择
        if metric == "tjur":
            r2 = float(np.nanmean(p_hat[ybin == 1]) - np.nanmean(p_hat[ybin == 0]))
            return float(np.clip(r2, 0.0, 1.0))

        if metric == "mplus_latent":
            try:
                if use_probit_for_binary:
                    from scipy.stats import norm
                    eta = norm.ppf(np.clip(p_hat, 1e-8, 1-1e-8))
                else:
                    p = np.clip(p_hat, 1e-8, 1-1e-8)
                    eta = np.log(p/(1-p))
            except Exception:
                return np.nan
            var_eta   = float(np.nanvar(eta, ddof=1))
            resid_var = 1.0 if use_probit_for_binary else (np.pi**2 / 3.0)
            denom = var_eta + resid_var
            if denom <= 0 or not np.isfinite(denom):
                return np.nan
            return float(np.clip(var_eta / denom, 0.0, 1.0))

        # 默认：Nagelkerke
        return r2_nagelkerke(ll0, ll1, len(d))


    def _pvalue_compare_bootstrap(panel_in: pd.DataFrame, y: str, ytype: str,
                                  ae_cols: List[str], efa_cols: List[str],
                                  need_common_cols_this_block: List[str],
                                  B: int = 300, seed: int = 42) -> float:
        # block 级别的样本并集
        subset_cols = [y] + [c for c in need_common_cols_this_block if c in panel_in.columns]
        if len(subset_cols) <= 1:
            return np.nan
        d_all = panel_in.dropna(subset=subset_cols, how="any").copy()
        if d_all.empty or not ae_cols or not efa_cols:
            return np.nan
        ae_use  = [c for c in ae_cols  if c in d_all.columns]
        efa_use = [c for c in efa_cols if c in d_all.columns]
        if not ae_use or not efa_use:
            return np.nan

        if ytype == "cont":
            r2_ae  = _total_r2_cont(d_all, y, ae_use)
            r2_efa = _total_r2_cont(d_all, y, efa_use)
        else:
            r2_ae  = _total_r2_bin(d_all, y, ae_use)
            r2_efa = _total_r2_bin(d_all, y, efa_use)
        if not (np.isfinite(r2_ae) and np.isfinite(r2_efa)):
            return np.nan
        obs_diff = r2_ae - r2_efa

        rng_local = np.random.default_rng(seed)
        n   = len(d_all)
        diffs = []
        for _ in range(int(B)):
            idx = rng_local.integers(0, n, size=n)
            db = d_all.iloc[idx].copy()
            try:
                if ytype == "cont":
                    r2a = _total_r2_cont(db, y, ae_use)
                    r2e = _total_r2_cont(db, y, efa_use)
                else:
                    r2a = _total_r2_bin(db, y, ae_use)
                    r2e = _total_r2_bin(db, y, efa_use)
            except Exception:
                continue
            if np.isfinite(r2a) and np.isfinite(r2e):
                diffs.append(r2a - r2e)

        if len(diffs) < 20:
            return np.nan
        diffs = np.array(diffs, dtype=float)
        p = 2 * min((diffs >= obs_diff).mean(), (diffs <= obs_diff).mean())
        return float(np.clip(p, 0.0, 1.0))

    # ---------------- 5) Hierarchical regression ----------------
    def hierarchical_with_poly(panel_in: pd.DataFrame, y: str, ytype: str,
                               blocks_model: List[List[str]],
                               need_common_cols_all: List[str]) -> List[dict]:
        if y not in panel_in.columns:
            return []

        rows, prev_model, prev_like = [], None, 0.0

        # 准备“统一样本”情况下的全集样本
        if strict_common_sample:
            subset_cols_all = [y] + [c for c in need_common_cols_all if c in panel_in.columns]
            if len(subset_cols_all) <= 1:
                return []
            base_df_all = panel_in.dropna(subset=subset_cols_all, how="any").copy()
        else:
            base_df_all = panel_in  # 逐块时，每块再做各自 listwise

        for i, preds in enumerate(blocks_model, start=1):
            use_cols = [c for c in preds if c in base_df_all.columns]
            if not use_cols:
                rows.append({"outcome": y, "block": i, "predictors": "",
                             "ΔR2": np.nan, "ΔNagelkerke_R2": np.nan,
                             "p_value": np.nan, "total_like": np.nan, "n_obs": 0})
                continue

            if strict_common_sample:
                d_all = base_df_all.copy()
            else:
                subset_cols_blk = [y] + use_cols
                d_all = base_df_all.dropna(subset=subset_cols_blk, how="any").copy()

            if d_all.empty:
                rows.append({"outcome": y, "block": i, "predictors": "",
                             "ΔR2": np.nan, "ΔNagelkerke_R2": np.nan,
                             "p_value": np.nan, "total_like": np.nan, "n_obs": 0})
                continue
            if ytype == "bin" and d_all[y].nunique() < 2:
                rows.append({"outcome": y, "block": i, "predictors": "",
                             "ΔR2": np.nan, "ΔNagelkerke_R2": np.nan,
                             "p_value": np.nan, "total_like": np.nan, "n_obs": len(d_all)})
                continue

            if ytype == "cont":
                d = _standardize_inplace(d_all.copy(), use_cols)
                formula = f"{y} ~ " + " + ".join(use_cols)
                model = smf.ols(formula, data=d).fit()
                r2 = float(model.rsquared)
                if prev_model is not None:
                    try:
                        pchg = model.compare_f_test(prev_model)[1]
                    except Exception:
                        pchg = np.nan
                else:
                    pchg = np.nan
                delta = r2 - (prev_like if np.isfinite(prev_like) else 0.0)
                rows.append({"outcome": y, "block": i, "predictors": "+".join(use_cols),
                             "ΔR2": float(delta) if np.isfinite(delta) else np.nan,
                             "ΔNagelkerke_R2": np.nan,
                             "p_value": float(pchg) if pchg == pchg else np.nan,
                             "total_like": r2, "n_obs": len(d)})
                prev_model, prev_like = model, r2

            else:
                # binary
                s = d_all[use_cols].std(numeric_only=True)
                use_cols2 = [c for c in use_cols if s.get(c, 0.0) > 1e-12]
                if not use_cols2:
                    rows.append({"outcome": y, "block": i, "predictors": "",
                                 "ΔR2": np.nan, "ΔNagelkerke_R2": np.nan,
                                 "p_value": np.nan, "total_like": np.nan, "n_obs": len(d_all)})
                    continue

                d = _standardize_inplace(d_all.copy(), use_cols2)
                ybin = d[y].astype(float).values
                ll0  = loglik_null_binary(ybin)
                formula = f"{y} ~ " + " + ".join(use_cols2)

                try:
                    model, used_reg, p_hat, ll1 = _fit_binary_general(formula, d, use_probit_for_binary)
                except Exception:
                    rows.append({"outcome": y, "block": i, "predictors": "+".join(use_cols2),
                                 "ΔR2": np.nan, "ΔNagelkerke_R2": np.nan,
                                 "p_value": np.nan, "total_like": np.nan, "n_obs": len(d)})
                    continue

                nk = r2_nagelkerke(ll0, ll1, len(d))

                # 嵌套检验：logit + 未正则时可用；probit 或正则化后不严格
                if (prev_model is not None) and (not used_reg) and hasattr(prev_model, "llf") and hasattr(model, "llf") and (not use_probit_for_binary):
                    try:
                        from scipy.stats import chi2
                        ll_prev = float(prev_model.llf)
                        df_diff = max(int(getattr(model, "df_model", 0) - getattr(prev_model, "df_model", 0)), 1)
                        stat = 2 * (float(model.llf) - ll_prev)
                        pchg = 1 - chi2.cdf(stat, df_diff)
                    except Exception:
                        pchg = np.nan
                else:
                    pchg = np.nan

                delta = nk - (prev_like if np.isfinite(prev_like) else 0.0)
                rows.append({"outcome": y, "block": i, "predictors": "+".join(use_cols2),
                             "ΔR2": np.nan,
                             "ΔNagelkerke_R2": float(delta) if np.isfinite(delta) else np.nan,
                             "p_value": float(pchg) if pchg == pchg else np.nan,
                             "total_like": nk, "n_obs": len(d)})
                prev_model, prev_like = model, nk

        return rows

    def run_model(panel_in: pd.DataFrame, label: str, blocks_model: List[List[str]],
                  need_common_cols_all: List[str]) -> pd.DataFrame:
        all_rows = []
        for y, ytype in validator_info.items():
            if y not in panel_in.columns:
                continue
            rows = hierarchical_with_poly(panel_in, y, ytype, blocks_model, need_common_cols_all)
            all_rows.extend(rows)
        res = pd.DataFrame(all_rows)

        def pick_delta(row):
            if pd.notna(row.get("ΔR2", np.nan)):
                return row["ΔR2"]
            if pd.notna(row.get("ΔNagelkerke_R2", np.nan)):
                return row["ΔNagelkerke_R2"]
            return np.nan

        if not res.empty:
            res = res.sort_values(["outcome", "block"]).copy()
            res["Δmetric"] = res.apply(pick_delta, axis=1)
            res["cum_delta"] = res.groupby("outcome")["Δmetric"].cumsum()
            res["baseline_R2"] = res["outcome"].map(lambda x: baseline_map.get(x, 0.0))
            res["total_R2"] = res["baseline_R2"] + res["cum_delta"]
            res["model"] = label
        return res

    res_ae_poly  = run_model(panel_all, model_label_ae,  ae_blocks,  need_common)
    res_efa_poly = run_model(panel_all, model_label_efa, efa_blocks,  need_common)

    # Export
    res_ae_csv  = os.path.join(save_dir, "res_df_ae_poly_only.csv")
    res_efa_csv = os.path.join(save_dir, "res_df_efa_poly_only.csv")
    res_ae_poly.to_csv(res_ae_csv, index=False)
    res_efa_poly.to_csv(res_efa_csv, index=False)

    # n_table（按最终各自结果的最后一块记录的 n_obs）
    if not res_ae_poly.empty and not res_efa_poly.empty:
        n_tbl = (
            pd.concat([res_ae_poly.assign(side="AE"), res_efa_poly.assign(side="EFA")])
              .sort_values(["outcome","model","block"])
              .groupby(["outcome","side"], as_index=False)["n_obs"].last()
              .pivot(index="outcome", columns="side", values="n_obs")
        )
    else:
        n_tbl = pd.DataFrame()

    # ---------------- 6) AE vs EFA per-block bootstrap p-values ----------------
    p_rows = []
    for y, ytype in validator_info.items():
        if y not in panel.columns:
            continue
        for i_blk in range(1, K+1):
            ae_cols_blk  = [c for c in ae_blocks[i_blk-1]  if c in panel_all.columns]
            efa_cols_blk = [c for c in efa_blocks[i_blk-1] if c in panel_all.columns]
            need_common_blk = sorted(set(ae_cols_blk) | set(efa_cols_blk))
            if not ae_cols_blk or not efa_cols_blk or not need_common_blk:
                p_val = np.nan
            else:
                p_val = _pvalue_compare_bootstrap(
                    panel_in=panel_all,
                    y=y,
                    ytype=ytype,
                    ae_cols=ae_cols_blk,
                    efa_cols=efa_cols_blk,
                    need_common_cols_this_block=need_common_blk,
                    B=compare_bootstrap_B,
                    seed=int(random_seed + 31*i_blk)
                )
            p_rows.append({"outcome": y, "block": i_blk, "p_value_compare": p_val})
    p_df = pd.DataFrame(p_rows)

    # Combine all results
    res_all = pd.concat([res_ae_poly, res_efa_poly], ignore_index=True) if not (res_ae_poly.empty and res_efa_poly.empty) else pd.DataFrame()
    if (res_all is not None and not res_all.empty
        and {"outcome","block"}.issubset(res_all.columns) and not p_df.empty):
        p_df["block"] = p_df["block"].astype(res_all["block"].dtype)
        res_all = res_all.merge(p_df, on=["outcome","block"], how="left")
    else:
        if "p_value_compare" not in res_all.columns:
            res_all["p_value_compare"] = np.nan

    # ---------------- 7) Plot grid ----------------
    if not res_all.empty:
        # outcomes = list(res_all["outcome"].unique())
        fixed_order = [
            "mh_service", "med_history", "brought_meds", "fes_conflict", 
            "school_conn", "fluid_cog", "cryst_cog", "avg_grades", 
            "dev_delay", "n_friends", 
              
        ]
        # 只保留 res_all 里实际有的 outcome
        outcomes = [o for o in fixed_order if o in res_all["outcome"].unique()]
        n = len(outcomes)
        n_col, n_row = (1,1) if n == 1 else (2, int(np.ceil(n/2)))
        fig, axes = plt.subplots(n_row, n_col, figsize=(n_col*4.8, n_row*3.3), sharex=False, sharey=False)
        axes = np.atleast_1d(axes)

        for idx, outcome in enumerate(outcomes):
            ax = axes.flat[idx]
            sub = res_all[res_all["outcome"] == outcome].sort_values(["model","block"])

            for m, g in sub.groupby("model"):
                ax.plot(g["block"], g["total_R2"], marker="o", lw=1.8, label=m)

            sub_wide = sub.pivot(index="block", columns="model", values="total_R2")
            p_by_block = (
                sub[sub["model"] == model_label_ae]
                .set_index("block")["p_value_compare"]
                .combine_first(sub.groupby("block")["p_value_compare"].first())
            )
            for blk, p in p_by_block.dropna().items():
                if blk in sub_wide.index and p < .05:
                    y_ae  = sub_wide.loc[blk, model_label_ae]  if model_label_ae  in sub_wide.columns else np.nan
                    y_efa = sub_wide.loc[blk, model_label_efa] if model_label_efa in sub_wide.columns else np.nan
                    y_mid = np.nanmean([y_ae, y_efa])
                    if np.isfinite(y_mid):
                        stars = "***" if p < .001 else "**" if p < .01 else "*"
                        ax.annotate(stars, (int(blk), y_mid), xytext=(0, 6),
                                    textcoords="offset points", ha="center", fontsize=8, weight="bold")
                        if np.isfinite(y_ae) and np.isfinite(y_efa):
                            ax.plot([blk, blk], [min(y_ae, y_efa), max(y_ae, y_efa)],
                                    linestyle=":", linewidth=.8, alpha=.6)

            finals = sub.groupby("model")["total_R2"].last().to_dict()
            nice = name_map.get(outcome, outcome)
            ae_final  = finals.get(model_label_ae,  np.nan)
            efa_final = finals.get(model_label_efa, np.nan)
            ttl = f"{nice}\n({model_label_ae}={ae_final:.3f}, {model_label_efa}={efa_final:.3f})" if (np.isfinite(ae_final) and np.isfinite(efa_final)) else nice
            ax.set_title(ttl, fontsize=9)
            if sub["block"].notna().any():
                try:
                    ax.set_xticks(range(1, int(sub["block"].max())+1))
                except Exception:
                    pass
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=1))    

            if idx == 0:
                ax.legend(frameon=False, fontsize=8)

        total_axes = axes.size if hasattr(axes, 'size') else len(axes.flat)
        for j in range(len(outcomes), total_axes):
            fig.delaxes(axes.flat[j])

        fig.text(0.5, 0.04, "Number of factors (cumulative)", ha="center", fontsize=12)
        fig.text(0.06, 0.5, "Total R² / Nagelkerke R²", va="center", rotation="vertical", fontsize=12)
        plt.tight_layout(rect=[0.06, 0.04, 1, 1])
        out_png = os.path.join(save_dir, "efa_vs_ae_poly_grid.png")
        plt.savefig(out_png, dpi=300)
        plt.close(fig)
    else:
        out_png = os.path.join(save_dir, "efa_vs_ae_poly_grid.png")
        plt.figure(figsize=(6, 4))
        plt.title("No valid outcomes")
        plt.savefig(out_png, dpi=300)
        plt.close()

    # ---------------- 8) Summary table ----------------
    if not res_all.empty:
        final_tbl = (
            res_all.sort_values(["outcome","model","block"])
                   .groupby(["outcome","model"], as_index=False)["total_R2"].last()
                   .pivot(index="outcome", columns="model", values="total_R2")
                   .rename_axis(None, axis=1)
        )
        final_tbl = final_tbl.assign(
            pretty_name=[name_map.get(o, o) for o in final_tbl.index]
        ).set_index("pretty_name", append=False)

        if model_label_ae in final_tbl.columns and model_label_efa in final_tbl.columns:
            final_tbl["diff (AE - EFA)"] = final_tbl.get(model_label_ae) - final_tbl.get(model_label_efa)
            efa_vals = final_tbl.get(model_label_efa).astype(float)
            with np.errstate(divide='ignore', invalid='ignore'):
                final_tbl["improve % (AE/EFA - 1)"] = np.where(
                    efa_vals.abs() > 1e-8,
                    final_tbl.get(model_label_ae).astype(float) / efa_vals - 1,
                    np.nan
                )
    else:
        final_tbl = pd.DataFrame()

    out_csv = os.path.join(save_dir, "efa_vs_ae_poly_summary.csv")
    final_tbl.to_csv(out_csv, float_format="%.6f")

    # 友好打印（不过度 dropna）
    cols_view = ["outcome","block","model","total_R2","p_value_compare"]
    to_view = res_all[cols_view].copy() if not res_all.empty else pd.DataFrame(columns=cols_view)
    mask = to_view[["outcome","block","model"]].notna().all(axis=1)
    print(to_view[mask].head(20))

    return {
        "res_ae": res_ae_poly,
        "res_efa": res_efa_poly,
        "summary": final_tbl,
        "n_table": n_tbl,
        "paths": {
            "res_ae_csv": res_ae_csv,
            "res_efa_csv": res_efa_csv,
            "summary_csv": out_csv,
            "grid_png": out_png
        }
    }
