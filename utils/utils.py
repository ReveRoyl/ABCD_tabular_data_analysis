import os
import re
import subprocess
import time
import cx_Oracle
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from deep_translator import GoogleTranslator
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import textwrap
import torch
import torch.nn as nn
import shap
# --------------------------------------------------------------------------------------------------

def get_cbcl_details(cbcl_item):
    """
    Get detailed information from the element.html file based on the provided cbcl_q field combination (e.g., "avg_cbcl_q08_p_cbcl_q10_p_cbcl_q78_p").

    Parameters:
        cbcl_item (str): The cbcl_q field combination to look up (e.g., "avg_cbcl_q08_p_cbcl_q10_p_cbcl_q78_p").

    Returns:
        str: Combined detailed information, or "N/A" if not found.
    """
    # Parse the element.html file
    with open(
        r"../data/element.html",
        "r",
        encoding="utf-8",
    ) as file:
        soup = BeautifulSoup(file, "html.parser")

    # Use regular expressions to extract all cbcl_q fields
    cbcl_pattern = re.compile(r"(cbcl_q\d+[a-z]*_p)")
    cbcl_items = cbcl_pattern.findall(cbcl_item)

    # Store detailed information for each cbcl field
    details = []

    for cbcl in cbcl_items:
        # Find <td> tags in the HTML that contain the cbcl field
        target = soup.find(
            lambda tag: tag.name == "td" and cbcl in tag.get_text(strip=True)
        )

        # Get detailed information
        if target:
            detail_info = target.find_next("td").get_text(strip=True)
            details.append(detail_info)
        else:
            details.append("N/A")

    # Combine all detailed information into a single string
    combined_details = "; ".join(details) if details else "N/A"

    return combined_details


if __name__ == "__main__":
    # Example call
    detail = get_cbcl_details("avg_cbcl_q08_p_cbcl_q10_p_cbcl_q78_p")
    print("Detailed information:", detail)


# --------------------------------------------------------------------------------------------------


def translate_text(df, language):
    """ 
    Input DataFrame df (in the form of Row_Name1, Row_Name2, as the number of factors)and language (string)
    
    Returns: a DataFrame containing translated detailed information 
    """

    assert os.path.exists(
        '../data/element.html'
    ), "element directory not found. Make sure you're running this code from the root directory of the project."

    # Parse the element.html file to get column names and detailed information
    with open(
        r"../data/element.html",
        "r",
        encoding="utf-8",
    ) as file:
        soup = BeautifulSoup(file, "html.parser")

    # Create a dictionary to store column names and corresponding detailed information
    result_df = pd.DataFrame()

    # Regular expression to extract cbcl_q column names
    cbcl_pattern = re.compile(r"(cbcl_q\d+[a-z]*_p)")

    for i in range(0, len(df.columns)):
        # Filter out loading values that meet the criteria
        # factor_values = df[f"Factor {i}"][df[f"Factor {i}"] > 0.1]

        original_text = []
        translated_text = []
        for column_name in df.iloc[:, i]:
            # Find all cbcl_q fields in the column_name
            cbcl_items = cbcl_pattern.findall(
                column_name
            )  # Extract all substrings that match the cbcl_qXX_p or cbcl_qXXh_p format

            # Initialize a list to store detailed information for each cbcl field
            original = []
            details = []
            for cbcl_item in cbcl_items:
                # Get detailed information for each cbcl field
                target = soup.find(
                    lambda tag: tag.name == "td"
                    and cbcl_item in tag.get_text(strip=True)
                )
                if target:
                    detail_info = target.find_next("td").get_text(strip=True)
                    # Save original detailed information
                    original.append(detail_info)

                    # Translate detailed information and add to the result
                    try:
                        translated_detail = GoogleTranslator(
                            source="es", target=language
                        ).translate(detail_info)
                    except AttributeError as e:
                        print(f"An error occurred: {e}")
                        translated_detail = detail_info
                    details.append(translated_detail)
                    time.sleep(0.25)

            # Combine all details into a single string and add to the list
            original_text.append("; ".join(original) if original else "N/A")
            translated_text.append("; ".join(details) if details else "N/A")
        # Create a temporary DataFrame to save factor names, column names, loading values, and detailed information
        temp_df = pd.DataFrame(
            {
                # f"Factor {i} Variable": factor_values.index,  # Store column names
                # f"Factor {i} Loading": factor_values.values,  # Store loading values
                f"Factor {i} Detail": original_text,  # Map detailed information
                f"Factor {i} Translated_Detail": translated_text,  # Map translated detailed information
            }
        )

        # Sort by loading values in descending order
        # sorted_df = temp_df.sort_values(by=f"Factor {i} Loading", ascending=False).reset_index(drop=True)
        # Merge the temporary DataFrame into the result DataFrame
        result_df = pd.concat(
            [result_df.reset_index(drop=True), temp_df.reset_index(drop=True)], axis=1
        )
    return result_df


# Get the raw fMRI data with nda-tool; after creating package in NDA
class GetfMRIdata:
    def __init__(self, package_id, password):
        user = f"k21116947_{package_id}"
        dsn = cx_Oracle.makedsn(
            "mindarvpc.cqahbwk3l1mb.us-east-1.rds.amazonaws.com",
            1521,
            service_name="ORCL",
        )
        self.conn = cx_Oracle.connect(user=user, password=password, dsn=dsn)
        self.s3_samples = []

    def fetch_data(self):
        cursor = self.conn.cursor()
        query = """
        SELECT ENDPOINT
        FROM S3_LINKS
        WHERE ENDPOINT LIKE '%baseline%' AND (ENDPOINT LIKE '%rsfMRI%' OR ENDPOINT LIKE '%T1%') AND ENDPOINT LIKE '%MPROC%' 
        --AND ENDPOINT LIKE '%NDARINV05ATJ1V1%'
        """
        cursor.execute(query)
        self.s3_samples = [row[0] for row in cursor.fetchall()]
        cursor.close()

    def save_data(self):
        if not self.s3_samples:
            self.fetch_data()
        np.savetxt("data/s3_links.txt", self.s3_samples, fmt="%s")
        # Assuming `downloadcmd` is a command-line tool you want to run
        try:
            subprocess.run(
                [
                    "downloadcmd",
                    "-dp",
                    "1236370",
                    "-t",
                    "data/s3_links.txt",
                    "-d",
                    "./data/fMRI_data",
                ],
                check=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while running downloadcmd: {e}")

    def close_connection(self):
        self.conn.close()

    def run_all(self):
        self.fetch_data()
        self.save_data()
        self.close_connection()

def find_column_in_csvs(root_folder, target_column, case_insensitive=True, verbose=True):
    """
    Search for CSV files under a root directory that contain a specific column name.

    Parameters:
    - root_folder (str): Root folder path containing CSV files
    - target_column (str): The column name or substring to search for
    - case_insensitive (bool): Whether to ignore case when matching (default: True)
    - verbose (bool): Whether to print the results (default: True)

    Returns:
    - found_files (list of tuples): List of (file_path, matching_column)
    """
    found_files = []

    for dirpath, dirnames, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.lower().endswith('.csv'):
                file_path = os.path.join(dirpath, filename)
                try:
                    df = pd.read_csv(file_path, nrows=1)
                    for col in df.columns:
                        col_check = col.lower() if case_insensitive else col
                        target_check = target_column.lower() if case_insensitive else target_column
                        if target_check in col_check:
                            found_files.append((file_path, col))
                except Exception as e:
                    if verbose:
                        print(f'⚠️ Failed to read file {file_path}: {e}')
    
    if verbose:
        if found_files:
            for path, column in found_files:
                print(f'✅ Found column "{column}" in file: {path}')
        else:
            print(f'❌ Column "{target_column}" was not found in any file.')

    return found_files



def wrap_labels(labels, width=20):
    return [textwrap.fill(label, width) for label in labels]

def compute_autoencoder_loadings_with_plot(latent_factors, X_train, items, top_k=8):
    """
    Compute the autoencoder "loading matrix" and visualize, for each latent factor,
    the top_k original features with the largest absolute loadings.

    Parameters
    ----------
    latent_factors : np.ndarray or pd.DataFrame
        Latent factor representations produced by the autoencoder encoder,
        shape (n_samples, n_latent_factors).
    X_train : np.ndarray or pd.DataFrame
        Original input features, shape (n_samples, n_original_features).
    items : list or array-like
        Names of the original features; length must match X_train.shape[1].
    top_k : int, default=8
        Number of top original features (by absolute loading) to visualize per latent factor.

    Returns
    -------
    loadings_df : pd.DataFrame
        Loading matrix as a DataFrame, with rows corresponding to original features
        and columns corresponding to latent factors.
    """

    # 转换为 numpy 数组
    latent_factors = (
        latent_factors.values if isinstance(latent_factors, pd.DataFrame) else latent_factors
    )
    original_features = (
        X_train.values if isinstance(X_train, pd.DataFrame) else X_train
    )

    n_original_features = original_features.shape[1]
    n_latent_factors = latent_factors.shape[1]

    # 标准化 latent_factors
    scaler = StandardScaler()
    latent_factors_scaled = scaler.fit_transform(latent_factors)

    loadings = []
    # 对每个原始特征回归
    for i in range(n_original_features):
        y = original_features[:, i]
        reg = LinearRegression().fit(latent_factors_scaled, y)
        loadings.append(reg.coef_)

    loadings_df = pd.DataFrame(
        loadings,
        index=items,
        columns=[f"Latent_{j+1}" for j in range(n_latent_factors)]
    )

    # ==== 可视化 ====
    sns.set(style='whitegrid')
    for col in loadings_df.columns:
        # 取绝对值 top_k 特征
        top_items = loadings_df[col].abs().sort_values(ascending=False).head(top_k).index
        top_data = loadings_df.loc[top_items, [col]]
        top_items_wrapped = wrap_labels(top_items, width=100)  # each label max 80 chars per line
        # 绘制热力图
        plt.figure(figsize=(6, 0.5*len(top_items_wrapped)))
        sns.heatmap(top_data, annot=True, cmap='coolwarm', center=0, cbar=True, yticklabels=top_items_wrapped)
        plt.title(f"Top {top_k} Loadings for {col}")
        plt.xlabel("Latent Dimension")
        plt.ylabel("Original Feature")
        plt.tight_layout()
        plt.show()

    return loadings_df

class DecoderHead(nn.Module):
    def __init__(self, decoder: nn.Module, out_idx: int = 0):
        super().__init__()
        self.decoder = decoder
        self.out_idx = out_idx
    def forward(self, z):
        xhat = self.decoder(z)                       # (batch, n_items)
        return xhat[:, self.out_idx:self.out_idx+1]  # (batch, 1) 单输出更稳

def compute_shap_loadings_decoder_only(
    decoder: nn.Module,
    Z,                       # (n_samples, n_latent) numpy / torch / pandas
    items,                   # 题项名，len == n_items
    device="cuda",
    background_size=32,
    sample_size=200,
    nsamples=16,             # 显式缩小 nsamples（默认~100-200很慢）
    eval_batch=128,          # 对 Z_eval 分批
    top_k=8,
    plot=True,
    freeze_decoder=True,
    seed=6,
):
    # ---- 设备与模型 ----
    use_cuda = (device == "cuda") and torch.cuda.is_available()
    dev = torch.device("cuda" if use_cuda else "cpu")
    dec = decoder.eval().to(dev)
    if freeze_decoder:
        for p in dec.parameters():
            p.requires_grad_(False)

    # ---- 准备输入数据 ----
    if isinstance(Z, torch.Tensor):
        Z_np = Z.detach().cpu().numpy()
    elif isinstance(Z, pd.DataFrame):
        Z_np = Z.values
    else:
        Z_np = np.asarray(Z)
    assert Z_np.ndim == 2, f"Z must be 2D, got shape {Z_np.shape}"
    n_all, n_latent = Z_np.shape

    rng = np.random.default_rng(seed)
    bg_idx = rng.choice(n_all, size=min(background_size, n_all), replace=False)
    ev_idx = rng.choice(n_all, size=min(sample_size,   n_all), replace=False)

    Z_bg   = torch.from_numpy(Z_np[bg_idx]).to(dev, dtype=torch.float32)
    Z_eval = torch.from_numpy(Z_np[ev_idx]).to(dev, dtype=torch.float32)

    # 输出维度（题项数）
    with torch.no_grad():
        n_items = dec(Z_eval[:1]).shape[1]
    item_index = list(items)
    if len(item_index) != n_items:
        # 保底：长度不一致时生成占位名以避免报错
        item_index = [str(s) for s in item_index[:n_items]] + \
                     [f"item_{i}" for i in range(len(item_index), n_items)]

    # ---- 复用一个 head + explainer ----
    head = DecoderHead(dec, out_idx=0).eval().to(dev)
    explainer = shap.GradientExplainer(head, Z_bg)

    load_signed = np.zeros((n_items, n_latent), dtype=np.float32)
    # 假设已得到 R 返回的旋转矩阵/载荷


    strength    = np.zeros((n_items, n_latent), dtype=np.float32)

    # ---- 逐题项，但不重复创建 explainer；对 Z_eval 分批 ----
    for i in range(n_items):
        head.out_idx = i
        parts = []
        for chunk in torch.split(Z_eval, eval_batch):
            sv = explainer.shap_values(chunk, nsamples=nsamples)
            # 统一形状为 (m_eval, n_latent)
            if isinstance(sv, list):
                sv = sv[0]
            sv = np.asarray(sv)
            if sv.ndim == 3:
                if sv.shape[-1] == 1:      # (m_eval, n_latent, 1)
                    sv = sv[..., 0]
                elif sv.shape[0] == 1:     # (1, m_eval, n_latent)
                    sv = sv[0]
                elif sv.shape[1] == 1:     # (m_eval, 1, n_latent)
                    sv = sv[:, 0, :]
                else:                      # 多输出，取第一个
                    sv = sv[..., 0]
            elif sv.ndim == 1:
                sv = sv[None, :]
            parts.append(sv)
        sv = np.concatenate(parts, axis=0)        # (m_eval, n_latent)
        load_signed[i, :] = sv.mean(axis=0)       # 带符号平均
        strength[i,   :] = np.abs(sv).mean(axis=0)# 强度平均

    # ---- 构造 DataFrame（先构造再绘图，避免 NameError）----
    cols = [f"Latent_{j+1}" for j in range(n_latent)]
    load_signed_df = pd.DataFrame(load_signed, index=item_index, columns=cols)
    strength_df    = pd.DataFrame(strength,    index=item_index, columns=cols)

    # ---- 可视化（按强度挑 top-k）----
    if plot:
        for col in cols:
            k = min(top_k, len(item_index))
            top_idx = strength_df[col].nlargest(k).index
            mat = load_signed_df.loc[top_idx, [col]].sort_values(col, key=np.abs, ascending=False)

            plt.figure(figsize=(6, 0.45 * len(top_idx)))
            # 也可以用条形图，这里保留你原本的 heatmap 风格
            plt.imshow(
                mat.values, aspect="auto", interpolation="nearest",
                cmap="coolwarm",
                vmin=-np.max(np.abs(mat.values)),
                vmax=np.max(np.abs(mat.values)),
            )
            plt.colorbar(label="mean SHAP (signed)")
            plt.yticks(range(len(top_idx)), top_idx)
            plt.xticks([0], [col])
            plt.title(f"Top {len(top_idx)} SHAP-based Loadings for {col}")
            plt.tight_layout()
            plt.show()

    return load_signed_df, strength_df

def check_reconstruction(X_test, reconstructed, qns, vals=(0.0, 0.5, 1.0), atol=1e-8):
    """
    Check the average reconstructed values for given discrete inputs.

    Parameters
    ----------
    X_test : np.ndarray
        Original test data of shape (n_samples, n_features).
    reconstructed : np.ndarray
        Reconstructed data of shape (n_samples, n_features).
    qns : pd.DataFrame
        Questionnaire DataFrame (assumes first column is ID, features start from column index 1).
    vals : tuple
        Values to check, e.g. (0.0, 0.5, 1.0).
    atol : float
        Tolerance for value matching in np.isclose.

    Returns
    -------
    results : list of tuples
        Each tuple is (column_name, value, mean_reconstruction, count).
    """
    results = []
    for j in range(X_test.shape[1]):
        colname = qns.columns[j+1]  # offset by 1 if qns has ID column
        for v in vals:
            mask = np.isclose(X_test[:, j], v, atol=atol)
            if mask.any():
                m = reconstructed[mask, j].mean()
                n = mask.sum()
                print(f"{colname}: {v} -> {m:.3f} (n={n})")
                results.append((colname, v, m, n))
    return results
