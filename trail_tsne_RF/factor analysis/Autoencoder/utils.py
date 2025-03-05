from bs4 import BeautifulSoup
# from googletrans import Translator
import re
import time
import pandas as pd
from deep_translator import GoogleTranslator
import cx_Oracle
import numpy as np
import os
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, scale
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_predict
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import r2_score, mean_squared_error, explained_variance_score
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from scipy.stats.distributions import halfcauchy
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from factor_analyzer import FactorAnalyzer
import seaborn as sns
import matplotlib.pyplot as plt
import torch.onnx
import netron
import matplotlib
#--------------------------------------------------------------------------------------------------

def get_cbcl_details(cbcl_item):
    """
    根据提供的 cbcl_q 字段组合（如 "avg_cbcl_q08_p_cbcl_q10_p_cbcl_q78_p"）从 element.html 文件中获取详细信息。
    
    参数:
        cbcl_item (str): 要查找的 cbcl_q 字段组合（如 "avg_cbcl_q08_p_cbcl_q10_p_cbcl_q78_p"）。
    
    返回:
        str: 详细信息的组合，如果找不到则返回 "N/A"。
    """
    # 解析 element.html 文件
    with open(r"G:\ABCD\script\trail\trail_tsne_RF\factor analysis\data\element.html", "r", encoding="utf-8") as file:
        soup = BeautifulSoup(file, "html.parser")

    # 使用正则表达式提取所有的 cbcl_q 字段
    cbcl_pattern = re.compile(r"(cbcl_q\d+[a-z]*_p)")
    cbcl_items = cbcl_pattern.findall(cbcl_item)
    
    # 存储每个 cbcl 字段的详细信息
    details = []

    for cbcl in cbcl_items:
        # 在 HTML 中查找包含 cbcl 的 <td> 标签
        target = soup.find(lambda tag: tag.name == "td" and cbcl in tag.get_text(strip=True))
        
        # 获取详细信息
        if target:
            detail_info = target.find_next("td").get_text(strip=True)
            details.append(detail_info)
        else:
            details.append("N/A")
    
    # 合并所有详细信息为一个字符串
    combined_details = "; ".join(details) if details else "N/A"

    return combined_details

# 示例调用
detail = get_cbcl_details("avg_cbcl_q08_p_cbcl_q10_p_cbcl_q78_p")
print("详细信息:", detail)



#--------------------------------------------------------------------------------------------------
""" 输入数据框DF(形如Row_Name1,Row_Name2),因子数量和语言(string), 返回一个包含翻译后详细信息的数据框"""

def translate_text(df, number_of_factors,language):

    # 解析 element.html 文件以获取列名和详细信息
    with open(r"G:\ABCD\script\trail\trail_tsne_RF\factor analysis\data\element.html", "r", encoding="utf-8") as file:
        soup = BeautifulSoup(file, "html.parser")

    # 创建一个字典来存储列名和对应的详细信息
    column_details = {}
    result_df = pd.DataFrame()

    # 提取 cbcl_q 列名的正则表达式
    cbcl_pattern = re.compile(r"(cbcl_q\d+[a-z]*_p)")

    for i in range(0, number_of_factors):
        # 筛选出符合条件的加载值
        # factor_values = df[f"Factor {i}"][df[f"Factor {i}"] > 0.1]
        
        original_text = []
        translated_text = []
        for column_name in df.iloc[:,i]:
            # 查找 column_name 中的所有 cbcl_q 字段
            cbcl_items = cbcl_pattern.findall(column_name)  # 提取所有符合 cbcl_qXX_p 或 cbcl_qXXh_p 格式的子串

            # 初始化存储每个 cbcl 字段详细信息的列表
            original = []
            details = []
            for cbcl_item in cbcl_items:
                # 获取每个 cbcl 字段的详细信息
                target = soup.find(lambda tag: tag.name == "td" and cbcl_item in tag.get_text(strip=True))
                if target:
                    detail_info = target.find_next("td").get_text(strip=True)
                    # 保存原始详细信息
                    original.append(detail_info)
                    
                    # 翻译详细信息并添加到结果
                    try:
                        translated_detail = GoogleTranslator(source='es', target=language).translate(detail_info)
                    except AttributeError as e:
                        print(f"An error occurred: {e}")
                        translated_detail = detail_info
                    details.append(translated_detail)
                    time.sleep(0.25)

            # 将所有细节合并为单个字符串，并添加到列表中
            original_text.append("; ".join(original) if original else "N/A")
            translated_text.append("; ".join(details) if details else "N/A")
        # 创建一个临时数据框保存因子名、列名、加载值和详细信息
        temp_df = pd.DataFrame({
            # f"Factor {i} Variable": factor_values.index,  # 存储列名
            # f"Factor {i} Loading": factor_values.values,  # 存储加载值
            f"Factor {i} Detail": original_text,  # 映射详细信息
            f"Factor {i} Translated_Detail": translated_text  # 映射翻译后详细信息
        })

        # 按加载值降序排序
        # sorted_df = temp_df.sort_values(by=f"Factor {i} Loading", ascending=False).reset_index(drop=True)
        # 将临时数据框合并到结果数据框
        result_df = pd.concat([result_df.reset_index(drop=True), temp_df.reset_index(drop=True)], axis=1)
    return result_df


# Get the raw fMRI data with nda-tool; after create package in NDA
class GetfMRIdata:
    def __init__(self, package_id, password):
        user = f"k21116947_{package_id}"
        dsn = cx_Oracle.makedsn("mindarvpc.cqahbwk3l1mb.us-east-1.rds.amazonaws.com", 1521, service_name="ORCL")
        self.conn = cx_Oracle.connect(user=user, password=password, dsn=dsn)
        self.s3_samples = []

    def fetch_data(self):
        cursor = self.conn.cursor()
        query = """
        SELECT ENDPOINT
        FROM S3_LINKS
        WHERE ENDPOINT LIKE '%baseline%' AND ENDPOINT LIKE '%rsfMRI%' AND ENDPOINT LIKE '%NDARINV005V6D2C%' AND ENDPOINT LIKE '%MPROC%' 
        """
        cursor.execute(query)
        self.s3_samples = [row[0] for row in cursor.fetchall()]
        cursor.close()

    def save_data(self):
        if not self.s3_samples:
            self.fetch_data()
        np.savetxt('data/s3_links.txt', self.s3_samples, fmt='%s')
        # Assuming `downloadcmd` is a command-line tool you want to run
        try:
            subprocess.run(['downloadcmd', '-dp', '1236370', '-t', 'data/s3_links.txt', '-d', './data/fMRI_data'], check=True)
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while running downloadcmd: {e}")

    def close_connection(self):
        self.conn.close()

    def run_all(self):
        self.fetch_data()
        self.save_data()
        self.close_connection()




# Autoencoder
import matplotlib.pyplot as plt

class Autoencoder:
    """
    function: Autoencoder:
    - This class implements an autoencoder using PyTorch. The autoencoder is trained on the input data and the latent features are extracted.

    args:
    - input_dim: int, Number of features in the input data
    - encoding_dim: int, Number of neurons in the latent layer
    - X_train: np.ndarray, Training data
    - X_val: np.ndarray, Validation data

    return:
    - latent_df: pd.DataFrame, DataFrame containing the latent features
    - reconstruction_errors: np.ndarray, Reconstruction errors for each sample
    - explained_variance_ratio: float, Explained variance ratio

    """
    def __init__(self, X_train, X_val, encoding_dim):
        # Create PyTorch datasets and dataloaders
        class QuestionnaireDataset(Dataset):
            def __init__(self, data):
                self.data = torch.tensor(data, dtype=torch.float32)
                
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                return self.data[idx], self.data[idx]  # Input and target are the same

        train_dataset = QuestionnaireDataset(X_train)
        val_dataset = QuestionnaireDataset(X_val)

        self.train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=32)

        layer1_neurons, layer2_neurons, layer3_neurons = 19, 79, 75

        # Step 2: Define the autoencoder architecture
        class AutoencoderModel(nn.Module):
            def __init__(self, input_dim, latent_dim, layer1_neurons=layer1_neurons, layer2_neurons=layer2_neurons, layer3_neurons=layer3_neurons):
                super(AutoencoderModel, self).__init__()
                
                # Encoder
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, layer1_neurons),
                    nn.LeakyReLU(negative_slope=0.01),
                    nn.Linear(layer1_neurons, layer2_neurons),
                    nn.LeakyReLU(negative_slope=0.01),
                    nn.Linear(layer2_neurons, layer3_neurons),
                    nn.LeakyReLU(negative_slope=0.01),
                    nn.Linear(layer3_neurons, latent_dim)
                )
                
                # Decoder
                self.decoder = nn.Sequential(
                    nn.Linear(latent_dim, layer3_neurons),
                    nn.LeakyReLU(negative_slope=0.01),
                    nn.Linear(layer3_neurons, layer2_neurons),
                    nn.LeakyReLU(negative_slope=0.01),
                    nn.Linear(layer2_neurons, layer1_neurons),
                    nn.LeakyReLU(negative_slope=0.01),
                    nn.Linear(layer1_neurons, input_dim),
                )
                
            def forward(self, x):
                # forward
                encoded = self.encoder(x)
                decoded = self.decoder(encoded)
                return decoded

        # Model initialization
        input_dim = X_train.shape[1]
        latent_dim = encoding_dim
        self.model = AutoencoderModel(input_dim, latent_dim)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=5)

    def train(self):
        best_val_loss = float('inf')
        patience = 10
        epochs_without_improvement = 0
        train_losses, val_losses = [], []

        for epoch in range(2000):
            self.model.train()
            train_loss = 0
            for batch_features, _ in self.train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_features)
                loss = self.criterion(outputs, batch_features)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item() * batch_features.size(0)
            train_losses.append(train_loss / len(self.train_loader.dataset))

            # Validation step
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_features, _ in self.val_loader:
                    outputs = self.model(batch_features)
                    loss = self.criterion(outputs, batch_features)
                    val_loss += loss.item() * batch_features.size(0)
            val_losses.append(val_loss / len(self.val_loader.dataset))
            self.scheduler.step(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        # Plot loss curves
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Curves')
        plt.show()

    # def get_latent_features(self):
    #     self.model.eval()
    #     with torch.no_grad():
    #         X_latent = self.model.encoder(torch.tensor(X_train, dtype=torch.float32)).numpy()
    #     return X_latent

    # def get_reconstruction_errors(self):
    #     self.model.eval()
    #     with torch.no_grad():
    #         reconstructed = self.model(torch.tensor(X_train, dtype=torch.float32))
    #         reconstruction_errors = torch.mean((torch.tensor(X_train, dtype=torch.float32) - reconstructed) ** 2, dim=1).numpy()
    #     return reconstruction_errors

    # def get_explained_variance_ratio(self):
    #     total_variance = np.var(X_train, axis=0).sum()
    #     self.model.eval()
    #     with torch.no_grad():
    #         reconstructed = self.model(torch.tensor(X_train, dtype=torch.float32))
    #         reconstruction_variance = np.var(reconstructed.numpy(), axis=0).sum()
    #     explained_variance_ratio = reconstruction_variance / total_variance
    #     return explained_variance_ratio

    def evaluate_on_data(self, X_scaled):
        self.model.eval()
        with torch.no_grad():
            # Forward pass to get reconstructed data
            reconstructed = self.model(torch.tensor(X_scaled, dtype=torch.float32))
            
            # Get the output of the encoder, i.e., latent factors
            latent_factors = self.model.encoder(torch.tensor(X_scaled, dtype=torch.float32)).numpy()
            
            # Calculate the variance of each latent factor
            latent_variances = np.var(latent_factors, axis=0)
            
            # Calculate the total variance of the original data
            total_variance = np.var(X_scaled, axis=0).sum()
            
            # Calculate the explained variance ratio for each latent factor
            explained_variance_ratios = latent_variances / total_variance
            
            # Print the explained variance ratio for each latent factor
            for i, ratio in enumerate(explained_variance_ratios):
                print(f"Explained variance ratio of latent factor {i+1}: {ratio:.8f}")
            
            # Calculate reconstruction errors
            reconstruction_errors = torch.mean((torch.tensor(X_scaled, dtype=torch.float32) - reconstructed) ** 2, dim=1).numpy()
            
            # Calculate the variance contribution of the reconstructed data
            reconstruction_variance = np.var(reconstructed.numpy(), axis=0).sum()
            
            # Calculate the total explained variance ratio
            explained_variance_ratio_total = reconstruction_variance / total_variance
            print(f"Total explained variance ratio (by all factors): {explained_variance_ratio_total:.8f}")
        
        return latent_factors, reconstruction_errors, explained_variance_ratios, explained_variance_ratio_total
    
    def plot_reconstruction_errors(self, datasets):
        self.model.eval()
        with torch.no_grad():
            for name, dataset in datasets.items():
                # Calculate reconstruction errors
                reconstructed = self.model(torch.tensor(dataset, dtype=torch.float32))
                reconstruction_errors = torch.mean((torch.tensor(dataset, dtype=torch.float32) - reconstructed) ** 2, dim=1).numpy()

                # Calculate quartiles
                q1 = np.percentile(reconstruction_errors, 25)
                q3 = np.percentile(reconstruction_errors, 75)
                iqr = q3 - q1

                # Define a range to identify outliers (1.5 times the IQR)
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr

                # Filter out outliers
                filtered_errors = reconstruction_errors[(reconstruction_errors >= lower_bound) & (reconstruction_errors <= upper_bound)]

                # Plot the distribution of reconstruction errors
                sns.histplot(filtered_errors, kde=True)
                plt.xlabel('Reconstruction Error')
                plt.title(f'Distribution of Reconstruction Errors ({name} dataset Without Outliers)')
                plt.show()

    def export_to_onnx(self, X_train, onnx_path="autoencoder_real_input.onnx"):
        # Use the first training sample as dummy input
        dummy_input = torch.tensor(X_train[:1], dtype=torch.float32)
        
        # Export the model to ONNX format
        torch.onnx.export(
            self.model,                          # Trained model
            dummy_input,                         # Use real data as example input
            onnx_path,                           # Output file path
            input_names=["input"],               # Input name
            output_names=["reconstructed"],      # Output name
            dynamic_axes={"input": {0: "batch_size"}, "reconstructed": {0: "batch_size"}},  # Dynamic batch size support
            opset_version=11                     # ONNX opset version
        )
        
        print(f"Model exported to {onnx_path}")
        
        # Start Netron to visualize the model
        netron.start(onnx_path)


# preict the generated factors with the original factors
class LassoAnalysis:
    '''
    function: preict the generated factors with the original factors

    args:
    - qns: np.ndarray, Questionnaire data
    - scores: np.ndarray, Factor scores
    - alpha_values: list, List of alpha values for Lasso regularization

    return:
    - pictures: plot the R^2 values for each factor
    '''
    def __init__(self, qns, scores, alpha_values=None):
        self.qns = qns
        self.scores = scores
        self.alpha_values = alpha_values if alpha_values is not None else [0.001, 0.01, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        self.r2_values = np.empty((5, len(self.alpha_values)))
        self.n_items = np.empty(len(self.alpha_values))
        self.pal = ['#4f4f4f', '#B80044', '#0e79b2', '#f9a800', '#00a087']
        self.prop = matplotlib.font_manager.FontProperties(fname="c:\\windows\\fonts\\nunitosans-light.ttf")
        matplotlib.rcParams['font.weight'] = 'light'
        matplotlib.rcParams['axes.facecolor'] = '#fbfbfb'

    def perform_analysis(self):
        for n, alpha in enumerate(tqdm(self.alpha_values)):
            clf = Lasso(alpha=alpha)
            clf.fit(self.qns, self.scores)
            pred = cross_val_predict(clf, self.qns, self.scores, cv=5)
            for i in range(5):
                self.r2_values[i, n] = r2_score(self.scores.iloc[:, i], pred[:, i])
            self.n_items[n] = np.any(clf.coef_.T != 0, axis=1).sum()

    def plot_r2_values(self):
        f, ax = plt.subplots(dpi=100, facecolor='white')
        for i in range(5):
            ax.plot(self.n_items, self.r2_values[i, :], label='Factor {0}'.format(i+1), color=self.pal[i])
        ax.set_xlabel("Number of items")
        ax.set_ylabel("$R^2$")
        ax.legend()
        ax2 = ax.twiny()
        ax2.set_xticklabels(self.alpha_values)
        ax2.set_xticks(self.n_items)
        ax.axvline(63, color='#8c8a8a', linestyle=':')
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=90)
        plt.tight_layout()
        plt.show()

    def plot_factor_predictions(self, alpha=0.125):
        clf = Lasso(alpha=alpha)
        pred = cross_val_predict(clf, self.qns, self.scores, cv=5)
        clf.fit(self.qns, self.scores)
        f, ax = plt.subplots(1, 5, figsize=(16, 3.5), dpi=100, facecolor='white')
        factors = ['V1', 'V2', 'V3', 'V4', 'V5']
        for i in range(5):
            sns.regplot(x=self.scores.iloc[:, i], y=pred[:, i], ax=ax[i], color=self.pal[i], scatter_kws={'alpha': 0.5})
            ax[i].set_title(factors[i] + '\n$R^2$ = {0}'.format(np.round(r2_score(self.scores.iloc[:, i], pred[:, i]), 5)), fontweight='light')
            ax[i].set_xlabel('True score')
            ax[i].set_ylabel('Predicted score')
        plt.tight_layout()
        plt.show()

    def plot_heatmap(self, alpha=0.125):
        clf = Lasso(alpha=alpha)
        clf.fit(self.qns, self.scores)
        plt.figure(dpi=100, figsize=(9, 1.5), facecolor='white')
        sns.heatmap(clf.coef_, cmap='Blues', yticklabels=['V1', 'V2', 'V3', 'V4', 'V5'])
        plt.xlabel("Question number")
        plt.ylabel("Factor")
        plt.tight_layout()
        plt.show()