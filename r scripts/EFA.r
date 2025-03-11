library(psych)
setwd('G:\\ABCD\\script\\trail\\data')
# 设置数据路径并加载数据
data_path <- file.path(getwd(), "..", "data")
data <- read.csv(file.path(data_path, "/cbcl_data_remove_unrelated.csv"))
#first column is ID
data_cleaned <- data[, -1]
id <- data[, 1]
# Step 2: 计算 Polychoric 相关矩阵
polychoric_corr <- polychoric(data_cleaned)$rho  # 使用 $rho 获取相关系数矩阵

# 保存 Polychoric 相关矩阵到 CSV 文件
write.csv(polychoric_corr, file = file.path(data_path, "correlation.csv"), row.names = TRUE)

# ------------------------------------------------------------------------------
### 对数据降维, 使用EFA ### \
# 进行因子分析
fa_result <- fa(data_cleaned, nfactors = 5, rotate = "geominQ", fm = "pa")

# 查看因子分析结果

print(fa_result)
fa_result$Vaccounted

# 提取每个个体的因子得分
factor_scores <- fa_result$scores

# 查看因子得分矩阵
head(factor_scores)  # 显示前几行的因子得分
dim(factor_scores)  # 查看因子得分矩阵的维度


id_column_cleaned <- id_column[!low_frequency_rows, , drop = FALSE]
factor_scores <- cbind(id_column_cleaned, factor_scores)
write.csv(factor_scores, file = "EFA.csv", row.names = TRUE)

# ------------------------------------------------------------------------------
### 对数据降维, 使用NMF
# 加载 NMF 包
library(NMF)
# 假设你有一个非负矩阵 data_matrix
# 进行 NMF 分解，设定要提取的成分数量为 5

nmf_result <- nmf(data_cleaned, rank = 5, method = "brunet", nrun = 30)

# 查看 NMF 结果
# print(nmf_result)

# 提取基矩阵 W
W <- basis(nmf_result)

# 提取系数矩阵 H
H <- coef(nmf_result)

# 查看 W 和 H
# print(W)
# print(H)
# 查看重构误差
print(paste("Reconstruction error:", nmf_result@residuals))

# 提取基矩阵和系数矩阵
W <- as.matrix(basis(nmf_result))
H <- as.matrix(coef(nmf_result))

# 执行矩阵乘法
X_reconstructed <- W %*% H

# 计算 Frobenius 范数
X_norm <- norm(data_cleaned, "F")
reconstruction_error <- norm(data_cleaned - X_reconstructed, "F")

# 计算相对误差
relative_error <- reconstruction_error / X_norm
print(paste("Relative Error:", relative_error))

V_hat <- W %*% H
# V_hat
# 计算原始数据矩阵 data_cleaned 的总方差
total_variance <- sum(sapply(data_cleaned, function(column) {
  sum((column - mean(column, na.rm = TRUE))^2, na.rm = TRUE)
}))
# total_variance
# 计算重构矩阵 V_hat 的方差
reconstructed_variance <- sum((V_hat - mean(V_hat))^2)

# 计算解释方差的比例
variance_explained <- reconstructed_variance / total_variance
cat("Variance explained by the NMF model:", variance_explained * 100, "%\n")


W <- cbind(id, W)
H_transposed = t(H)
# write.csv(H_transposed, file = "nmf_H.csv", row.names = TRUE)
#save W and data_cleaned
# write.csv(W, file = "nmf_W.csv", row.names = TRUE)