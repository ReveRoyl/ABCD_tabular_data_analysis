# 加载 psych 包
# install.packages("psych")
library(psych)

# 设置数据路径并加载数据
data_path <- "data"
data <- read.csv(file.path(data_path, "labels.csv"))
# data_path <- "G:/ABCD/script/trail/trail_tsne_RF/test"
# data <- read.csv(file.path(data_path, "all_year_labels.csv"))
setwd(file.path(data_path, "/output/"))

# 删除指定列
# data <- data[, !(names(data) %in% c("Unnamed: 0", "src_subject_id"))]

# 选择以 "cbcl" 开头的列
label_columns <- c("src_subject_id", grep("^cbcl", names(data), value = TRUE))
data <- data[, label_columns]

# 假设数据框的第一列是 ID 列，我们将其单独保留出来
id_column <- data[, 1, drop = FALSE]  # 保留原始的第一列 (ID 列)
data_to_clean <- data[, -1]  # 提取从第二列开始的数据进行清理

# Step 2: 找到并移除低频项（列）
# 移除低频率（超过 99.5% 为 0）的列
low_frequency_columns <- names(data_to_clean)[sapply(data_to_clean, function(col) mean(col == 0) > 0.995)]
data_cleaned <- data_to_clean[, !(names(data_to_clean) %in% low_frequency_columns)]

# 输出被移除的列
cat("Removed columns with low frequency:", low_frequency_columns, "\n")

# Step 3: 找到并移除低频项（行）
# 移除低频率（超过 99.5% 为 0）的行
low_frequency_rows <- apply(data_cleaned, 1, function(row) mean(row == 0) > 0.995)
data_cleaned <- data_cleaned[!low_frequency_rows, ]

# 输出被移除的行
cat("Removed rows with low frequency:", which(low_frequency_rows), "\n")



# Step 2: 计算 Polychoric 相关矩阵
polychoric_corr <- polychoric(data_cleaned)$rho  # 使用 $rho 获取相关系数矩阵


# 保存 Polychoric 相关矩阵到 CSV 文件
write.csv(polychoric_corr, file = file.path(data_path, "correlation.csv"), row.names = TRUE)

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

# ---------------------------------------------------------------------------------------------------------------------------------------
### 对数据降维, 使用NMF ### 
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




# Step 4: 将原始的 ID 列合并回清理后的数据框，并确保行对齐
# 保留没有被移除行的 ID 列
id_column_cleaned <- id_column[!low_frequency_rows, , drop = FALSE]
data_cleaned <- cbind(id_column_cleaned, data_cleaned)

W <- cbind(id_column_cleaned, W)
H_transposed = t(H)
# write.csv(H_transposed, file = "nmf_H.csv", row.names = TRUE)
#save W and data_cleaned
# write.csv(W, file = "nmf_W.csv", row.names = TRUE)

# write.csv(data_cleaned, file = "data_cleaned.csv", row.names = TRUE)
