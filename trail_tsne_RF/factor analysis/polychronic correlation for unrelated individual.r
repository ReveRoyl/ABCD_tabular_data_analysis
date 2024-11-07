# 加载 psych 包
# install.packages("psych")
library(psych)

# 设置数据路径并加载数据
data_path <- "G:/ABCD/script/trail/trail_tsne_RF"
data <- read.csv(file.path(data_path, "merged_twins.csv"))

# data_path <- "G:/ABCD/script/trail/trail_tsne_RF/test"
# data <- read.csv(file.path(data_path, "all_year_labels.csv"))

# 删除指定列
# data <- data[, !(names(data) %in% c("Unnamed: 0", "src_subject_id"))]

# 选择以 "cbcl" 开头的列
label_columns <- grep("^cbcl", names(data), value = TRUE)
data <- data[, label_columns]

# Step 1: 找到并移除频率过低的项（超过99.5%的值为0）
low_frequency_columns <- names(data)[sapply(data, function(col) mean(col == 0) > 0.995)]
data_cleaned <- data[, !(names(data) %in% low_frequency_columns)]
cat("Removed columns with low frequency:", low_frequency_columns, "\n")

# Step 2: 计算 Polychoric 相关矩阵
polychoric_corr <- polychoric(data_cleaned)$rho  # 使用 $rho 获取相关系数矩阵


# 保存 Polychoric 相关矩阵到 CSV 文件
write.csv(polychoric_corr, file = file.path(data_path, "factor analysis/output/polychoric_correlation_matrix_twins.csv"), row.names = TRUE)

