library(psych)
setwd('G:\\ABCD\\script\\trail\\data')
# 设置数据路径并加载数据
data_path <- file.path(getwd(), "..", "data")
data <- read.csv(file.path(data_path, "/cbcl_data_remove_useless_items.csv"))
label_columns <- c("src_subject_id", grep("^cbcl", names(data), value = TRUE))
data <- data[, label_columns]

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

# 重新添加 ID 列
id_column_cleaned <- id_column[!low_frequency_rows, , drop = FALSE]
data_cleaned <- cbind(id_column_cleaned, data_cleaned)

# 输出被移除的行
cat("Removed rows with low frequency:", which(low_frequency_rows), "\n")
# 保存清理后的数据
write.csv(data_cleaned, file = "cbcl_data_remove_unrelated.csv", row.names = FALSE)
