library(psych)
setwd('G:\\ABCD\\script\\trail\\data')
# 设置数据路径并加载数据
data_path <- file.path(getwd(), "..", "data")
data <- read.csv(file.path(data_path, "/cbcl_data_remove_useless_items.csv"))
label_columns <- c("src_subject_id", grep("^cbcl", names(data), value = TRUE))
data <- data[, label_columns]

id_column <- data[, 1, drop = FALSE]  # save the ID column
data_to_clean <- data[, -1]  # remove the ID column

# Step 2: Find and remove low-frequency items (columns)
# Remove columns with low frequency (more than 99.5% are 0)
low_frequency_columns <- names(data_to_clean)[sapply(data_to_clean, function(col) mean(col == 0) > 0.995)]
data_cleaned <- data_to_clean[, !(names(data_to_clean)
                                  %in% low_frequency_columns)]

# Romoved columns
cat("Removed columns with low frequency:", low_frequency_columns, "\n")

# Step 3: Find and remove low-frequency items (rows)
# Remove rows with low frequency (more than 99.5% are 0)
low_frequency_rows <- apply(data_cleaned, 1,
                            function(row) mean(row == 0) > 0.995)
data_cleaned <- data_cleaned[!low_frequency_rows, ]

# Add the ID column back
id_column_cleaned <- id_column[!low_frequency_rows, , drop = FALSE]
data_cleaned <- cbind(id_column_cleaned, data_cleaned)

# Removed rows
cat("Removed rows with low frequency:", which(low_frequency_rows), "\n")
# save data
write.csv(data_cleaned,
          file = "cbcl_data_remove_unrelated.csv",
          row.names = FALSE)