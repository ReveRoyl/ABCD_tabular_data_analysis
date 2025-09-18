# ============================================================
# CBCL cleaning in the original structure + polychoric r>0.75 aggregation
# ============================================================
library(psych)
library(igraph)

setwd('G:\\ABCD\\script\\trail\\data')
data_path <- file.path(getwd(), "..", "data")
data <- read.csv(file.path(data_path, "cbcl_data_all.csv"),
                 check.names = FALSE)

# Keep ID + cbcl* columns, like your original script
label_columns <- c("src_subject_id", grep("^cbcl", names(data), value = TRUE))
data <- data[, label_columns]

id_column     <- data[, 1, drop = FALSE]   # save ID
data_to_clean <- data[, -1, drop = FALSE]  # remove ID

# ------------------------------------------------------------
# Step 1: remove low-frequency items (columns)
# Criterion: >99.5% are 0  (add na.rm=TRUE to be robust)
# ------------------------------------------------------------
low_frequency_columns <- names(data_to_clean)[
  sapply(data_to_clean, function(col) mean(col == 0, na.rm = TRUE) > 0.995)
]

data_cleaned <- data_to_clean[, !(names(data_to_clean) %in% low_frequency_columns), drop = FALSE]
cat("Removed columns with low frequency:", 
    ifelse(length(low_frequency_columns)>0, paste(low_frequency_columns, collapse = ", "), "None"),
    "\n")

# ------------------------------------------------------------
# Step 2: (NEW) polychoric correlations + aggregate highly correlated items
# - Do NOT remove rows (subjects)
# - Items with <2 distinct observed values are dropped (polychoric requirement)
# ------------------------------------------------------------

# Coerce remaining items to numeric (keep NA)
X <- data_cleaned
for (nm in colnames(X)) {
  if (is.factor(X[[nm]]) || is.character(X[[nm]])) {
    suppressWarnings({ X[[nm]] <- as.numeric(as.character(X[[nm]])) })
  }
}

# Drop items that still have <2 distinct non-NA values
valid_levels_count <- sapply(X, function(col) length(unique(na.omit(col))))
too_few_levels <- names(valid_levels_count)[valid_levels_count < 2]
if (length(too_few_levels) > 0) {
  message("Dropped items for <2 distinct values: ", paste(too_few_levels, collapse = ", "))
  X <- X[, setdiff(colnames(X), too_few_levels), drop = FALSE]
}

# If fewer than 2 items remain, just save and exit
if (ncol(X) < 2L) {
  warning("Fewer than 2 items remain after filtering; skipping aggregation.")
  out <- cbind(id_column, X)
  write.csv(out, file = "cbcl_data_remove_unrelated.csv", row.names = FALSE)
  quit(save = "no")
}

# Compute polychoric correlation on ordered factors
X_ord <- as.data.frame(lapply(X, function(v) {
  factor(v, levels = sort(unique(na.omit(v))), ordered = TRUE)
}))

pc <- polychoric(X_ord, correct = 0, global = TRUE, smooth = FALSE, weight = NULL, na.rm = TRUE)
R <- pc$rho

item_names <- colnames(X)
stopifnot(ncol(R) == length(item_names))
colnames(R) <- rownames(R) <- item_names

# Threshold to build graph
thr <- 0.75
A <- (R > thr)
A[is.na(A)] <- FALSE
diag(A) <- FALSE

g <- igraph::graph_from_adjacency_matrix(A, mode = "undirected", diag = FALSE)
igraph::V(g)$name <- item_names

comps <- igraph::components(g)
combo_map <- split(igraph::V(g)$name, f = comps$membership)
combo_map <- combo_map[sapply(combo_map, length) >= 2]  # keep only clusters with >=2 items

# Aggregate by row means for each cluster; keep non-aggregated originals
# ------------------------------------------------------------
# Step 2b: Aggregate by row means for each cluster
# New column names = "avg_" + joined item names
# ------------------------------------------------------------
X_num <- as.data.frame(lapply(X, function(v) as.numeric(v)))
combos <- list()

if (length(combo_map) > 0) {
  for (i in seq_along(combo_map)) {
    grp <- combo_map[[i]]  # character vector of item names
    # Construct column name: avg_ + concatenated original names
    combo_name <- paste0("avg_", paste(grp, collapse = "_"))
    combos[[combo_name]] <- rowMeans(X_num[, grp, drop = FALSE], na.rm = TRUE)
  }
  combos_df <- as.data.frame(combos)

  # Remove originals that were aggregated, then add combos
  aggregated_items <- unique(unlist(combo_map))
  X_final <- cbind(
    X_num[, setdiff(colnames(X_num), aggregated_items), drop = FALSE],
    combos_df
  )
} else {
  message("No clusters with r > ", thr, " found; no combos created.")
  X_final <- X_num
}


# ------------------------------------------------------------
# Step 3: Save (rows are never removed)
# ------------------------------------------------------------
data_cleaned_final <- cbind(id_column, X_final)
write.csv(data_cleaned_final, file = "cbcl_data_remove_low_frequency.csv", row.names = FALSE)

# Log
message("Final item count: ", ncol(X_final),
        "; combos created: ", ifelse(length(combo_map) > 0, length(combo_map), 0))
