library(cluster)

dir.create("figures", showWarnings = FALSE)
dir.create("outputs", showWarnings = FALSE)

# Import and inspect

stores <- read.csv("kirana_stores.csv", stringsAsFactors = FALSE)

cat("Rows:", nrow(stores), " Columns:", ncol(stores), "\n\n")
str(stores)
cat("\nSummary (before cleaning):\n"); print(summary(stores))
cat("\nMissing values per column:\n"); print(colSums(is.na(stores)))

features <- c("orders_per_month", "avg_order_value_inr", "credit_days",
              "sku_breadth", "return_pct")

# Invalid values: business rules each attribute must satisfy
invalid <- with(stores, list(
  orders_per_month    = which(orders_per_month    <= 0),
  avg_order_value_inr = which(avg_order_value_inr <= 0),
  credit_days         = which(credit_days         <  0),
  sku_breadth         = which(sku_breadth         <= 0),
  return_pct          = which(return_pct < 0 | return_pct > 100)
))
cat("\nInvalid values found (row numbers):\n"); print(invalid)

# ---- Resolve: invalid -> NA, then impute every NA with the column median ----
# The median is used (not the mean) because it is robust to the skew and
# extreme values in these business variables. Only ~1% of cells are affected,
# so imputation keeps all 150 stores without distorting the distributions.
clean <- stores
for (v in features) {
  clean[[v]][invalid[[v]]] <- NA
}
cleaning_log <- data.frame(
  attribute   = features,
  missing     = sapply(features, function(v) sum(is.na(stores[[v]]))),
  invalid     = sapply(features, function(v) length(invalid[[v]])),
  median_used = sapply(features, function(v) median(clean[[v]], na.rm = TRUE))
)
for (v in features) {
  clean[[v]][is.na(clean[[v]])] <- median(clean[[v]], na.rm = TRUE)
}
cat("\nCleaning log:\n"); print(cleaning_log, row.names = FALSE)
cat("\nMissing values after cleaning:", sum(is.na(clean[features])), "\n")
write.csv(cleaning_log, "outputs/cleaning_log.csv", row.names = FALSE)
write.csv(clean, "outputs/kirana_stores_clean.csv", row.names = FALSE)

# Select clustering attributes

X <- clean[, features]
rownames(X) <- clean$store_id

cat("\nScale of the raw attributes (very different units/ranges):\n")
print(round(sapply(X, function(col) c(mean = mean(col), sd = sd(col),
                                       min = min(col), max = max(col))), 2))

# z-score standardisation

Z <- scale(X)                      # (x - mean) / sd for every column
cat("\nAfter scaling - means ~0 and sds = 1:\n")
print(round(rbind(mean = colMeans(Z), sd = apply(Z, 2, sd)), 3))

# Euclidean distance matrix

D <- dist(Z, method = "euclidean")
cat("\nDistance matrix: ", attr(D, "Size"), "x", attr(D, "Size"), "\n")
print(round(as.matrix(D)[1:5, 1:5], 2))

# Agglomerative clustering: single and complete linkage

hc_single   <- hclust(D, method = "single")
hc_complete <- hclust(D, method = "complete")

plot_dend <- function(hc, title, file, k = NULL) {
  png(file, width = 1800, height = 900, res = 150)
  plot(hc, labels = FALSE, hang = -1, main = title,
       xlab = "Kirana stores (n = 150)", sub = "", ylab = "Merge height (Euclidean distance)")
  if (!is.null(k)) rect.hclust(hc, k = k, border = 2:(k + 1))
  dev.off()
}
plot_dend(hc_single,   "Single linkage dendrogram",   "figures/dendrogram_single.png")
plot_dend(hc_complete, "Complete linkage dendrogram", "figures/dendrogram_complete.png")

# Candidate k from the dendrogram: heights of the last merges

# The last merges of the tree are the ones that join big groups. A large jump
# between consecutive merge heights = a long vertical gap = a natural cut.
merge_table <- function(hc, top = 8) {
  h <- rev(tail(hc$height, top))            # largest heights first
  data.frame(k_if_cut_below = 2:(top + 1),  # clusters left if we cut just below this merge
             merge_height   = round(h, 3),
             gap_to_next    = round(h - c(h[-1], NA), 3))
}
gaps_single   <- merge_table(hc_single)
gaps_complete <- merge_table(hc_complete)
cat("\nSingle linkage - top merge heights:\n");   print(gaps_single,   row.names = FALSE)
cat("\nComplete linkage - top merge heights:\n"); print(gaps_complete, row.names = FALSE)
write.csv(gaps_single,   "outputs/merge_heights_single.csv",   row.names = FALSE)
write.csv(gaps_complete, "outputs/merge_heights_complete.csv", row.names = FALSE)

# Average silhouette width for candidate k (2 to 6), both linkages

sil_summary <- function(hc, method, ks = 2:6) {
  do.call(rbind, lapply(ks, function(k) {
    cl <- cutree(hc, k = k)
    s  <- silhouette(cl, D)
    data.frame(linkage = method, k = k,
               avg_silhouette = round(mean(s[, "sil_width"]), 3),
               cluster_sizes  = paste(table(cl), collapse = "/"),
               n_negative_sil = sum(s[, "sil_width"] < 0))
  }))
}
sil_table <- rbind(sil_summary(hc_single, "single"),
                   sil_summary(hc_complete, "complete"))
cat("\nAverage silhouette widths:\n"); print(sil_table, row.names = FALSE)
write.csv(sil_table, "outputs/silhouette_summary.csv", row.names = FALSE)

# Silhouette vs k chart
png("figures/silhouette_vs_k.png", width = 1200, height = 800, res = 150)
plot(2:6, sil_table$avg_silhouette[sil_table$linkage == "complete"], type = "b",
     pch = 19, col = "steelblue", ylim = range(sil_table$avg_silhouette, 0),
     xlab = "Number of clusters (k)", ylab = "Average silhouette width",
     main = "Average silhouette width by k")
lines(2:6, sil_table$avg_silhouette[sil_table$linkage == "single"], type = "b",
      pch = 17, col = "firebrick")
abline(h = 0, lty = 3, col = "grey50")
legend("bottomleft", c("Complete linkage", "Single linkage"),
       col = c("steelblue", "firebrick"), pch = c(19, 17), lty = 1, bty = "n")
dev.off()

# Final choice  (set after reading the dendrograms + silhouette table)

final_method <- "complete"
final_k      <- 4
hc_final     <- if (final_method == "complete") hc_complete else hc_single
clusters     <- cutree(hc_final, k = final_k)

# Dendrograms with the chosen cut drawn on them
plot_dend(hc_complete, sprintf("Complete linkage - cut at k = %d", final_k),
          "figures/dendrogram_complete_cut.png", k = final_k)
plot_dend(hc_single, sprintf("Single linkage - cut at k = %d", final_k),
          "figures/dendrogram_single_cut.png", k = final_k)

cat("\nSingle-linkage cluster sizes at k =", final_k, ":\n")
print(table(cutree(hc_single, k = final_k)))
cat("\nCross-tab: single (rows) vs complete (cols) at k =", final_k, "\n")
print(table(single = cutree(hc_single, k = final_k), complete = clusters))

# Silhouette plot of the final solution

sil_final <- silhouette(clusters, D)
png("figures/silhouette_final.png", width = 1400, height = 1000, res = 150)
plot(sil_final, border = NA, col = 2:(final_k + 1),
     main = sprintf("Silhouette plot - %s linkage, k = %d", final_method, final_k))
dev.off()

sil_by_cluster <- summary(sil_final)$clus.avg.widths
cat("\nAverage silhouette width per cluster:\n"); print(round(sil_by_cluster, 3))
cat("Overall average silhouette width:", round(summary(sil_final)$avg.width, 3), "\n")

neg <- which(sil_final[, "sil_width"] < 0)
misfits <- data.frame(store_id = rownames(X)[neg],
                      cluster  = sil_final[neg, "cluster"],
                      neighbor = sil_final[neg, "neighbor"],
                      sil_width = round(sil_final[neg, "sil_width"], 3))
cat("\nStores with negative silhouette (possibly in the wrong cluster):", nrow(misfits), "\n")
print(misfits, row.names = FALSE)
write.csv(misfits, "outputs/negative_silhouette_stores.csv", row.names = FALSE)

# Cluster profile in ORIGINAL units (cleaned data, not z-scores)

clean$cluster <- clusters
profile <- aggregate(clean[, features], by = list(cluster = clean$cluster), FUN = mean)
profile$n_stores <- as.vector(table(clean$cluster))
profile <- profile[, c("cluster", "n_stores", features)]
profile[, features] <- round(profile[, features], 1)
cat("\nCluster profile (means, original units):\n"); print(profile, row.names = FALSE)

# Standardised profile too: shows which attributes make each cluster distinctive
z_profile <- aggregate(as.data.frame(Z), by = list(cluster = clusters), FUN = mean)
z_profile[, -1] <- round(z_profile[, -1], 2)
cat("\nCluster profile (mean z-scores):\n"); print(z_profile, row.names = FALSE)

write.csv(profile,   "outputs/cluster_profile_original_units.csv", row.names = FALSE)
write.csv(z_profile, "outputs/cluster_profile_zscores.csv",        row.names = FALSE)
write.csv(clean,     "outputs/kirana_stores_with_clusters.csv",    row.names = FALSE)

# Profile heatmap-style bar chart of z-scores
png("figures/cluster_profile_z.png", width = 1600, height = 1000, res = 150)
barplot(t(as.matrix(z_profile[, -1])), beside = TRUE,
        names.arg = paste("Cluster", z_profile$cluster),
        col = c("#4E79A7", "#F28E2B", "#59A14F", "#E15759", "#B07AA1"),
        ylim = c(-1.2, 3.2),
        ylab = "Mean z-score", main = "Cluster profiles (standardised attributes)")
abline(h = 0)
legend("top", legend = features, bty = "n", cex = 0.8, ncol = 3,
       fill = c("#4E79A7", "#F28E2B", "#59A14F", "#E15759", "#B07AA1"))
dev.off()

# Q4.2  Business labels (assigned after reading the profile above)

labels <- c("1" = "High-Velocity Neighbourhood Stores",
            "2" = "Low-Frequency Basic Stores",
            "3" = "Bulk Wholesale-Retailers",
            "4" = "Credit-Risk Stores")
profile$segment <- labels[as.character(profile$cluster)]
cat("\nLabelled segments:\n")
print(profile[, c("cluster", "segment", "n_stores")], row.names = FALSE)
write.csv(profile, "outputs/cluster_profile_original_units.csv", row.names = FALSE)