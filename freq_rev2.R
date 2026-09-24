# Published cellwise implementations on the enlarged head-to-head data (freq_rev2.py exports).
# For each dataset: DDC (imputed data), cellMCD (scatter and imputed data), and 2SGS
# (DDC imputation followed by the generalized S-estimator).
suppressMessages({library(cellWise); library(GSE)})
args <- commandArgs(trailingOnly = TRUE)
dir_in <- "rev2_h2h_data"; dir_out <- "rev2_h2h_out"
dir.create(dir_out, showWarnings = FALSE)
files <- list.files(dir_in, pattern = "_X\\.csv$")
for (f in files) {
  stem <- sub("_X\\.csv$", "", f)
  if (file.exists(file.path(dir_out, paste0(stem, "_gse.csv")))) next
  X <- as.matrix(read.csv(file.path(dir_in, f), header = FALSE))
  set.seed(1)
  ddc <- DDC(X, DDCpars = list(silent = TRUE))
  write.table(ddc$Ximp, file.path(dir_out, paste0(stem, "_ddcimp.csv")),
              sep = ",", row.names = FALSE, col.names = FALSE)
  write.table(abs(ddc$stdResid), file.path(dir_out, paste0(stem, "_ddcres.csv")),
              sep = ",", row.names = FALSE, col.names = FALSE)
  set.seed(1)
  mcd <- try(cellMCD(X), silent = TRUE)
  if (!inherits(mcd, "try-error")) {
    write.table(mcd$S, file.path(dir_out, paste0(stem, "_mcdS.csv")),
                sep = ",", row.names = FALSE, col.names = FALSE)
    write.table(mcd$Ximp, file.path(dir_out, paste0(stem, "_mcdimp.csv")),
                sep = ",", row.names = FALSE, col.names = FALSE)
    write.table(abs(mcd$Zres), file.path(dir_out, paste0(stem, "_mcdres.csv")),
                sep = ",", row.names = FALSE, col.names = FALSE)
  }
  g <- try(GSE(as.matrix(ddc$Ximp)), silent = TRUE)
  S <- if (inherits(g, "try-error")) cov(ddc$Ximp) else as.matrix(g@S)
  write.table(S, file.path(dir_out, paste0(stem, "_gse.csv")),
              sep = ",", row.names = FALSE, col.names = FALSE)
  cat("done", stem, "\n")
}
