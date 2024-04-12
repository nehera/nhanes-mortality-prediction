library(tidyverse)
library(reticulate)  # We use reticulate to load numpy
np <- import("numpy")

# Function to process and save MTF images for different datasets
process_and_save_mtf_images <- function(set_label) {

  # Paths for the files
  mtf_images_path <- paste0("data/mtf_images_", set_label, ".npy")
  mort_cov_path <- paste0("data/mort_cov_", set_label, ".csv")
  
  # Use numpy module to load MTF images
  mtf_images <- np$load(mtf_images_path)
  mort_cov <- read.csv(mort_cov_path)
  
  # Prepare labels and keys
  outcome_label <- ifelse(mort_cov$mortstat == 1, "dead", "alive")
  subject_keys <- mort_cov$SEQN
  
  # Function to make heatmap from a matrix
  make_mtf_heatmap <- function(image_matrix) {
    image_data <- as.table(as.matrix(image_matrix)) %>%
      as.data.frame()
    colnames(image_data) <- c("x", "y", "value")
    
    ggplot(image_data, aes(x = x, y = y, fill = value)) +
      geom_tile() +
      scale_fill_gradient(low = "blue", high = "red") +
      coord_fixed() +
      theme_minimal() +
      theme(axis.text = element_blank(),
            axis.ticks = element_blank(),
            axis.title = element_blank(),
            legend.position = "none")
  }
  
  # Apply the heatmap function
  mtf_heatmaps <- apply(mtf_images, 1, make_mtf_heatmap)
  
  # Save heatmaps as .png files
  for (i in seq_along(subject_keys)) {
    dir.create(paste0("data/", set_label, "/", outcome_label[i]), recursive = TRUE, showWarnings = FALSE)
    fname_i <- paste0("data/", set_label, "/", outcome_label[i], "/", subject_keys[i], ".png")
    ggsave(fname_i, mtf_heatmaps[[i]], width = 20, height = 7, units = "cm")
  }
}

# Usage:
process_and_save_mtf_images("train")
process_and_save_mtf_images("valid")
process_and_save_mtf_images("test")
