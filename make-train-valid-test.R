library(tidyverse)
library(rnhanesdata)
library(caret)

# 7455 participants from D wave wore the device for at most 7 days
dim(PAXINTEN_D)[1]/ 7 # days
activity_ids <- unique(PAXINTEN_D$SEQN)

# 10,348 participants in D wave in total
dim(Mortality_2015_D)

Mortality_2015_D_elig <- Mortality_2015_D %>% 
  # Filter to those eligible for mortality data linkage & those with activity data
  filter(eligstat==1 & SEQN %in% activity_ids) 

# 4,859 participants available after filtering
nrow(Mortality_2015_D_elig)

# Look at proportion of alive to dead
table(Mortality_2015_D_elig$mortstat)

# Split into train, validate, and test 70/15/15
sample_ids <- Mortality_2015_D_elig$SEQN
set.seed(1251)

# Create indices for the training set (70% of the data)
train_indices <- createDataPartition(sample_ids, p = 0.70, list = FALSE, times = 1)

# Subset the sample_ids to create the training set
train_set <- sample_ids[train_indices]

# For validation and test sets, you need to split the remaining 30%
remaining_ids <- sample_ids[-train_indices]

# Create indices for the validation set (50% of the remaining, which is 15% of total)
validate_indices <- createDataPartition(remaining_ids, p = 0.50, list = FALSE, times = 1)

# Subset the remaining_ids for validation set
validate_set <- remaining_ids[validate_indices]

# The rest goes to the test set
test_set <- remaining_ids[-validate_indices]

# Output the sets
DataPartitions <- list(train = train_set, validate = validate_set, test = test_set)

# Numbers in each set
set_size <- sapply(DataPartitions, length)

# Make csv
set_lookup <- data.frame(
  SEQN=c(DataPartitions$train,
         DataPartitions$validate,
         DataPartitions$test),
  set=c(rep("train", set_size[1]),
        rep("valid", set_size[2]),
        rep("test", set_size[3]))
)

# Store sets
write.csv(set_lookup, file = "set_lookup.csv")
