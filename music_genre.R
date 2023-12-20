## Music Genre Prediction
## Authors: Fruzsina Ladanyi, Gayeong Kweon
## CSE 5160 Machine Learning
## Due Date: 12/13/2023
## https://www.kaggle.com/datasets/vicsuperman/prediction-of-music-genre/

# Packages used
# install.packages("caret")
# install.packages("class")
# install.packages("e1071")
# install.packages("ggplot2")
# install.packages("MLmetrics")
# install.packages("pROC")

# Load libraries
library(caret)
library(class)
library(e1071)
library(ggplot2)
library(MLmetrics)
library(pROC)

# Data Collection
musd <- read.csv('music_genre.csv', header = TRUE)

# Remove irrelevant columns
musd <- musd[, !names(musd) %in% c("instance_id", "track_name", "obtained_date")]

# Remove rows with incomplete or marked data
musd <- musd[complete.cases(musd), ]
musd <- musd[!apply(musd, 1, function(row) any(row %in% c("empty_field", "?"))), ]

# Convert relevant categorical columns to numeric
musd$artist_name <- as.numeric(factor(musd$artist_name, levels = unique(musd$artist_name)))
musd$key <- as.numeric(factor(musd$key, levels = unique(musd$key)))
musd$mode <- as.numeric(musd$mode == "Major")

# Convert other columns to numeric
musd$tempo <- as.numeric(musd$tempo)

# Create X, y
X <- musd[, !names(musd) %in% c("music_genre")]

# Normalize numeric features
normalize <- function(x) (x - min(x)) / (max(x) - min(x))
X <- as.data.frame(lapply(X, normalize))

y <- factor(musd$music_genre)


# Inspecting dataset for missing data
valid_names <- make.names(levels(y))
level_mapping <- setNames(valid_names, levels(y))
y <- factor(y, levels = level_mapping)

factor_levels <- levels(y)
na_rows_y <- is.na(y)
X <- X[!na_rows_y, ]
y <- y[!na_rows_y]
table(y)

# Drop levels with no instances
y <- droplevels(y, exclude = 'Hip.Hop')


#Deciding predictors to use

# Initialize an empty vector to store eta coefficients
eta_values <- numeric(ncol(X))

# Calculate eta for each predictor and the categorical response variable
for (i in 1:ncol(X)) {
  cross_tab <- table(X[, i], y)
  chi_square_stat <- chisq.test(cross_tab)$statistic
  rows <- nrow(cross_tab) - 1
  cols <- ncol(cross_tab) - 1
  eta_values[i] <- sqrt(chi_square_stat / (sum(cross_tab) * min(rows, cols)))
}

# Print eta coefficients for each predictor
print(eta_values)
short_colnames <- substr(colnames(X), 1, 5)
barplot(eta_values, names.arg = short_colnames, col = "skyblue", main = "Eta Coefficients for Predictors", ylab = "Eta Coefficient")

# Remove columns with low eta coefficients
X <- X[, !names(X) %in% c("key", "liveliness", "mode")]
dim(X)

# KNN Modeling
# Set the random seed for reproducibility
set.seed(123)

# Cross-validation for finding the best k
k_values <- 1:21
cv_errors <- numeric(length(k_values))

# Create folds for cross-validation
folds <- createFolds(y, k = 5, list = TRUE, returnTrain = FALSE)

for (k in k_values) {
  fold_errors <- numeric(length(folds))
  
  for (i in seq_along(folds)) {
    test_indices <- folds[[i]]
    knn_model <- knn(train = X[-test_indices, ], test = X[test_indices, ], cl = y[-test_indices], k = k)
    fold_errors[i] <- mean(knn_model != y[test_indices])
  }
  
  cv_errors[k] <- mean(fold_errors)
}

# Find the best k value
best_k <- k_values[which.min(cv_errors)]
cat("The best value for k in KNN is:", best_k, "\n")

cv_err_df <- data.frame(k = 1:21, cv_error_rate = cv_errors)
plot(cv_err_df, type = 'b', main = 'Cross-Validation Error Rates for Different k Values', xlab = 'k', ylab = 'Error Rate')

# Retraining model with best value for visualization purposes
ctrl <- trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = multiClassSummary)
best_knn_model <- train(x = X, y = y, method = "knn", tuneGrid = data.frame(k = best_k), trControl = ctrl)
predicted_probs <- predict(best_knn_model, newdata = X, type = "prob")

# ROC Curve
class_levels <- levels(y)
class_of_interest <- class_levels[1]
binary_response <- ifelse(y == class_of_interest, 1, 0)
roc_one_vs_rest <- roc(binary_response, predicted_probs[, class_of_interest])
plot(roc_one_vs_rest, col = "blue", main = "KNN ROC Curves for Different Classes")

for (class_index in 2:length(class_levels)) {
  class_of_interest <- class_levels[class_index]
  
  # Create binary response
  binary_response <- ifelse(y == class_of_interest, 1, 0)
  
  # Create ROC curve
  roc_one_vs_rest <- roc(binary_response, predicted_probs[, class_of_interest])
  lines(roc_one_vs_rest, col = rainbow(length(class_levels))[class_index], lwd = 2)
  legend("bottomright", legend = class_levels, col = rainbow(length(class_levels)), lwd = 2)
}

# Confusion matrix
predicted_labels <- predict(best_knn_model, newdata = X)
conf_matrix <- confusionMatrix(predicted_labels, y)
print(conf_matrix)

# Scatterplots with 2 different features

custom_colors <- rainbow(9)

# First plot
feature1 <- X[, 3]
feature2 <- X[, 6]
x_range <- seq(min(feature1), max(feature1), length.out = 100)
y_range <- seq(min(feature2), max(feature2), length.out = 100)
grid <- expand.grid(feature1 = x_range, feature2 = y_range)
grid_predictions <- knn(train = X[, c(3, 6)], test = grid, cl = y, k = best_k)

plot(
  feature1, feature2, 
  col = custom_colors[as.factor(y)], pch = 20, 
  main = "KNN Classification Plot", 
  xlab = "Acousticness", ylab = "Energy"
)
points(grid$feature1, grid$feature2, col = custom_colors[as.factor(grid_predictions)], pch = 25, cex = 0.1)
legend("topright", legend = levels(as.factor(y)), col = custom_colors, pch = 19, title = "Classes", cex = 0.5)

# Second Plot
feature1 <- X[, 4]
feature2 <- X[, 9]
x_range <- seq(min(feature1), max(feature1), length.out = 100)
y_range <- seq(min(feature2), max(feature2), length.out = 100)
grid <- expand.grid(feature1 = x_range, feature2 = y_range)
grid_predictions <- knn(train = X[, c(4, 9)], test = grid, cl = y, k = best_k)

plot(
  feature1, feature2, 
  col = custom_colors[as.factor(y)], pch = 20, 
  main = "KNN Classification Plot", 
  xlab = "Dancebilty", ylab = "Speechiness"
)
points(grid$feature1, grid$feature2, col = custom_colors[as.factor(grid_predictions)], pch = ".", cex = 0.5)
legend("topright", legend = levels(as.factor(y)), col = custom_colors, pch = 19, title = "Classes", cex = 0.5)


# SVM Modeling

# Best C
df <- cbind(X,y)
ctrl <- trainControl(method = "cv", number = 5)
svm_model <- train(y ~., data = df, method = "svmRadial", trControl = ctrl)
svm_model

tune_grid <- svm_model$results

ggplot(tune_grid, aes(x = C, y = Accuracy)) +
  geom_line() +
  geom_point() +
  scale_x_log10() +  # If C is on a log scale, adjust the x-axis accordingly
  labs(title = "SVM Model Performance vs. C Value",
       x = "C Value (Regularization Strength)",
       y = "Accuracy")


# Evaluate SVM model
SVMpredictions <- predict(svm_model, newdata = X)
conf_matrix <- confusionMatrix(SVMpredictions, y)
print(conf_matrix)

# ROC curve for each class

svmfit <- svm(y ~ ., data = df, kernel = "linear", cost = 1, scale = FALSE, probability = TRUE)
predicted_probs <- predict(svmfit, df, probability = TRUE)
roc_curves <- list()

plot(0, 0, type = "n", xlim = c(1, 0), ylim = c(0, 1),
     xlab = "False Positive Rate", ylab = "True Positive Rate",
     main = "SVM ROC Curves for All Classes")

class_colors <- rainbow(length(levels(df$y)))

for (class in levels(df$y)) {
  probs_class <- attr(predicted_probs, "probabilities")[, class]
  response_class <- as.numeric(df$y == class)
  roc_curve <- roc(response_class, probs_class)
  roc_curves[[class]] <- roc_curve
  class_position <- match(class, levels(df$y))
  lines(roc_curves[[class]], col = class_colors[class_position], lwd = 2)
  print(class_colors[as.numeric(factor(class))])
  
  # Calculate and print the AUC for the current class
  auc_value <- auc(roc_curve)
  cat("AUC for Class", class, ":", auc_value, "\n")
}

legend("bottomright", legend = levels(df$y), col = class_colors, lwd = 2)

# Scatterplots
# First plot

feature1 <- X[, 3]
feature2 <- X[, 6]

# Plot the features
plot(
  feature1, feature2, 
  col = custom_colors[as.factor(SVMpredictions)], pch = 20, 
  main = "SVM Classification Plot", 
  xlab = "Acousticness", ylab = "Energy"
)

points(grid$feature1, grid$feature2, col = custom_colors[as.factor(SVMpredictions)], pch = ".", cex = 0.5)
legend("topright", legend = levels(as.factor(SVMpredictions)), col = custom_colors, pch = 19, title = "Classes", cex = 0.5)

feature1 <- X[, 4]
feature2 <- X[, 9]

plot(
  feature1, feature2, 
  col = custom_colors[as.factor(SVMpredictions)], pch = 20, 
  main = "SVM Classification Plot", 
  xlab = "Danceability", ylab = "Speechiness"
)
points(grid$feature1, grid$feature2, col = custom_colors[as.factor(SVMpredictions)], pch = ".", cex = 0.5)
legend("topright", legend = levels(as.factor(SVMpredictions)), col = custom_colors, pch = 19, title = "Classes", cex = 0.5)