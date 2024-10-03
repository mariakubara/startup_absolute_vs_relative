library(tidyverse)
library(sf)
library(sp)
library(dbscan)
library(xgboost)
library(caret)
library(factoextra)


# load warsaw districs
waw.sf <- st_read("data/dzielnice_Warszawy/dzielnice_Warszawy.shp")
waw.sf <- st_transform(waw.sf, crs = "+proj=longlat +datum=NAD83")

waw.sf.line <- st_cast(waw.sf,"MULTILINESTRING")
waw.total <- st_union(waw.sf)

# loading data prepared for modelling (anonymised)
load("data/2024-06-21 data for modelling.RData")

#groups of variables inlcuded
# Y - Total assets last
initial
absolute
relative_neighbourhood
relative_agglomeration


### Make some xgboost ################################################

# Split data into training and testing sets
model_data <- cbind(Y,X)

# Prepare data for XGBoost
data_xgb <- xgb.DMatrix(data = as.matrix(model_data %>% select(-Y)), label = model_data$Y)

# model tuning ##
set.seed(123)

# Define a tuning grid
tune_grid <- expand.grid(
  nrounds = c(100, 200),        # Number of boosting iterations
  max_depth = c(4, 6, 8),       # Maximum depth of a tree
  eta = c(0.01, 0.1, 0.3),      # Learning rate
  gamma = c(0, 1),              # Minimum loss reduction
  colsample_bytree = c(0.5, 0.7, 0.8, 1),  # Subsample ratio of columns
  min_child_weight = c(1, 3, 5),      # Minimum sum of instance weight
  subsample = c(0.5, 0.7, 1)    # Subsample ratio of the training instance
)

# Define the control for the cross-validation
train_control <- trainControl(
  method = "cv",               # Cross-validation
  number = 5,                  # Number of folds
  verboseIter = TRUE,          # Output training progress
  allowParallel = TRUE         # Allow parallel processing
)

# Train the model with cross-validation
xgb_tune <- train(
  #data = dtrain,
  x = as.matrix(model_data %>% select(-Y)), 
  y = model_data$Y,
  method = "xgbTree", 
  trControl = train_control, 
  tuneGrid = tune_grid,
  metric = "RMSE"              # Use RMSE to evaluate models
)

# View the best parameters
print(xgb_tune$bestTune)


# Extract the best parameters
best_params <- xgb_tune$bestTune

# Train the final model with the best parameters
final_model <- xgboost(
  data = data_xgb,
  nrounds = best_params$nrounds,
  max_depth = best_params$max_depth,
  eta = best_params$eta,
  gamma = best_params$gamma,
  colsample_bytree = best_params$colsample_bytree,
  min_child_weight = best_params$min_child_weight,
  subsample = best_params$subsample,
  objective = "reg:squarederror"
)

# train any model

# Define XGBoost parameters
xgb_params <- list(
  objective = "reg:squarederror",  # Use appropriate objective for your target variable
  eta = 0.1,
  max_depth = 6,
  subsample = 0.7,
  colsample_bytree = 0.8,
  eval_metric = "rmse"  # Root Mean Squared Error
)

# Train XGBoost model
xgb_model <- xgb.train(
  params = xgb_params,
  data = data_xgb,
  nrounds = 100,
  watchlist = list(train = data_xgb),
  early_stopping_rounds = 10,
  maximize = FALSE
)

# Print the evaluation log
print(xgb_model$evaluation_log)
final_model_hyperki <- final_model # po hyperparametr tuning, gorszy
final_model <- xgb_model

print(final_model$evaluation_log)

# Predict on the test set
predictions <- predict(final_model, newdata = as.matrix(model_data %>% select(-Y)))

# predictions <- predict(xgb_model, newdata = as.matrix(model_data %>% select(-Y)))

# Evaluate model performance
rmse <- sqrt(mean((model_data$Y - predictions)^2))
cat("RMSE on all data:", rmse, "\n")


# Plot the predicted vs actual values
plot(model_data$Y, predictions, main = "Predicted vs Actual Values",
     xlab = "Actual", ylab = "Predicted", pch = 19, col = "blue")
abline(0, 1, col = "red")




### DALEX package ####################################################


# Install and load the DALEX package
library(DALEX)
library(ingredients)

# Define the prediction function
predict_function <- function(model, newdata) {
  predict(model, newdata = xgb.DMatrix(data = as.matrix(newdata)))
}

# Create the explainer object
explainer <- explain(
  model = final_model,
  data = as.matrix(model_data %>% select(-Y)),
  y = Y,
  predict_function = predict_function,
  label = "XGBoost Model"
)

# Model Performance
mp <- model_performance(explainer)
plot(mp)

# Variable Importance
vi <- variable_importance(explainer)
plot(vi)

# Single Prediction Explanation
# Select a specific instance to explain
instance <- model_data[1, ] %>% select(-Y)
shap <- predict_parts(explainer, new_observation = instance, type = "shap")
plot(shap)


instance2 <- model_data[80, ] %>% select(-Y) # b.mały kapitał
shap <- predict_parts(explainer, new_observation = instance2, type = "shap")
plot(shap)

instance <- model_data[40, ] %>% select(-Y) 
shap <- predict_parts(explainer, new_observation = instance, type = "shap")
plot(shap)


# Partial Dependence Plot for a specific feature (e.g., distance_to_cbd)
pdp <- partial_dependency(explainer, variables = "investors")
plot(pdp)

pdp <- partial_dependency(explainer, variables = "Total assets first")
plot(pdp)

pdp <- partial_dependency(explainer, variables = "Employees first")
plot(pdp)

pdp <- partial_dependency(explainer, variables = "distance_to_airport")
plot(pdp)
#600-400

pdp <- partial_dependency(explainer, variables = c("localAgglomerationIndex", "distance_to_airport"))
plot(pdp)

pdp <- partial_dependency(explainer, variables = c("localAgglomerationIndex", "distance_to_centre"))
plot(pdp)

pdp <- partial_dependency(explainer, variables = c("localAgglomerationIndex", "distance_to_centre", "distance_to_airport"))
plot(pdp)

#1200-500

instance$CBD <- as.numeric(instance$CBD)
instance$distance_to_centre <- as.numeric(instance$distance_to_centre)
instance$distance_to_airport <- as.numeric(instance$distance_to_airport)

instance[] <- lapply(instance, as.numeric)
# 
instance2[] <- lapply(instance2, as.numeric)


bd_startup <- predict_parts(explainer, xgb.DMatrix(data=data.matrix(instance)), 
                            type = "break_down_interactions")
bd_startup

bd_startup <- predict_parts(explainer, instance) #, type = "break_down_interactions")
bd_startup

bd_startup2 <- predict_parts(explainer, instance2) #, type = "break_down_interactions")
bd_startup2

plot(bd_startup, show_boxplots = FALSE) + 
  ggtitle("Break down values for an example startup","") # + 

plot(bd_startup2, show_boxplots = FALSE) + 
  ggtitle("Break down values for an example startup","") # + 


cp_startup<- predict_profile(explainer, instance2)
plot(cp_startup, variables = c("investors", "Employees first"))


cp_startup<- predict_profile(explainer, instance)
plot(cp_startup, variables = c("investors", "Employees first"))

#1200-400
save(explainer, final_model, final_model_hyperki, best_params, model_data,
     file = "data/2024-06-20 modelResults optimal all params less vars.RData")


### PCA analysis ####################################################

# Install and load required packages
library(factoextra)
library(ggplot2)
library(dplyr)

# Perform PCA for each category
pca_initial_state <- prcomp(X[, initial], scale. = TRUE)
pca_location <- prcomp(X[, absolute], scale. = TRUE)
pca_relative_agglomeration <- prcomp(X[, relative_agglomeration], scale. = TRUE)
pca_relative_neighbourhood <- prcomp(X[, relative_neighbourhood], scale. = TRUE)


# Explained variance plots for each PCA
fviz_eig(pca_initial_state, addlabels = TRUE, ylim = c(0, 100)) +
  ggtitle("Scree Plot: Initial State Variables")

fviz_eig(pca_location, addlabels = TRUE, ylim = c(0, 100)) +
  ggtitle("Scree Plot: Location Variables")

fviz_eig(pca_relative_agglomeration, addlabels = TRUE, ylim = c(0, 100)) +
  ggtitle("Scree Plot: Relative Location Variables - Agglomeration")

fviz_eig(pca_relative_neighbourhood, addlabels = TRUE, ylim = c(0, 100)) +
  ggtitle("Scree Plot: Relative Location Variables - Neighbourhood")

# Get  the PCA components and add them to the original dataset
model_data <- data.frame(Y) 

model_data$pca_initial_state= pca_initial_state$x[, 1]
model_data$pca_location = pca_location$x[, 1]
model_data$pca_relative_agglomeration = pca_relative_agglomeration$x[, 1]
model_data$pca_relative_neighbourhood = pca_relative_neighbourhood$x[, 1]


model_dataPCA <- model_data

# Summary of PCA for variance explained
summary(pca_initial_state)
summary(pca_location)
summary(pca_relative_agglomeration)
summary(pca_relative_neighbourhood)


### model with PCA #############################################################

# Prepare data for XGBoost
data_xgb <- xgb.DMatrix(data = as.matrix(model_data %>% select(-Y)), label = model_data$Y)

# model tuning ##

# Define a tuning grid
tune_grid <- expand.grid(
  nrounds = c(100, 200),        # Number of boosting iterations
  max_depth = c(4, 6, 8),       # Maximum depth of a tree
  eta = c(0.01, 0.1, 0.3),      # Learning rate
  gamma = c(0, 1),              # Minimum loss reduction
  colsample_bytree = c(0.5, 0.7, 1),  # Subsample ratio of columns
  min_child_weight = c(1, 3, 5),      # Minimum sum of instance weight
  subsample = c(0.5, 0.7, 1)    # Subsample ratio of the training instance
)

# Define the control for the cross-validation
train_control <- trainControl(
  method = "cv",               # Cross-validation
  number = 5,                  # Number of folds
  verboseIter = TRUE,          # Output training progress
  allowParallel = TRUE         # Allow parallel processing
)

# Train the model with cross-validation
xgb_tune <- train(
  #data = dtrain,
  x = as.matrix(model_data %>% select(-Y)), 
  y = model_data$Y,
  method = "xgbTree", 
  trControl = train_control, 
  tuneGrid = tune_grid,
  metric = "RMSE"              # Use RMSE to evaluate models
)

# View the best parameters
print(xgb_tune$bestTune)



# Extract the best parameters
best_paramsPCA <- xgb_tune$bestTune


# Train the final model with the best parameters
final_modelPCA <- xgboost(
  data = data_xgb,
  nrounds = best_paramsPCA$nrounds,
  max_depth = best_paramsPCA$max_depth,
  eta = best_paramsPCA$eta,
  gamma = best_paramsPCA$gamma,
  colsample_bytree = best_paramsPCA$colsample_bytree,
  min_child_weight = best_paramsPCA$min_child_weight,
  subsample = best_paramsPCA$subsample,
  objective = "reg:squarederror"
)


# Print the evaluation log
print(final_modelPCA$evaluation_log)

# Predict on the test set
predictions <- predict(final_modelPCA, newdata = as.matrix(model_data %>% select(-Y)))

# Evaluate model performance
rmse <- sqrt(mean((model_data$Y - predictions)^2))
cat("RMSE on all data:", rmse, "\n")


# Plot the predicted vs actual values
plot(model_data$Y, predictions, main = "Predicted vs Actual Values",
     xlab = "Actual", ylab = "Predicted", pch = 19, col = "blue")
abline(0, 1, col = "red")




### DALEX on PCA ####################################################

# Define the prediction function
predict_function <- function(model, newdata) {
  predict(model, newdata = xgb.DMatrix(data = as.matrix(newdata)))
}

# Create the explainer object
explainerPCA <- explain(
  model = final_modelPCA,
  data = as.matrix(model_dataPCA %>% select(-Y)),
  y = Y,
  predict_function = predict_function,
  label = "XGBoost Model for PCA"
)

# Model Performance
mp <- model_performance(explainerPCA)
plot(mp)

# Variable Importance
vi <- variable_importance(explainerPCA)
plot(vi)

# Single Prediction Explanation
# Select a specific instance to explain
instance <- model_data[1, ] %>% select(-Y)
shap <- predict_parts(explainerPCA, new_observation = instance, type = "shap")
plot(shap)

instance <- model_data[10, ] %>% select(-Y)
shap <- predict_parts(explainerPCA, new_observation = instance, type = "shap")
plot(shap)

# Partial Dependence Plot for a specific feature (e.g., distance_to_cbd)
pdp <- partial_dependency(explainerPCA, variables = "pca_initial_state")
plot(pdp)

pdp <- partial_dependency(explainerPCA, variables = "pca_location")
plot(pdp)

pdp <- partial_dependency(explainerPCA, variables = "pca_relative_neighbourhood")
plot(pdp)

pdp <- partial_dependency(explainerPCA, variables = "pca_relative_agglomeration")
plot(pdp)

pdp <- partial_dependency(explainerPCA, variables = c("pca_initial_state", "pca_location",
                                                      "pca_relative_neighbourhood",
                                                      "pca_relative_agglomeration"))
plot(pdp)


perf_general <- model_performance(explainer)
perf_PCA <- model_performance(explainerPCA)
perf_PCA

plot(perf_general,perf_PCA, geom = "boxplot")


# saving the outcome
#save(explainerPCA, final_modelPCA, best_paramsPCA, model_dataPCA,
#     file = "data/2024-06-20 modelResults optimal PCA first instance less vars.RData")


