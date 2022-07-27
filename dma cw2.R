# Import packages
library(ggplot2)
library(ggsci)
library(dplyr)
library(GGally)
library(tidyverse)
library(tidymodels)
library(parsnip)
library(discrim)
library(DMwR2)
library(skimr)
library(vip)
library(purrr)
library(ggpubr)

# Read Data Set
source_data <- read.csv2("https://storage.googleapis.com/kagglesdsdata/datasets/228/482/diabetes.csv?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20220504%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20220504T170049Z&X-Goog-Expires=259199&X-Goog-SignedHeaders=host&X-Goog-Signature=2d5db8ceda7875d161b119a03255184cd75cccb8751251f8e79162a8f5cac82846953651415aa3fac9187e88fa3a6eb4e1b8f1ac493a7bd3e4bc71259b5eadf5fcfeaacf6c92defd32eedca701ef0133cd98d02608bc450d32e4b4107b146eaf2e258dcca2d1acbca5449dff061d4919d597fa9bc5409b1a208a934484a393bdef813bd6053da52c646e6dbe500257b6ce6818a873d627528268799c6f9cf80ea0b2596ebf167475d431c82af9eae4bdfb02c60f634217b16e22b04158f0bba88431e956f0fba1ee9a5bda705facdcb084a7e8d3baad74aac35020baf0f0a7d0120911afe82eb6dcff7b5ac2ec48f179a7683c1ccac5cae0c498171a018fff63", 
                         header = TRUE, sep = ",", stringsAsFactors = FALSE)

# Exploratory Data Analysis(EDA) & Data Preprocessing
glimpse(source_data)
## Data transformation
## Both the BMI & DiabetesPedigreeFunction columns are non-numeric, need to be converted...
## Meanwhile, the Outcome column need to be transform into no-numeric in order to make it easier to categorize later...
source_data %>% mutate_if(is.character,as.numeric) %>% mutate(Outcome = as_factor(ifelse(Outcome == 1, "Positive", "Negative"))) -> raw_data

## 
skim(raw_data)
## No missing value.
## Mutate 0 value of to median.

## zero frequency
zero_freq <- data.frame(Features = c("Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"), 
                        Frequencies_of_Zero = c(length(which(raw_data$Glucose == 0)),
                                                length(which(raw_data$BloodPressure == 0)),
                                                length(which(raw_data$SkinThickness == 0)),
                                                length(which(raw_data$Insulin == 0)),
                                                length(which(raw_data$BMI == 0))))

ggplot(zero_freq, aes(x = Features, y = Frequencies_of_Zero, fill = Features)) +
  scale_fill_manual(values = c("#E64B35CC","#4DBBD5CC","#00A087CC","#3C5488CC","#F39B7FCC")) +
  geom_bar(alpha = 0.8, width = 0.6, stat = "identity") +
  theme_light() +
  geom_text(aes(label = Frequencies_of_Zero), size = 3.6, hjust = 0.5, vjust = -1) +
  labs(y = "Frequencies of Zero")

## Through counting the zero values in these columns, we found that the Insulin and SkinThickness columns contained an inordinate number of zero values. 
## If all the rows containing zero values were removed, the size of the dataset would be almost halved. 
## So we decided to replace the zero values with NAs temporarily.
## After splitting data, we would change the NAs on training set to the median of the corresponding columns.
temp <- raw_data[,setdiff(names(raw_data), c("Pregnancies", "DiabetesPedigreeFunction","Age","Outcome"))]
temp[temp==0] <- NA
raw_data[,names(temp)] <- temp

raw_pairs_plot <- ggpairs(raw_data,aes(color = Outcome),lower=list(continuous="smooth")) + 
  theme_bw() + theme(plot.title=element_text(face='bold',color='black',hjust=0.5,size=12))

processed_pairs_plot <- raw_data %>% knnImputation(k = 5) %>%
  ggpairs(aes(color = Outcome),lower=list(continuous="smooth")) + 
    theme_bw() + 
    theme(plot.title=element_text(face='bold',color='black',hjust=0.5,size=12))

## The three pairs of variables with the highest correlations were SkinThickness and BMI, Insulin and Glucose, and Age and Pregnancies, with correlations of 0.648, 0.581 and 0.544 respectively.
## Thus we did not see the need to reduce the dimensionality of the data.

# Splitting Data
set.seed(20396662)

splited_data <- initial_split(raw_data, strata = Outcome)
training_data <- training(splited_data)
testing_data <- testing(splited_data)
cv_folds <- vfold_cv(training_data, v = 10, strata = Outcome)

# Using Recipes to process data
raw_recipe_data <- recipe(Outcome ~.,training_data)
processed_recipe_data <- recipe(Outcome ~., data = training_data) %>% 
  step_impute_knn(Glucose, Insulin, SkinThickness, BloodPressure, BMI) %>% 
  step_BoxCox(SkinThickness) %>% 
  step_log(Insulin,DiabetesPedigreeFunction,Age) %>% 
  step_normalize(all_predictors())

# Modelling

## Four Models use Only Processed Data

## K-Nearest Neighbors
knn_model <- nearest_neighbor() %>% set_mode("classification") %>% set_engine("kknn")
## Workflow 
knn_workflow_processed <- workflow() %>% add_recipe(processed_recipe_data) %>% add_model(knn_model)
set.seed(20406680)  
## Cross validate 
knn_fit_processed <- knn_workflow_processed %>% fit_resamples(cv_folds, control = control_resamples(save_pred = TRUE))
## Finalize workflow
knn_best_auc_processed <- select_best(knn_fit_processed, "roc_auc")
final_knn_workflow_processed <- finalize_workflow(knn_workflow_processed, knn_best_auc_processed)

## Random Forest
rf_model <- rand_forest() %>% set_mode("classification") %>% set_engine("ranger")
## Workflow 
rf_workflow_processed <- workflow() %>% add_recipe(processed_recipe_data) %>% add_model(rf_model)
set.seed(20406680)  
## Cross validate 
rf_fit_processed <- rf_workflow_processed %>% fit_resamples(cv_folds, control = control_resamples(save_pred = TRUE))
## Finalize workflow
rf_best_auc_processed <- select_best(rf_fit_processed, "roc_auc")
final_rf_workflow_processed <- finalize_workflow(rf_workflow_processed, rf_best_auc_processed)

## Bayesian additive regression trees (BART)
bart_model <- parsnip::bart() %>% set_mode("classification") %>% set_engine("dbarts")
## Workflow 
bart_workflow_processed <- workflow() %>% add_recipe(processed_recipe_data) %>% add_model(bart_model)
set.seed(20406680)  
## Cross validate 
bart_fit_processed <- bart_workflow_processed %>% fit_resamples(cv_folds, control = control_resamples(save_pred = TRUE))
## Finalize workflow
bart_best_auc_processed <- select_best(bart_fit_processed, "roc_auc")
final_bart_workflow_processed <- finalize_workflow(bart_workflow_processed, bart_best_auc_processed)

## Radial basis function Support Vector Machines (SVM)
svm_model <- svm_rbf() %>% set_mode("classification") %>% set_engine("kernlab")
## Workflow 
svm_workflow_processed <- workflow() %>% add_recipe(processed_recipe_data) %>% add_model(svm_model)
set.seed(20406680)  
## Cross validate 
svm_fit_processed <- svm_workflow_processed %>% fit_resamples(cv_folds, control = control_resamples(save_pred = TRUE))
## Finalize workflow
svm_best_auc_processed <- select_best(svm_fit_processed, "roc_auc")
final_svm_workflow_processed <- finalize_workflow(svm_workflow_processed, svm_best_auc_processed)

## Four Models use Both Raw Data and Processed Data

## Logistic Regression
lr_model <- logistic_reg() %>% set_mode("classification") %>% set_engine("glm")
## Workflow 
lr_workflow_raw <- workflow() %>% add_recipe(raw_recipe_data) %>% add_model(lr_model)
lr_workflow_processed <- workflow() %>% add_recipe(processed_recipe_data) %>% add_model(lr_model)
set.seed(20396662)  
## Cross validate 
lr_fit_raw <- lr_workflow_raw %>% fit_resamples(cv_folds, control = control_resamples(save_pred = TRUE))
lr_fit_processed <- lr_workflow_processed %>% fit_resamples(cv_folds, control = control_resamples(save_pred = TRUE))
## Finalize workflow
lr_best_auc_raw <- select_best(lr_fit_raw, "roc_auc")
lr_best_auc_processed <- select_best(lr_fit_processed, "roc_auc")
final_lr_workflow_raw <- finalize_workflow(lr_workflow_raw, lr_best_auc_raw)
final_lr_workflow_processed <- finalize_workflow(lr_workflow_processed, lr_best_auc_processed)

## Naïve Bayes
nb_model <- naive_Bayes() %>% set_mode("classification") %>% set_engine("naivebayes")
## Workflow
nb_workflow_raw <- workflow() %>% add_recipe(raw_recipe_data) %>% add_model(nb_model)
nb_workflow_processed <- workflow() %>% add_recipe(processed_recipe_data) %>% add_model(nb_model)
set.seed(20396662)  
## Cross validate 
nb_fit_raw <- nb_workflow_raw %>% fit_resamples(cv_folds, control = control_resamples(save_pred = TRUE))
nb_fit_processed <- nb_workflow_processed %>% fit_resamples(cv_folds, control = control_resamples(save_pred = TRUE))
## Finalize workflow
nb_best_auc_raw <- select_best(nb_fit_raw, "roc_auc")
nb_best_auc_processed <- select_best(nb_fit_processed, "roc_auc")
final_nb_workflow_raw <- finalize_workflow(nb_workflow_raw, nb_best_auc_raw)
final_nb_workflow_processed <- finalize_workflow(nb_workflow_processed, nb_best_auc_processed)

## Boosted Trees
bt_model <- boost_tree() %>% set_mode("classification") %>% set_engine("xgboost")
## Workflow 
bt_workflow_raw <- workflow() %>% add_recipe(raw_recipe_data) %>% add_model(bt_model)
bt_workflow_processed <- workflow() %>% add_recipe(processed_recipe_data) %>% add_model(bt_model)
set.seed(20396662)  
## Cross validate 
bt_fit_raw <- bt_workflow_raw %>% fit_resamples(cv_folds, control = control_resamples(save_pred = TRUE))
bt_fit_processed <- bt_workflow_processed %>% fit_resamples(cv_folds, control = control_resamples(save_pred = TRUE))
## Finalize workflow
bt_best_auc_raw <- select_best(bt_fit_raw, "roc_auc")
bt_best_auc_processed <- select_best(bt_fit_processed, "roc_auc")
final_bt_workflow_raw <- finalize_workflow(bt_workflow_raw, bt_best_auc_raw)
final_bt_workflow_processed <- finalize_workflow(bt_workflow_processed, bt_best_auc_processed)

## Multilayer Perceptron Model
mlp_model <- mlp() %>% set_mode("classification") %>%   set_engine("nnet")
## Workflow 
mlp_workflow_raw <- workflow() %>% add_recipe(raw_recipe_data) %>% add_model(mlp_model)
mlp_workflow_processed <- workflow() %>% add_recipe(processed_recipe_data) %>% add_model(mlp_model)
set.seed(20396662)  
## Cross validate 
mlp_fit_raw <- mlp_workflow_raw %>% fit_resamples(cv_folds, control = control_resamples(save_pred = TRUE))
mlp_fit_processed <- mlp_workflow_processed %>% fit_resamples(cv_folds, control = control_resamples(save_pred = TRUE))
## Finalize workflow
mlp_best_auc_raw <- select_best(mlp_fit_raw, "roc_auc")
mlp_best_auc_processed <- select_best(mlp_fit_processed, "roc_auc")
final_mlp_workflow_raw <- finalize_workflow(mlp_workflow_raw, mlp_best_auc_raw)
final_mlp_workflow_processed <- finalize_workflow(mlp_workflow_processed, mlp_best_auc_processed)

# Training Result



## ROC Curves of Validation Set for top penalized Models Using Processed Data
lr_auc_processed <- lr_fit_processed %>% collect_predictions(parameters = lr_best_auc_processed) %>% roc_curve(Outcome, .pred_Positive) %>% mutate(model = "Logistic Regression")
nb_auc_processed <- nb_fit_processed %>% collect_predictions(parameters = nb_best_auc_processed) %>% roc_curve(Outcome, .pred_Positive) %>% mutate(model = "Navie Bayes")
knn_auc_processed <- knn_fit_processed %>% collect_predictions(parameters = knn_best_auc_processed) %>% roc_curve(Outcome, .pred_Positive) %>% mutate(model = "K-Nearest Neighbors")
rf_auc_processed <- rf_fit_processed %>% collect_predictions(parameters = rf_best_auc_processed) %>% roc_curve(Outcome, .pred_Positive) %>% mutate(model = "Random Forests")
bart_auc_processed <- bart_fit_processed %>% collect_predictions(parameters = bart_best_auc_processed) %>% roc_curve(Outcome, .pred_Positive) %>% mutate(model = "BART")
bt_auc_processed <- bt_fit_processed %>% collect_predictions(parameters = bt_best_auc_processed) %>% roc_curve(Outcome, .pred_Positive) %>% mutate(model = "Boosted Trees")
svm_auc_processed <- svm_fit_processed %>% collect_predictions(parameters = svm_best_auc_processed) %>% roc_curve(Outcome, .pred_Positive) %>% mutate(model = "SVM_rbf")
mlp_auc_processed <- mlp_fit_processed %>% collect_predictions(parameters = mlp_best_auc_processed) %>% roc_curve(Outcome, .pred_Positive) %>% mutate(model = "MLP")

bind_rows(lr_auc_processed, nb_auc_processed, knn_auc_processed, rf_auc_processed, bart_auc_processed, bt_auc_processed, svm_auc_processed, mlp_auc_processed) %>%
  ggplot(aes(x = 1 - specificity, y = sensitivity, col = model)) +
  geom_path(lwd = 1.5, alpha = 0.8) +
  geom_abline(lty = 3) + 
  coord_equal() +
  theme_light() + 
  labs(x = "1 - Specificity", y = "Sensitivity") +
  theme(axis.title.x = element_text(size = 15), 
        axis.title.y = element_text(size = 15),
        axis.text.x = element_text(size = 12),
        axis.text.y = element_text(size = 12))

## ROC Curves of Validation Set for top penalized Models Using Both Raw Data &Processed Data
lr_auc_raw <- lr_fit_raw %>% collect_predictions(parameters = lr_best_auc_raw) %>% roc_curve(Outcome, .pred_Positive) %>% mutate(model = "Logistic Regression")
nb_auc_raw <- nb_fit_raw %>% collect_predictions(parameters = nb_best_auc_raw) %>% roc_curve(Outcome, .pred_Positive) %>% mutate(model = "Navie Bayes")
bt_auc_raw <- bt_fit_raw %>% collect_predictions(parameters = bt_best_auc_raw) %>% roc_curve(Outcome, .pred_Positive) %>% mutate(model = "Boosted Trees")
mlp_auc_raw <- mlp_fit_raw %>% collect_predictions(parameters = mlp_best_auc_raw) %>% roc_curve(Outcome, .pred_Positive) %>% mutate(model = "MLP")

bind_rows(lr_auc_raw, nb_auc_raw, bt_auc_raw, mlp_auc_raw) %>%
  ggplot(aes(x = 1 - specificity, y = sensitivity, col = model)) +
  geom_path(lwd = 1.5, alpha = 0.8) +
  geom_abline(lty = 3) + 
  coord_equal() +
  theme_light() + 
  labs(x = "1 - Specificity", y = "Sensitivity") +
  theme(axis.title.x = element_text(size = 15), 
        axis.title.y = element_text(size = 15),
        axis.text.x = element_text(size = 12),
        axis.text.y = element_text(size = 12))

bind_rows(lr_auc_processed, nb_auc_processed, bt_auc_processed, mlp_auc_processed) %>%
  ggplot(aes(x = 1 - specificity, y = sensitivity, col = model)) +
  geom_path(lwd = 1.5, alpha = 0.8) +
  geom_abline(lty = 3) + 
  coord_equal() +
  theme_light() + 
  labs(x = "1 - Specificity", y = "Sensitivity") +
  theme(axis.title.x = element_text(size = 15), 
        axis.title.y = element_text(size = 15),
        axis.text.x = element_text(size = 12),
        axis.text.y = element_text(size = 12))

# Testing

set.seed(20396662)
last_lr_fit_processed <- last_fit(final_lr_workflow_processed, splited_data)
last_nb_fit_processed <- last_fit(final_nb_workflow_processed, splited_data)
last_knn_fit_processed <- last_fit(final_knn_workflow_processed, splited_data)
last_rf_fit_processed <- last_fit(final_rf_workflow_processed, splited_data)
last_bart_fit_processed <- last_fit(final_bart_workflow_processed, splited_data)
last_bt_fit_processed <- last_fit(final_bt_workflow_processed, splited_data)
last_svm_fit_processed <- last_fit(final_svm_workflow_processed, splited_data)
last_mlp_fit_processed <- last_fit(final_mlp_workflow_processed, splited_data)

last_lr_fit_raw <- last_fit(final_lr_workflow_raw, splited_data)
last_nb_fit_raw <- last_fit(final_nb_workflow_raw, splited_data)
last_bt_fit_raw <- last_fit(final_bt_workflow_raw, splited_data)
last_mlp_fit_raw <- last_fit(final_mlp_workflow_raw, splited_data)

# Evaluation

## Accuracy
## Using Processed Data
last_lr_fit_processed %>% collect_predictions() %>% accuracy(Outcome, .pred_class) %>% mutate(model = "Logistic Regression") -> accur_processed
last_bt_fit_processed %>% collect_predictions() %>% accuracy(Outcome, .pred_class) %>% mutate(model = "Navie Bayes") %>% rbind(accur_processed) -> accur_processed
last_knn_fit_processed %>% collect_predictions() %>% accuracy(Outcome, .pred_class) %>% mutate(model = "K-Nearest Neighbors") %>% rbind(accur_processed) -> accur_processed
last_rf_fit_processed %>% collect_predictions() %>% accuracy(Outcome, .pred_class) %>% mutate(model = "Random Forests") %>% rbind(accur_processed) -> accur_processed
last_bart_fit_processed %>% collect_predictions() %>% accuracy(Outcome, .pred_class) %>% mutate(model = "BART") %>% rbind(accur_processed) -> accur_processed
last_bt_fit_processed %>% collect_predictions() %>% accuracy(Outcome, .pred_class) %>% mutate(model = "Boosted Trees") %>% rbind(accur_processed) -> accur_processed
last_svm_fit_processed %>% collect_predictions() %>% accuracy(Outcome, .pred_class) %>% mutate(model = "SVM_rbf") %>% rbind(accur_processed) -> accur_processed
last_mlp_fit_processed %>% collect_predictions() %>% accuracy(Outcome, .pred_class) %>% mutate(model = "MLP") %>% rbind(accur_processed) -> accur_processed
accur_processed[order(-accur_processed$.estimate),]

accur_processed %>% mutate(.estimate = round(accur_processed$.estimate, 3)) %>% ggplot(aes(x = model, y = .estimate, fill = model)) +
  geom_bar(alpha = 0.8, width = 0.6, stat = "identity") +
  labs(x = "Models", y = "AUC") +
  geom_text(aes(label = .estimate), size = 3.6, hjust = 0.5, vjust = -1) +
  theme_light()

last_lr_fit_raw %>% collect_predictions() %>% accuracy(Outcome, .pred_class) %>% mutate(model = "Logistic Regression") -> accur_raw
last_bt_fit_raw %>% collect_predictions() %>% accuracy(Outcome, .pred_class) %>% mutate(model = "Navie Bayes") %>% rbind(accur_raw) -> accur_raw
last_bt_fit_raw %>% collect_predictions() %>% accuracy(Outcome, .pred_class) %>% mutate(model = "Boosted Trees") %>% rbind(accur_raw) -> accur_raw
last_mlp_fit_raw %>% collect_predictions() %>% accuracy(Outcome, .pred_class) %>% mutate(model = "MLP") %>% rbind(accur_raw) -> accur_raw
accur_raw[order(-accur_raw$.estimate),]

accur_comparison <- accur_raw %>% mutate(DataUsed = "Raw Data")
accur_processed[accur_processed$model=="Logistic Regression",] %>%
  rbind(accur_processed[accur_processed$model=="Navie Bayes",]) %>%
  rbind(accur_processed[accur_processed$model=="Boosted Trees",]) %>%
  rbind(accur_processed[accur_processed$model=="MLP",]) %>% 
  mutate(DataUsed = "Processed Data") %>%
  rbind(accur_comparison) %>%
  mutate(.estimate = round(.$.estimate, 3)) %>%
  ggplot(aes(x=model, y=.estimate, fill=DataUsed)) + geom_bar(alpha = 0.8, width = 0.6,stat="identity", position="dodge") +
  scale_fill_manual(values=c("#0073C2FF","#EFC000FF")) +
  labs(x="Models", y="Accuracy") +
  geom_text(aes(label = .estimate), position=position_dodge(width=0.5), size = 3.6, hjust = 0.5, vjust = -1) +
  theme_light()

## Precision

last_lr_fit_processed %>% collect_predictions() %>% precision(Outcome, .pred_class) %>% mutate(model = "Logistic Regression") -> preci_processed
last_bt_fit_processed %>% collect_predictions() %>% precision(Outcome, .pred_class) %>% mutate(model = "Navie Bayes") %>% rbind(preci_processed) -> preci_processed
last_knn_fit_processed %>% collect_predictions() %>% precision(Outcome, .pred_class) %>% mutate(model = "K-Nearest Neighbors") %>% rbind(preci_processed) -> preci_processed
last_rf_fit_processed %>% collect_predictions() %>% precision(Outcome, .pred_class) %>% mutate(model = "Random Forests") %>% rbind(preci_processed) -> preci_processed
last_bart_fit_processed %>% collect_predictions() %>% precision(Outcome, .pred_class) %>% mutate(model = "BART") %>% rbind(preci_processed) -> preci_processed
last_bt_fit_processed %>% collect_predictions() %>% precision(Outcome, .pred_class) %>% mutate(model = "Boosted Trees") %>% rbind(preci_processed) -> preci_processed
last_svm_fit_processed %>% collect_predictions() %>% precision(Outcome, .pred_class) %>% mutate(model = "SVM_rbf") %>% rbind(preci_processed) -> preci_processed
last_mlp_fit_processed %>% collect_predictions() %>% precision(Outcome, .pred_class) %>% mutate(model = "MLP") %>% rbind(preci_processed) -> preci_processed
preci_processed[order(-preci_processed$.estimate),]

preci_processed %>% mutate(.estimate = round(preci_processed$.estimate, 3)) %>% ggplot(aes(x = model, y = .estimate, fill = model)) +
  geom_bar(alpha = 0.8, width = 0.6, stat = "identity") +
  xlab("Models") + ylab("Precision") +
  geom_text(aes(label = .estimate), size = 3.6, hjust = 0.5, vjust = -1) +
  theme_light()

last_lr_fit_raw %>% collect_predictions() %>% precision(Outcome, .pred_class) %>% mutate(model = "Logistic Regression") -> preci_raw
last_bt_fit_raw %>% collect_predictions() %>% precision(Outcome, .pred_class) %>% mutate(model = "Navie Bayes") %>% rbind(preci_raw) -> preci_raw
last_bt_fit_raw %>% collect_predictions() %>% precision(Outcome, .pred_class) %>% mutate(model = "Boosted Trees") %>% rbind(preci_raw) -> preci_raw
last_mlp_fit_raw %>% collect_predictions() %>% precision(Outcome, .pred_class) %>% mutate(model = "MLP") %>% rbind(preci_raw) -> preci_raw
preci_raw[order(-preci_raw$.estimate),]

preci_comparison <- preci_raw %>% mutate(DataUsed = "Raw Data")
preci_processed[preci_processed$model=="Logistic Regression",] %>%
  rbind(preci_processed[preci_processed$model=="Navie Bayes",]) %>%
  rbind(preci_processed[preci_processed$model=="Boosted Trees",]) %>%
  rbind(preci_processed[preci_processed$model=="MLP",]) %>% 
  mutate(DataUsed = "Processed Data") %>%
  rbind(preci_comparison) %>%
  mutate(.estimate = round(.$.estimate, 3)) %>%
  ggplot(aes(x=model, y=.estimate, fill=DataUsed)) + geom_bar(alpha = 0.8, width = 0.6,stat="identity", position="dodge") +
  scale_fill_manual(values=c("#0073C2FF","#EFC000FF")) +
  labs(x="Models", y="Precision") +
  geom_text(aes(label = .estimate), position=position_dodge(width=0.5), size = 3.6, hjust = 0.5, vjust = -1) +
  theme_light()

## Recall

last_lr_fit_processed %>% collect_predictions() %>% recall(Outcome, .pred_class) %>% mutate(model = "Logistic Regression") -> reca_processed
last_bt_fit_processed %>% collect_predictions() %>% recall(Outcome, .pred_class) %>% mutate(model = "Navie Bayes") %>% rbind(reca_processed) -> reca_processed
last_knn_fit_processed %>% collect_predictions() %>% recall(Outcome, .pred_class) %>% mutate(model = "K-Nearest Neighbors") %>% rbind(reca_processed) -> reca_processed
last_rf_fit_processed %>% collect_predictions() %>% recall(Outcome, .pred_class) %>% mutate(model = "Random Forests") %>% rbind(reca_processed) -> reca_processed
last_bart_fit_processed %>% collect_predictions() %>% recall(Outcome, .pred_class) %>% mutate(model = "BART") %>% rbind(reca_processed) -> reca_processed
last_bt_fit_processed %>% collect_predictions() %>% recall(Outcome, .pred_class) %>% mutate(model = "Boosted Trees") %>% rbind(reca_processed) -> reca_processed
last_svm_fit_processed %>% collect_predictions() %>% recall(Outcome, .pred_class) %>% mutate(model = "SVM_rbf") %>% rbind(reca_processed) -> reca_processed
last_mlp_fit_processed %>% collect_predictions() %>% recall(Outcome, .pred_class) %>% mutate(model = "MLP") %>% rbind(reca_processed) -> reca_processed
reca_processed[order(-reca_processed$.estimate),]

reca_processed %>% mutate(.estimate = round(reca_processed$.estimate, 3)) %>% ggplot(aes(x = model, y = .estimate, fill = model)) +
  geom_bar(alpha = 0.8, width = 0.6, stat = "identity") +
  xlab("Models") + ylab("Recall") +
  geom_text(aes(label = .estimate), size = 3.6, hjust = 0.5, vjust = -1) +
  theme_light()

last_lr_fit_raw %>% collect_predictions() %>% recall(Outcome, .pred_class) %>% mutate(model = "Logistic Regression") -> reca_raw
last_bt_fit_raw %>% collect_predictions() %>% recall(Outcome, .pred_class) %>% mutate(model = "Navie Bayes") %>% rbind(reca_raw) -> reca_raw
last_bt_fit_raw %>% collect_predictions() %>% recall(Outcome, .pred_class) %>% mutate(model = "Boosted Trees") %>% rbind(reca_raw) -> reca_raw
last_mlp_fit_raw %>% collect_predictions() %>% recall(Outcome, .pred_class) %>% mutate(model = "MLP") %>% rbind(reca_raw) -> reca_raw
reca_raw[order(-reca_raw$.estimate),]

reca_comparison <- reca_raw %>% mutate(DataUsed = "Raw Data")
reca_processed[reca_processed$model=="Logistic Regression",] %>%
  rbind(reca_processed[reca_processed$model=="Navie Bayes",]) %>%
  rbind(reca_processed[reca_processed$model=="Boosted Trees",]) %>%
  rbind(reca_processed[reca_processed$model=="MLP",]) %>% 
  mutate(DataUsed = "Processed Data") %>%
  rbind(reca_comparison) %>%
  mutate(.estimate = round(.$.estimate, 3)) %>%
  ggplot(aes(x=model, y=.estimate, fill=DataUsed)) + geom_bar(alpha = 0.8, width = 0.6,stat="identity", position="dodge") +
  scale_fill_manual(values=c("#0073C2FF","#EFC000FF")) +
  labs(x="Models", y="Recall") +
  geom_text(aes(label = .estimate), position=position_dodge(width=0.5), size = 3.6, hjust = 0.5, vjust = -1) +
  theme_light()

## F1-measures

last_lr_fit_processed %>% collect_predictions() %>% f_meas(Outcome, .pred_class) %>% mutate(model = "Logistic Regression") -> fmeas_processed
last_bt_fit_processed %>% collect_predictions() %>% f_meas(Outcome, .pred_class) %>% mutate(model = "Navie Bayes") %>% rbind(fmeas_processed) -> fmeas_processed
last_knn_fit_processed %>% collect_predictions() %>% f_meas(Outcome, .pred_class) %>% mutate(model = "K-Nearest Neighbors") %>% rbind(fmeas_processed) -> fmeas_processed
last_rf_fit_processed %>% collect_predictions() %>% f_meas(Outcome, .pred_class) %>% mutate(model = "Random Forests") %>% rbind(fmeas_processed) -> fmeas_processed
last_bart_fit_processed %>% collect_predictions() %>% f_meas(Outcome, .pred_class) %>% mutate(model = "BART") %>% rbind(fmeas_processed) -> fmeas_processed
last_bt_fit_processed %>% collect_predictions() %>% f_meas(Outcome, .pred_class) %>% mutate(model = "Boosted Trees") %>% rbind(fmeas_processed) -> fmeas_processed
last_svm_fit_processed %>% collect_predictions() %>% f_meas(Outcome, .pred_class) %>% mutate(model = "SVM_rbf") %>% rbind(fmeas_processed) -> fmeas_processed
last_mlp_fit_processed %>% collect_predictions() %>% f_meas(Outcome, .pred_class) %>% mutate(model = "MLP") %>% rbind(fmeas_processed) -> fmeas_processed
fmeas_processed[order(-fmeas_processed$.estimate),]

fmeas_processed %>% mutate(.estimate = round(fmeas_processed$.estimate, 3)) %>% ggplot(aes(x = model, y = .estimate, fill = model)) +
  geom_bar(alpha = 0.8, width = 0.6, stat = "identity") +
  xlab("Models") + ylab("F-measure") +
  geom_text(aes(label = .estimate), size = 3.6, hjust = 0.5, vjust = -1) +
  theme_light()

last_lr_fit_raw %>% collect_predictions() %>% f_meas(Outcome, .pred_class) %>% mutate(model = "Logistic Regression") -> fmeas_raw
last_bt_fit_raw %>% collect_predictions() %>% f_meas(Outcome, .pred_class) %>% mutate(model = "Navie Bayes") %>% rbind(fmeas_raw) -> fmeas_raw
last_bt_fit_raw %>% collect_predictions() %>% f_meas(Outcome, .pred_class) %>% mutate(model = "Boosted Trees") %>% rbind(fmeas_raw) -> fmeas_raw
last_mlp_fit_raw %>% collect_predictions() %>% f_meas(Outcome, .pred_class) %>% mutate(model = "MLP") %>% rbind(fmeas_raw) -> fmeas_raw
fmeas_raw[order(-fmeas_raw$.estimate),]

fmeas_comparison <- fmeas_raw %>% mutate(DataUsed = "Raw Data")
fmeas_processed[fmeas_processed$model=="Logistic Regression",] %>%
  rbind(fmeas_processed[fmeas_processed$model=="Navie Bayes",]) %>%
  rbind(fmeas_processed[fmeas_processed$model=="Boosted Trees",]) %>%
  rbind(fmeas_processed[fmeas_processed$model=="MLP",]) %>% 
  mutate(DataUsed = "Processed Data") %>%
  rbind(reca_comparison) %>%
  mutate(.estimate = round(.$.estimate, 3)) %>%
  ggplot(aes(x=model, y=.estimate, fill=DataUsed)) + geom_bar(alpha = 0.8, width = 0.6,stat="identity", position="dodge") +
  scale_fill_manual(values=c("#0073C2FF","#EFC000FF")) +
  labs(x="Models", y="F1-measures") +
  geom_text(aes(label = .estimate), position=position_dodge(width=0.5), size = 3.6, hjust = 0.5, vjust = -1) +
  theme_light()

## ROC Curves & AUC

## Using Processed Data
## Logistic Regression
lr_ROC_plt_processed <- last_lr_fit_processed %>% collect_predictions() %>% 
  roc_curve(Outcome, .pred_Positive) %>% ggplot(aes(x = 1 - specificity, y = sensitivity)) + 
    geom_abline(lty = 2, alpha = 0.5, color = "gray50", size = 1.2) + 
    geom_path(size = 1.2, color = "#E64B35CC") + 
    theme_light() + 
    labs(x = "1 - Specificity", y = "Sensitivity", title = "ROC for Logistic Regression") +
  theme(plot.title = element_text(size = 16, hjust = 0.5),
        axis.title.x = element_text(size = 15), 
        axis.title.y = element_text(size = 15),
        axis.text.x = element_text(size = 12),
        axis.text.y = element_text(size = 12))

## Naive Bayes
nb_ROC_plt_processed <- last_nb_fit_processed %>% collect_predictions() %>% 
   roc_curve(Outcome, .pred_Positive) %>% ggplot(aes(x = 1 - specificity, y = sensitivity)) + 
   geom_abline(lty = 2, alpha = 0.5, color = "gray50", size = 1.2) + 
   geom_path(size = 1.2, color = "#4DBBD5CC") + 
   theme_light() + 
   labs(x = "1 - Specificity", y = "Sensitivity", title = "ROC for Naive Bayes") +
  theme(plot.title = element_text(size = 16, hjust = 0.5),
        axis.title.x = element_text(size = 15), 
        axis.title.y = element_text(size = 15),
        axis.text.x = element_text(size = 12),
        axis.text.y = element_text(size = 12))

## K-Nearest Neighbors
knn_ROC_plt_processed <- last_knn_fit_processed %>% collect_predictions() %>% 
  roc_curve(Outcome, .pred_Positive) %>% ggplot(aes(x = 1 - specificity, y = sensitivity)) + 
   geom_abline(lty = 2, alpha = 0.5, color = "gray50", size = 1.2) + 
   geom_path(size = 1.2, color = "#00A087CC") + 
   theme_light() + 
   labs(x = "1 - Specificity", y = "Sensitivity", title = "ROC for KNN") +
  theme(plot.title = element_text(size = 16, hjust = 0.5),
        axis.title.x = element_text(size = 15), 
        axis.title.y = element_text(size = 15),
        axis.text.x = element_text(size = 12),
        axis.text.y = element_text(size = 12))

## Random Forests
rf_ROC_plt_processed <- last_rf_fit_processed %>% collect_predictions() %>% 
  roc_curve(Outcome, .pred_Positive) %>% ggplot(aes(x = 1 - specificity, y = sensitivity)) + 
   geom_abline(lty = 2, alpha = 0.5, color = "gray50", size = 1.2) + 
   geom_path(size = 1.2, color = "#3C5488CC") + 
   theme_light() + 
   labs(x = "1 - Specificity", y = "Sensitivity", title = "ROC for Logistic Regression") +
  theme(plot.title = element_text(size = 16, hjust = 0.5),
        axis.title.x = element_text(size = 15), 
        axis.title.y = element_text(size = 15),
        axis.text.x = element_text(size = 12),
        axis.text.y = element_text(size = 12))

## BART
bart_ROC_plt_processed <- last_bart_fit_processed %>% collect_predictions() %>% 
  roc_curve(Outcome, .pred_Positive) %>% ggplot(aes(x = 1 - specificity, y = sensitivity)) + 
   geom_abline(lty = 2, alpha = 0.5, color = "gray50", size = 1.2) + 
   geom_path(size = 1.2, color = "#F39B7FCC") + 
   theme_light() + 
   labs(x = "1 - Specificity", y = "Sensitivity", title = "ROC for BART") +
  theme(plot.title = element_text(size = 16, hjust = 0.5),
        axis.title.x = element_text(size = 15), 
        axis.title.y = element_text(size = 15),
        axis.text.x = element_text(size = 12),
        axis.text.y = element_text(size = 12))

## Boosted Trees
bt_ROC_plt_processed <- last_bt_fit_processed %>% collect_predictions() %>% 
  roc_curve(Outcome, .pred_Positive) %>% ggplot(aes(x = 1 - specificity, y = sensitivity)) + 
   geom_abline(lty = 2, alpha = 0.5, color = "gray50", size = 1.2) + 
   geom_path(size = 1.2, color = "#8491B4CC") + 
   theme_light() + 
   labs(x = "1 - Specificity", y = "Sensitivity", title = "ROC for Boosted Trees") +
  theme(plot.title = element_text(size = 16, hjust = 0.5),
        axis.title.x = element_text(size = 15), 
        axis.title.y = element_text(size = 15),
        axis.text.x = element_text(size = 12),
        axis.text.y = element_text(size = 12))

## Radial basis function Support Vector Machines(SVM)
svm_ROC_plt_processed <- last_svm_fit_processed %>% collect_predictions() %>% 
  roc_curve(Outcome, .pred_Positive) %>% ggplot(aes(x = 1 - specificity, y = sensitivity)) + 
   geom_abline(lty = 2, alpha = 0.5, color = "gray50", size = 1.2) + 
   geom_path(size = 1.2, color = "#91D1C2CC") + 
   theme_light() + 
   labs(x = "1 - Specificity", y = "Sensitivity", title = "ROC for SVM") +
  theme(plot.title = element_text(size = 16, hjust = 0.5),
        axis.title.x = element_text(size = 15), 
        axis.title.y = element_text(size = 15),
        axis.text.x = element_text(size = 12),
        axis.text.y = element_text(size = 12))

## Multilayer Perceptron Model
mlp_ROC_plt_processed <- last_mlp_fit_processed %>% collect_predictions() %>% 
  roc_curve(Outcome, .pred_Positive) %>% ggplot(aes(x = 1 - specificity, y = sensitivity)) + 
   geom_abline(lty = 2, alpha = 0.5, color = "gray50", size = 1.2) + 
   geom_path(size = 1.2, color = "#DC0000CC") + 
   theme_light() + 
   labs(x = "1 - Specificity", y = "Sensitivity", title = "ROC for MLP") +
   theme(plot.title = element_text(size = 16, hjust = 0.5),
        axis.title.x = element_text(size = 15), 
        axis.title.y = element_text(size = 15),
        axis.text.x = element_text(size = 12),
        axis.text.y = element_text(size = 12))

ggarrange(lr_ROC_plt_processed, nb_ROC_plt_processed, knn_ROC_plt_processed, rf_ROC_plt_processed, bart_ROC_plt_processed, bt_ROC_plt_processed, svm_ROC_plt_processed, mlp_ROC_plt_processed, ncol = 4, nrow = 2)

## Using Raw Data

## Logistic Regression
lr_ROC_plt_raw <- last_lr_fit_raw %>% collect_predictions() %>% 
  roc_curve(Outcome, .pred_Positive) %>% ggplot(aes(x = 1 - specificity, y = sensitivity)) + 
  geom_abline(lty = 2, alpha = 0.5, color = "gray50", size = 1.2) + 
  geom_path(size = 1.2, color = "#E64B35CC") + 
  theme_light() + 
  labs(x = "1 - Specificity", y = "Sensitivity", title = "ROC for Logistic Regression") +
  theme(plot.title = element_text(size = 16, hjust = 0.5),
        axis.title.x = element_text(size = 15), 
        axis.title.y = element_text(size = 15),
        axis.text.x = element_text(size = 12),
        axis.text.y = element_text(size = 12))

## Naive Bayes
nb_ROC_plt_raw <- last_nb_fit_raw %>% collect_predictions() %>% 
  roc_curve(Outcome, .pred_Positive) %>% ggplot(aes(x = 1 - specificity, y = sensitivity)) + 
  geom_abline(lty = 2, alpha = 0.5, color = "gray50", size = 1.2) + 
  geom_path(size = 1.2, color = "#4DBBD5CC") + 
  theme_light() + 
  labs(x = "1 - Specificity", y = "Sensitivity", title = "ROC for Naive Bayes") +
  theme(plot.title = element_text(size = 16, hjust = 0.5),
        axis.title.x = element_text(size = 15), 
        axis.title.y = element_text(size = 15),
        axis.text.x = element_text(size = 12),
        axis.text.y = element_text(size = 12))

## Boosted Trees
bt_ROC_plt_raw <- last_bt_fit_raw %>% collect_predictions() %>% 
  roc_curve(Outcome, .pred_Positive) %>% ggplot(aes(x = 1 - specificity, y = sensitivity)) + 
  geom_abline(lty = 2, alpha = 0.5, color = "gray50", size = 1.2) + 
  geom_path(size = 1.2, color = "#8491B4CC") + 
  theme_light() + 
  labs(x = "1 - Specificity", y = "Sensitivity", title = "ROC for Boosted Trees") +
  theme(plot.title = element_text(size = 16, hjust = 0.5),
        axis.title.x = element_text(size = 15), 
        axis.title.y = element_text(size = 15),
        axis.text.x = element_text(size = 12),
        axis.text.y = element_text(size = 12))

## Multilayer Perceptron Model
mlp_ROC_plt_raw <- last_mlp_fit_raw %>% collect_predictions() %>% 
  roc_curve(Outcome, .pred_Positive) %>% ggplot(aes(x = 1 - specificity, y = sensitivity)) + 
  geom_abline(lty = 2, alpha = 0.5, color = "gray50", size = 1.2) + 
  geom_path(size = 1.2, color = "#DC0000CC") + 
  theme_light() + 
  labs(x = "1 - Specificity", y = "Sensitivity", title = "ROC for MLP") +
  theme(plot.title = element_text(size = 16, hjust = 0.5),
        axis.title.x = element_text(size = 15), 
        axis.title.y = element_text(size = 15),
        axis.text.x = element_text(size = 12),
        axis.text.y = element_text(size = 12))

ggarrange(lr_ROC_plt_raw, nb_ROC_plt_raw, bt_ROC_plt_raw, mlp_ROC_plt_raw, ncol = 4, nrow = 1)

## Last ROC Curves of each Model using Processed Data
last_lr_roc_processed <- last_lr_fit_processed %>% collect_predictions() %>% roc_curve(Outcome, .pred_Positive) %>% mutate(model = "Logistic Regression")
last_nb_roc_processed <- last_nb_fit_processed %>% collect_predictions() %>% roc_curve(Outcome, .pred_Positive) %>% mutate(model = "Navie Bayes")
last_knn_roc_processed <- last_knn_fit_processed %>% collect_predictions() %>% roc_curve(Outcome, .pred_Positive) %>% mutate(model = "K-Nearest Neighbors")
last_rf_roc_processed <- last_rf_fit_processed %>% collect_predictions() %>% roc_curve(Outcome, .pred_Positive) %>% mutate(model = "Random Forests")
last_bart_roc_processed <- last_bart_fit_processed %>% collect_predictions() %>% roc_curve(Outcome, .pred_Positive) %>% mutate(model = "BART")
last_bt_roc_processed <- last_bt_fit_processed %>% collect_predictions() %>% roc_curve(Outcome, .pred_Positive) %>% mutate(model = "Boosted Trees")
last_svm_roc_processed <- last_svm_fit_processed %>% collect_predictions() %>% roc_curve(Outcome, .pred_Positive) %>% mutate(model = "SVM_rbf")
last_mlp_roc_processed <- last_mlp_fit_processed %>% collect_predictions() %>% roc_curve(Outcome, .pred_Positive) %>% mutate(model = "MLP")

bind_rows(last_lr_roc_processed, last_nb_roc_processed, last_knn_roc_processed, last_rf_roc_processed, last_bart_roc_processed, last_bt_roc_processed, last_svm_roc_processed, last_mlp_roc_processed) %>%
  ggplot(aes(x = 1 - specificity, y = sensitivity, col = model)) +
  geom_path(lwd = 1.5, alpha = 0.8) +
  geom_abline(lty = 3) + 
  coord_equal() + theme_light() +
  labs(x="1 - Specificity", y = "Sensitivity") +
  theme(axis.title.x = element_text(size = 15), 
        axis.title.y = element_text(size = 15),
        axis.text.x = element_text(size = 12),
        axis.text.y = element_text(size = 12))  

## Last ROC Curves of each Model using Both Raw Data and Processed Data
last_lr_roc_raw <- last_lr_fit_raw %>% collect_predictions() %>% roc_curve(Outcome, .pred_Positive) %>% mutate(model = "Logistic Regression")
last_nb_roc_raw <- last_nb_fit_raw %>% collect_predictions() %>% roc_curve(Outcome, .pred_Positive) %>% mutate(model = "Navie Bayes")
last_bt_roc_raw <- last_bt_fit_raw %>% collect_predictions() %>% roc_curve(Outcome, .pred_Positive) %>% mutate(model = "Boosted Trees")
last_mlp_roc_raw <- last_mlp_fit_raw %>% collect_predictions() %>% roc_curve(Outcome, .pred_Positive) %>% mutate(model = "MLP")

bind_rows(last_lr_roc_raw, last_nb_roc_raw, last_bt_roc_raw, last_mlp_roc_raw) %>%
  ggplot(aes(x = 1 - specificity, y = sensitivity, col = model)) +
  geom_path(lwd = 1.5, alpha = 0.8) +
  geom_abline(lty = 3) + 
  coord_equal() + theme_light() +
  labs(x="1 - Specificity", y = "Sensitivity") +
  theme(axis.title.x = element_text(size = 15), 
        axis.title.y = element_text(size = 15),
        axis.text.x = element_text(size = 12),
        axis.text.y = element_text(size = 12))

bind_rows(last_lr_roc_processed, last_nb_roc_processed, last_bt_roc_processed, last_mlp_roc_processed) %>%
  ggplot(aes(x = 1 - specificity, y = sensitivity, col = model)) +
  geom_path(lwd = 1.5, alpha = 0.8) +
  geom_abline(lty = 3) + 
  coord_equal() + theme_light() +
  labs(x="1 - Specificity", y = "Sensitivity") +
  theme(axis.title.x = element_text(size = 15), 
        axis.title.y = element_text(size = 15),
        axis.text.x = element_text(size = 12),
        axis.text.y = element_text(size = 12))

## AUC of each model

## Using Processed Data
last_lr_fit_processed %>% collect_predictions() %>% roc_auc(Outcome, .pred_Positive) %>% mutate(model = "Logistic Regression") -> last_auc_processed
last_nb_fit_processed %>% collect_predictions() %>% roc_auc(Outcome, .pred_Positive) %>% mutate(model = "Navie Bayes") %>% rbind(last_auc_processed) -> last_auc_processed
last_knn_fit_processed %>% collect_predictions() %>% roc_auc(Outcome, .pred_Positive) %>% mutate(model = "KNN") %>% rbind(last_auc_processed) -> last_auc_processed
last_rf_fit_processed %>% collect_predictions() %>% roc_auc(Outcome, .pred_Positive) %>% mutate(model = "Random Forests") %>% rbind(last_auc_processed) -> last_auc_processed
last_bart_fit_processed %>% collect_predictions() %>% roc_auc(Outcome, .pred_Positive) %>% mutate(model = "BART") %>% rbind(last_auc_processed) -> last_auc_processed
last_bt_fit_processed %>% collect_predictions() %>% roc_auc(Outcome, .pred_Positive) %>% mutate(model = "Boosted Trees") %>% rbind(last_auc_processed) -> last_auc_processed
last_svm_fit_processed %>% collect_predictions() %>% roc_auc(Outcome, .pred_Positive) %>% mutate(model = "SVM_rbf") %>% rbind(last_auc_processed) -> last_auc_processed
last_mlp_fit_processed %>% collect_predictions() %>% roc_auc(Outcome, .pred_Positive) %>% mutate(model = "MLP") %>% rbind(last_auc_processed) -> last_auc_processed
last_auc_processed[order(-last_auc_processed$.estimate),]

last_auc_processed %>% mutate(.estimate = round(last_auc_processed$.estimate, 3)) %>% ggplot(aes(x = model, y = .estimate, fill = model)) +
  geom_bar(alpha = 0.8, width = 0.6, stat = "identity") +
  labs(x = "Models", y = "AUC") +
  geom_text(aes(label = .estimate), size = 3.6, hjust = 0.5, vjust = -1) +
  theme_light() + theme(plot.title=element_text(hjust=0.5)) +
  theme(axis.title.x = element_text(size = 15), 
        axis.title.y = element_text(size = 15),
        axis.text.x = element_text(size = 12),
        axis.text.y = element_text(size = 12))

## Using Both Raw Data & Processed Data

last_lr_fit_raw %>% collect_predictions() %>% roc_auc(Outcome, .pred_Positive) %>% mutate(model = "Logistic Regression") -> last_auc_raw
last_nb_fit_raw %>% collect_predictions() %>% roc_auc(Outcome, .pred_Positive) %>% mutate(model = "Navie Bayes") %>% rbind(last_auc_raw) -> last_auc_raw
last_bt_fit_raw %>% collect_predictions() %>% roc_auc(Outcome, .pred_Positive) %>% mutate(model = "Boosted Trees") %>% rbind(last_auc_raw) -> last_auc_raw
last_mlp_fit_raw %>% collect_predictions() %>% roc_auc(Outcome, .pred_Positive) %>% mutate(model = "MLP") %>% rbind(last_auc_raw) -> last_auc_raw
last_auc_raw[order(-last_auc_raw$.estimate),]

auc_comparison <- last_auc_raw %>% mutate(DataUsed = "Raw Data")
last_auc_processed[last_auc_processed$model=="Logistic Regression",] %>%
  rbind(last_auc_processed[last_auc_processed$model=="Navie Bayes",]) %>%
  rbind(last_auc_processed[last_auc_processed$model=="Boosted Trees",]) %>%
  rbind(last_auc_processed[last_auc_processed$model=="MLP",]) %>% 
  mutate(DataUsed = "Processed Data") %>%
  rbind(auc_comparison) %>%
  mutate(.estimate = round(.$.estimate, 3)) %>%
  ggplot(aes(x=model, y=.estimate, fill=DataUsed)) + geom_bar(alpha = 0.8, width = 0.6,stat="identity", position="dodge") +
  scale_fill_manual(values=c("#0073C2FF","#EFC000FF")) +
  labs(x="Models", y="AUC") +
  geom_text(aes(label = .estimate), position=position_dodge(width=0.5), size = 3.6, hjust = 0.5, vjust = -1) +
  theme_light()

# Variable Importance Plot of lr, bt, mlp

last_lr_fit_processed %>% extract_fit_parsnip() %>% 
  vip(mapping = aes_string(fill = "Variable")) + 
  theme_light() + 
  theme(axis.title.x = element_text(size = 15),
        axis.text.x = element_text(size = 12),
        axis.text.y = element_text(size = 12))
last_bt_fit_processed %>% extract_fit_parsnip() %>% 
  vip(mapping = aes_string(fill = "Variable")) + 
  theme_light() + 
  theme(axis.title.x = element_text(size = 15),
        axis.text.x = element_text(size = 12),
        axis.text.y = element_text(size = 12))
last_mlp_fit_processed %>% extract_fit_parsnip() %>% 
  vip(mapping = aes_string(fill = "Variable")) + 
  theme_light() + 
  theme(axis.title.x = element_text(size = 15),
        axis.text.x = element_text(size = 12),
        axis.text.y = element_text(size = 12))













