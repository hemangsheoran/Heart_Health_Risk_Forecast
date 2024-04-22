import pandas as pd
import numpy as np
import lightgbm as lgb
from catboost import CatBoostClassifier
import pickle
from tensorflow.keras.layers import LSTM
import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from joblib import dump
from sklearn.svm import SVC
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import joblib
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from joblib import load
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

x_train = pd.read_csv('x_train.csv')
y_train = pd.read_csv('y_train.csv')

x_test = pd.read_csv('x_test.csv')
y_test = pd.read_csv('y_test.csv')


scaler = StandardScaler()
train_input_scaled = scaler.fit_transform(x_train)
test_input_scaled = scaler.fit_transform(x_test)


train_first_predictions = []
train_second_predictions = []
train_both_predictions = []
train_final_predictions = []

test_first_predictions = []
test_second_predictions = []
test_both_predictions = []
test_final_predictions = []


#FIRST_BRANCH

#cnn+svm 2 layer model
svm_2_layer = load('2_layer_svm.pkl')
cnn_2_layer = load_model('2_layer_cnn.keras')

train_start_2_layer = cnn_2_layer.predict(train_input_scaled)
train_result_2_layer = svm_2_layer.predict(train_start_2_layer)
train_first_predictions.append(train_result_2_layer)
train_both_predictions.append(train_result_2_layer)

test_start_2_layer = cnn_2_layer.predict(test_input_scaled)
test_result_2_layer = svm_2_layer.predict(test_start_2_layer)
test_first_predictions.append(test_result_2_layer)
test_both_predictions.append(test_result_2_layer)

accuracy_2_layer = accuracy_score(y_test, test_result_2_layer)
print(f'Accuracy of 2 Layer cnn + svm Hybrid Model: {accuracy_2_layer}')



#cnn+svm 4 layer model
svm_4_layer = load('svm_4_layer.pkl')
cnn_4_layer = load_model('cnn_4_layer.keras')

train_start_4_layer = cnn_4_layer.predict(train_input_scaled)
train_result_4_layer = svm_4_layer.predict(train_start_4_layer)
train_first_predictions.append(train_result_4_layer)
train_both_predictions.append(train_result_4_layer)

test_start_4_layer = cnn_4_layer.predict(test_input_scaled)
test_result_4_layer = svm_4_layer.predict(test_start_4_layer)
test_first_predictions.append(test_result_4_layer)
test_both_predictions.append(test_result_4_layer)

accuracy_4_layer = accuracy_score(y_test, test_result_4_layer)
print(f'Accuracy of 4 Layer cnn + svm Hybrid Model: {accuracy_4_layer}')




#cnn+svm 8 layer model
svm_8_layer = load('svm_8_layer.pkl')
cnn_8_layer = load_model('cnn_8_layer.keras')

train_start_8_layer = cnn_8_layer.predict(train_input_scaled)
train_result_8_layer = svm_8_layer.predict(train_start_8_layer)
train_first_predictions.append(train_result_8_layer)
train_both_predictions.append(train_result_8_layer)

test_start_8_layer = cnn_8_layer.predict(test_input_scaled)
test_result_8_layer = svm_8_layer.predict(test_start_8_layer)
test_first_predictions.append(test_result_8_layer)
test_both_predictions.append(test_result_8_layer)

accuracy_8_layer = accuracy_score(y_test, test_result_8_layer)
print(f'Accuracy of 8 Layer cnn + svm Hybrid Model: {accuracy_8_layer}')




#cnn+svm 16 layer model
svm_16_layer = load('svm_16_layer.pkl')
cnn_16_layer = load_model('cnn_16_layer.keras')

train_start_16_layer = cnn_16_layer.predict(train_input_scaled)
train_result_16_layer = svm_16_layer.predict(train_start_16_layer)
train_first_predictions.append(train_result_16_layer)
train_both_predictions.append(train_result_16_layer)

test_start_16_layer = cnn_16_layer.predict(test_input_scaled)
test_result_16_layer = svm_16_layer.predict(test_start_16_layer)
test_first_predictions.append(test_result_16_layer)
test_both_predictions.append(test_result_16_layer)

accuracy_16_layer = accuracy_score(y_test, test_result_16_layer)
print(f'Accuracy of 16 Layer cnn + svm Hybrid Model: {accuracy_16_layer}')



#cnn+svm 24 layer model
svm_24_layer = load('svm_24_layer.pkl')
cnn_24_layer = load_model('cnn_24_layer.keras')

train_start_24_layer = cnn_24_layer.predict(train_input_scaled)
train_result_24_layer = svm_24_layer.predict(train_start_24_layer)
train_first_predictions.append(train_result_24_layer)
train_both_predictions.append(train_result_24_layer)

test_start_24_layer = cnn_24_layer.predict(test_input_scaled)
test_result_24_layer = svm_24_layer.predict(test_start_24_layer)
test_first_predictions.append(test_result_24_layer)
test_both_predictions.append(test_result_24_layer)

accuracy_24_layer = accuracy_score(y_test, test_result_24_layer)
print(f'Accuracy of 24 Layer cnn + svm Hybrid Model: {accuracy_24_layer}')




#XGboost Model
xgmodel = XGBClassifier()
xgmodel.load_model('xgboost_model.json')

train_result_xgmodel = xgmodel.predict(x_train)
train_first_predictions.append(train_result_xgmodel)
train_both_predictions.append(train_result_xgmodel)

test_result_xgmodel = xgmodel.predict(x_test)
test_first_predictions.append(test_result_xgmodel)
test_both_predictions.append(test_result_xgmodel)

accuracy_xgmodel = accuracy_score(y_test, test_result_xgmodel)
print(f'Accuracy of XGBoost Model: {accuracy_xgmodel}')




#Logistic Regression Model
logistic_model = joblib.load('logistic_regression_model.pkl')

train_result_logistic = logistic_model.predict(x_train)
train_first_predictions.append(train_result_logistic)
train_both_predictions.append(train_result_logistic)

test_result_logistic = logistic_model.predict(x_test)
test_first_predictions.append(test_result_logistic)
test_both_predictions.append(test_result_logistic)

accuracy_logistic = accuracy_score(y_test, test_result_logistic)
print(f'Accuracy of Logistic Regression Model: {accuracy_logistic}')



#First Branch Final
svm_first_branch_final = load('final_svm_linking.pkl')
cnn_first_branch_final = load_model('final_cnn_linking.keras')
train_first_branch_array_combined = np.array(train_first_predictions).T

train_first_branch = cnn_first_branch_final.predict(train_first_branch_array_combined)
train_final_result_first_branch = svm_first_branch_final.predict(train_first_branch)
train_final_predictions.append(train_final_result_first_branch)

test_first_branch_array_combined = np.array(test_first_predictions).T
test_first_branch = cnn_first_branch_final.predict(test_first_branch_array_combined)
test_final_result_first_branch = svm_first_branch_final.predict(test_first_branch)
test_final_predictions.append(test_final_result_first_branch)

accuracy_first_branch_final = accuracy_score(y_test, test_final_result_first_branch)
print(f'Accuracy of First Branch Final Model is: {accuracy_first_branch_final}')





#SECOND BRANCH

#random forest + logistic regression model
loaded_rf_1_1 = joblib.load('random_forest_1_1.pkl')
loaded_lr_1_2 = joblib.load('logistic_regression_1_2.pkl')

train_rf_predictions = loaded_rf_1_1.predict(x_train)
x_train_rf = x_train.copy()
x_train_rf['rf_predictions'] = train_rf_predictions
train_rf_lr_predictions_first = loaded_lr_1_2.predict(x_train_rf)
train_second_predictions.append(train_rf_lr_predictions_first)
train_both_predictions.append(train_rf_lr_predictions_first)

test_rf_predictions = loaded_rf_1_1.predict(x_test)
x_test_rf = x_test.copy()
x_test_rf['rf_predictions'] = test_rf_predictions
test_rf_lr_predictions_first = loaded_lr_1_2.predict(x_test_rf)
test_second_predictions.append(test_rf_lr_predictions_first)
test_both_predictions.append(test_rf_lr_predictions_first)

accuracy_rf_lr = accuracy_score(y_test, test_rf_lr_predictions_first)
print(f'Accuracy of Random Forest + Logistic Regression Model is: {accuracy_rf_lr}')


#Decision Tree + logistic regression model
loaded_dt_2_1 = joblib.load('decision_tree_2_1.pkl')
loaded_lr_2_2 = joblib.load('logistic_regression_2_2.pkl')

train_dt_predictions = loaded_dt_2_1.predict(x_train)
x_train_dt = x_train.copy()
x_train_dt['second_predictions'] = train_dt_predictions
train_dt_lr_predictions_first = loaded_lr_2_2.predict(x_train_dt)
train_second_predictions.append(train_dt_lr_predictions_first)
train_both_predictions.append(train_dt_lr_predictions_first)

test_dt_predictions = loaded_dt_2_1.predict(x_test)
x_test_dt = x_test.copy()
x_test_dt['second_predictions'] = test_dt_predictions
test_dt_lr_predictions_first = loaded_lr_2_2.predict(x_test_dt)
test_second_predictions.append(test_dt_lr_predictions_first)
test_both_predictions.append(test_dt_lr_predictions_first)

accuracy_dt_lr = accuracy_score(y_test, test_dt_lr_predictions_first)
print(f'Accuracy of Decision Tree + Logistic Regression Model is: {accuracy_dt_lr}')



#cnn + logistic model second branch
cnn_logistic = load_model('cnn_logistic.keras')
loaded_cnn_lr = joblib.load('cnn_logistic_regression.pkl')

train_cnn_logistic1 = cnn_logistic.predict(train_input_scaled)
train_predictions_cnn_logistic = loaded_cnn_lr.predict(train_cnn_logistic1)
train_second_predictions.append(train_predictions_cnn_logistic)
train_both_predictions.append(train_predictions_cnn_logistic)

test_cnn_logistic1 = cnn_logistic.predict(test_input_scaled)
test_predictions_cnn_logistic = loaded_cnn_lr.predict(test_cnn_logistic1)
test_second_predictions.append(test_predictions_cnn_logistic)
test_both_predictions.append(test_predictions_cnn_logistic)

accuracy_cn_lr = accuracy_score(y_test, test_predictions_cnn_logistic)
print(f'Accuracy of CNN + Logistic Regression Model is: {accuracy_cn_lr}')



#LightGBM + Catboost model
lightgbm_model = lgb.Booster(model_file='lightgbm_catboost.txt')
catboost_lightgbm_loaded = CatBoostClassifier()
catboost_lightgbm_loaded.load_model('catboost_lightgbm')

train_catboost_first = lightgbm_model.predict(x_train, num_iteration=lightgbm_model.best_iteration)
train_cat_first = pd.DataFrame({'Lightgbm': train_catboost_first})
train_catboost_final = catboost_lightgbm_loaded.predict(train_cat_first)
train_second_predictions.append(train_catboost_final)
train_both_predictions.append(train_catboost_final)

test_catboost_first = lightgbm_model.predict(x_test, num_iteration=lightgbm_model.best_iteration)
test_cat_first = pd.DataFrame({'Lightgbm': test_catboost_first})
test_catboost_final = catboost_lightgbm_loaded.predict(test_cat_first)
test_second_predictions.append(test_catboost_final)
test_both_predictions.append(test_catboost_final)

accuracy_lightgbm_catboost = accuracy_score(y_test, test_catboost_final)
print(f'Accuracy of Lightgbm + Catboost Model is: {accuracy_lightgbm_catboost}')


#LightGBM + Naive bayes model
with open("naive_bayes_lightGBM.pkl", "rb") as f:
    loaded_naive = pickle.load(f)

train_naive_bayes_first = lightgbm_model.predict(x_train, num_iteration=lightgbm_model.best_iteration)
train_naive_first = pd.DataFrame({'Lightgbm': train_naive_bayes_first})
train_naive_final = loaded_naive.predict(train_naive_first)
train_second_predictions.append(train_naive_final)
train_both_predictions.append(train_naive_final)

test_naive_bayes_first = lightgbm_model.predict(x_test, num_iteration=lightgbm_model.best_iteration)
test_naive_first = pd.DataFrame({'Lightgbm': test_naive_bayes_first})
test_naive_final = loaded_naive.predict(test_naive_first)
test_second_predictions.append(test_naive_final)
test_both_predictions.append(test_naive_final)

accuracy_lightgbm_naive_bayes = accuracy_score(y_test, test_naive_final)
print(f'Accuracy of Lightgbm + Naive Bayes Model is: {accuracy_lightgbm_naive_bayes}')



# print(train_second_branch_array_combined)
# train_data1 = lgb.Dataset(train_second_branch_array_combined, label=y_train)
# val_data1 = lgb.Dataset(test_second_branch_array_combined, label=y_test, reference=train_data1)
# params = {
#     'objective': 'binary',
#     'metric': 'binary_error',
#     'num_leaves': 31,
#     'learning_rate': 0.05,
#     'feature_fraction': 0.9,
#     'bagging_fraction': 0.8,
#     'bagging_freq': 5,
#     'verbose': 0,
#     'early_stopping_rounds': 10
# }
# num_round = 100
# bst = lgb.train(params, train_data1, num_round, valid_sets=[val_data1])
#
# # Predict on the validation set
# y_pred2 = bst.predict(test_second_branch_array_combined, num_iteration=bst.best_iteration)
# y_pred_binary = [1 if pred > 0.5 else 0 for pred in y_pred2]
# accuracy = accuracy_score(y_test, y_pred_binary)
# print("Validation Accuracy:", accuracy)
#
#
# dtrain = xgb.DMatrix(train_second_branch_array_combined, label=y_train)
# dval = xgb.DMatrix(test_second_branch_array_combined, label=y_test)
#
# # Define parameters for XGBoost
# params = {
#     'objective': 'binary:logistic',
#     'eval_metric': 'error',  # You can use other evaluation metrics like 'auc' or 'logloss'
#     'max_depth': 6,
#     'eta': 0.3,
#     'subsample': 0.8,
#     'colsample_bytree': 0.8,
#     'verbosity': 0
# }
#
# # Train the XGBoost model
# num_rounds = 100
# bst = xgb.train(params, dtrain, num_rounds, evals=[(dval, 'eval')], early_stopping_rounds=10)
#
# # Predict on the validation set
# y_pred_prob = bst.predict(dval)
# y_pred_binary2 = [1 if pred > 0.5 else 0 for pred in y_pred_prob]
#
# # Evaluate the model
# accuracy = accuracy_score(y_test, y_pred_binary2)
# print("Validation Accuracy2:", accuracy)
#
#
#
# log_reg_model = LogisticRegression()
# log_reg_model.fit(train_second_branch_array_combined, y_train)
#
# # Predict on the validation set
# y_pred = log_reg_model.predict(test_second_branch_array_combined)
#
# # Evaluate the model
# accuracy = accuracy_score(y_test, y_pred)
# print("Validation Accuracy 3:", accuracy)
#
#
#
# decision_tree_model = DecisionTreeClassifier(random_state=42)
# decision_tree_model.fit(train_second_branch_array_combined, y_train)
#
# # Predict on the validation set
# y_pred7 = decision_tree_model.predict(test_second_branch_array_combined)
#
# # Evaluate the model
# accuracy = accuracy_score(y_test, y_pred7)
# print("Validation Accuracy 4:", accuracy)

# #Train second Branch Final Model

# cnn_final4_layer = Sequential()
# cnn_final4_layer.add(Dense(64, activation='relu', input_shape=(train_second_branch_array_combined.shape[1],)))
# cnn_final4_layer.add(Dense(32, activation='relu'))
# cnn_final4_layer.add(Dense(16, activation='relu'))
# cnn_final4_layer.add(Dense(1, activation='sigmoid'))
# cnn_final4_layer.compile(optimizer='SGD', loss='binary_crossentropy', metrics=['accuracy'])
# cnn_final4_layer.fit(train_second_branch_array_combined, y_train, epochs=10, batch_size=32, validation_data=(test_second_branch_array_combined, y_test))
# final_result = cnn_final4_layer.predict(test_second_branch_array_combined)
# threshold = 0.5
# final_result_binary = np.where(final_result >= threshold, 1, 0)
# accuracy6 = accuracy_score(y_test, final_result_binary)
# print(f'Accuracy on the cnn test set: {accuracy6}')
# cnn_final4_layer.save('second_branch_final_cnn_model.keras')

#Second layer cnn final model
train_second_branch_array_combined = np.array(train_second_predictions).T
test_second_branch_array_combined = np.array(test_second_predictions).T
second_branch_final_cnn_model = load_model('second_branch_final_cnn_model.keras')
train_second_branch_result = second_branch_final_cnn_model.predict(train_second_branch_array_combined)
threshold = 0.5
train_final_result_binary = np.where(train_second_branch_result >= threshold, 1, 0)
train_final_result_binary = train_final_result_binary.flatten()
train_final_predictions.append(train_final_result_binary)
test_second_branch_result = second_branch_final_cnn_model.predict(test_second_branch_array_combined)
test_final_result_binary = np.where(test_second_branch_result >= threshold, 1, 0)
test_final_result_binary = test_final_result_binary.flatten()
test_final_predictions.append(test_final_result_binary)
accuracy6 = accuracy_score(y_test, test_final_result_binary)
print(f'Accuracy of second branch Final cnn model: {accuracy6}')


#First + Second Branch combined output model
both_branch_final_model = joblib.load('both_branch_final_logistic_model.pkl')
train_both_branch_array_combined = np.array(train_both_predictions).T
test_both_branch_array_combined = np.array(test_both_predictions).T

train_both_branch_result = both_branch_final_model.predict(train_both_branch_array_combined)
train_final_predictions.append(train_both_branch_result)
test_both_branch_result = both_branch_final_model.predict(test_both_branch_array_combined)
test_final_predictions.append(test_both_branch_result)
accuracy = accuracy_score(y_test, test_both_branch_result)
print("Accuracy of both Branches Logistic Regression Model:", accuracy)




# Final Model
train_final_combined = np.array(train_final_predictions).T
test_final_combined = np.array(test_final_predictions).T


Final_last_cnn_5_layer_model = load_model('Final_last_CNN_5_LAYER_model.keras')
final_result_after_all_model = Final_last_cnn_5_layer_model.predict(test_final_combined)

final_result_binary = np.where(final_result_after_all_model >= 0.5, 1, 0)
final_result_binary = final_result_binary.flatten()

accuracy = accuracy_score(y_test, final_result_binary)
print("Final Accuracy:", accuracy)

conf_matrix = confusion_matrix(y_test, final_result_binary)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
print(classification_report(y_test, final_result_binary))
plt.show()















