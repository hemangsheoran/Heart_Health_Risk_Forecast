import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import pickle
x_train = pd.read_csv("x_train.csv")
x_test = pd.read_csv("x_test.csv")
y_train = pd.read_csv("y_train.csv")
y_test = pd.read_csv("y_test.csv")

import lightgbm as lgb
from sklearn.metrics import accuracy_score



# Create the LightGBM dataset
train_data = lgb.Dataset(x_train, label=y_train)
test_data = lgb.Dataset(x_test, label=y_test)

# Define the parameters for the LightGBM model
params = {
    'objective': 'binary',
    'metric': 'binary_error',  # We use binary error as the evaluation metric
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0,
    'early_stopping_rounds': 10
}

# Train the LightGBM model
num_round = 100
bst = lgb.train(params, train_data, num_round, valid_sets=[test_data])
y_predtrain = bst.predict(x_train, num_iteration=bst.best_iteration)
y_pred = bst.predict(x_test, num_iteration=bst.best_iteration)
y_pred_binary = [1 if pred >= 0.5 else 0 for pred in y_pred]  # Convert probabilities to binary predictions

cattrain = pd.DataFrame({'Lightgbm': y_predtrain})
cattest = pd.DataFrame({'Lightgbm': y_pred})
# Evaluate the model
accuracy = accuracy_score(y_test, y_pred_binary)
print("Test Accuracy:", accuracy)
print(y_pred)
print(y_predtrain)

# Create Naive Bayes model
nb_model = GaussianNB()

# Train the model
nb_model.fit(cattrain, y_train.values.ravel())

# Predict on the test set
y_pred = nb_model.predict(cattest)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)




with open("naive_bayes_lightGBM.pkl", "wb") as f:
    pickle.dump(nb_model, f)



