# Importing essential libraries
import pandas as pd
import joblib
import lightgbm as lgb
from catboost import CatBoostClassifier

import pickle


from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from joblib import load
from xgboost import XGBClassifier
scaler = StandardScaler()



svm_2_layer = load('2_layer_svm.pkl')
cnn_2_layer = load_model('2_layer_cnn.keras')

svm_4_layer = load('svm_4_layer.pkl')
cnn_4_layer = load_model('cnn_4_layer.keras')

svm_8_layer = load('svm_8_layer.pkl')
cnn_8_layer = load_model('cnn_8_layer.keras')

svm_16_layer = load('svm_16_layer.pkl')
cnn_16_layer = load_model('cnn_16_layer.keras')

svm_24_layer = load('svm_24_layer.pkl')
cnn_24_layer = load_model('cnn_24_layer.keras')

svm_final = load('final_svm_linking.pkl')
cnn_final = load_model('final_cnn_linking.keras')


xgmodel = XGBClassifier()
xgmodel.load_model('xgboost_model.json')

logistic_model = joblib.load('logistic_regression_model.pkl')


# Branch 2
#first hybrid
loaded_rf_1_1 = joblib.load('random_forest_1_1.pkl')
loaded_lr_1_2 = joblib.load('logistic_regression_1_2.pkl')

#second hybrid
loaded_dt_2_1 = joblib.load('decision_tree_2_1.pkl')
loaded_lr_2_2 = joblib.load('logistic_regression_2_2.pkl')


#cnn_logistic
cnn_logistic = load_model('cnn_logistic.keras')
loaded_cnn_lr = joblib.load('cnn_logistic_regression.pkl')

#Lightgbm_catboost
lightgbm_catboost_loaded = lgb.Booster(model_file='lightgbm_catboost.txt')
catboost_lightgbm_loaded = CatBoostClassifier()
catboost_lightgbm_loaded.load_model('catboost_lightgbm')


#lightgbm_naivebyes
with open("naive_bayes_lightGBM.pkl", "rb") as f:
    loaded_naive = pickle.load(f)


app = Flask(__name__)


@app.route('/')
def home():
    return render_template('main.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        age = int(request.form['age'])
        sex = request.form.get('sex')
        cp = request.form.get('cp')
        trestbps = int(request.form['trestbps'])
        chol = int(request.form['chol'])
        fbs = request.form.get('fbs')
        restecg = int(request.form['restecg'])
        thalach = int(request.form['thalach'])
        exang = request.form.get('exang')
        oldpeak = float(request.form['oldpeak'])
        slope = request.form.get('slope')




        data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope]])

        columns = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina',
                   'Oldpeak', 'ST_Slope']
        pddata = pd.DataFrame(data, columns=columns)


        pd1data = pddata.astype({
            'Age': np.int64,
            'Sex': np.int64,
            'ChestPainType': np.int64,
            'RestingBP': np.int64,
            'Cholesterol': np.int64,
            'FastingBS': np.int64,
            'RestingECG': np.int64,
            'MaxHR': np.int64,
            'ExerciseAngina': np.int64,
            'Oldpeak': float,
            'ST_Slope': np.int64
        })
        x_test = pd.read_csv('x_test.csv')
        #x_train = pd.read_csv('x_train.csv')
        #x_test.info()
        x_test_lightgbm = pd.concat([x_test, pd1data], ignore_index=True)
        x_test = pd.concat([x_test, pddata], ignore_index=True)
        #x_test.info()
        #print(pd1data)
        #print(x_test)
        user_input_scaled = scaler.fit_transform(x_test)
        print(user_input_scaled[150])
        svm_model_test_predictions = []
        start_2_layer = cnn_2_layer.predict(user_input_scaled)
        result_2_layer = svm_2_layer.predict(start_2_layer)
        print(result_2_layer[150])
        svm_model_test_predictions.append([result_2_layer[150]])

        start_4_layer = cnn_4_layer.predict(user_input_scaled)
        result_4_layer = svm_4_layer.predict(start_4_layer)
        print(result_4_layer[150])
        svm_model_test_predictions.append([result_4_layer[150]])
        start_8_layer = cnn_8_layer.predict(user_input_scaled)
        result_8_layer = svm_8_layer.predict(start_8_layer)
        print(result_8_layer[150])
        svm_model_test_predictions.append([result_8_layer[150]])

        start_16_layer = cnn_16_layer.predict(user_input_scaled)
        result_16_layer = svm_16_layer.predict(start_16_layer)
        print(result_16_layer[150])
        svm_model_test_predictions.append([result_16_layer[150]])

        start_24_layer = cnn_24_layer.predict(user_input_scaled)
        result_24_layer = svm_24_layer.predict(start_24_layer)
        print(result_24_layer[150])
        svm_model_test_predictions.append([result_24_layer[150]])

        y_pred = xgmodel.predict(pd1data)
        print(y_pred)

        svm_model_test_predictions.append(y_pred)
        predictions_logistic = logistic_model.predict(pd1data)
        print(predictions_logistic)
        svm_model_test_predictions.append(predictions_logistic)

        #svm_model_test_predictions = [[result_2_layer[150],result_4_layer[150],result_8_layer[150],result_16_layer[150],result_24_layer[150],y_pred[0],predictions_logistic[0]]]
        print(svm_model_test_predictions)
        combined_outputs_test = np.array(svm_model_test_predictions).T
        print(combined_outputs_test)
        cnn_predictions_test = cnn_final.predict(combined_outputs_test)
        final_result = svm_final.predict(cnn_predictions_test)
        print(final_result[0])






        #Second Branch

        # Load Logistic Regression model
        print(x_test)
        predictions = loaded_rf_1_1.predict(x_test)
        x_test1 = x_test.copy()
        x_test1['rf_predictions'] = predictions
        lr_predictions_first = loaded_lr_1_2.predict(x_test1)


        #second
        print(x_test)
        predictions1 = loaded_dt_2_1.predict(x_test)
        x_test2 = x_test.copy()
        x_test2['second_predictions'] = predictions1
        predictions_second = loaded_lr_2_2.predict(x_test2)



        #cnn_logistic_regression
        cnn_logistic1 = cnn_logistic.predict(user_input_scaled)
        predictions_cnn_logistic = loaded_cnn_lr.predict(cnn_logistic1)

        # Lightgbm_catboost
        y_preds = lightgbm_catboost_loaded.predict(x_test_lightgbm, num_iteration=lightgbm_catboost_loaded.best_iteration)
        cattest1 = pd.DataFrame({'Lightgbm': y_preds})
        y_predset = catboost_lightgbm_loaded.predict(cattest1)



        #lIGHTGBM_naive
        y_predst = lightgbm_catboost_loaded.predict(x_test_lightgbm, num_iteration=lightgbm_catboost_loaded.best_iteration)
        cattest2 = pd.DataFrame({'Lightgbm': y_predst})
        naive_result = loaded_naive.predict(cattest2)

        final = result_2_layer[150] + result_4_layer[150] + result_8_layer[150] + result_16_layer[150] + result_24_layer[150] + y_pred[0] + predictions_logistic[0]



        accuracy = (final / 7) * 100


        print("First Branch----------")
        print("Layer 2 result", result_2_layer[150])
        print("Layer 4 result", result_4_layer[150])
        print("Layer 8 result", result_8_layer[150])
        print("Layer 16 result", result_16_layer[150])
        print("Layer 24 result", result_24_layer[150])
        print("Xgboost result", y_pred[0])
        print("Logistic Regression result", predictions_logistic[0])


        print("Second Branch---------")

        print("Random Forest + Logistic Regression", lr_predictions_first[150])
        print("Decision Tree + Logistic Regression", predictions_second[150])
        print("CNN + Logistic Regression", predictions_cnn_logistic[150])
        print("LightGBM + Catboost", y_predset[150])
        print("LightGBM + Naive_bayes", naive_result[150])
        




        return render_template('result.html', prediction=final_result[0], accurate=accuracy, layer_2=result_2_layer[150], layer_4=result_4_layer[150], layer_8=result_8_layer[150], layer_16=result_16_layer[150], layer_24=result_24_layer[150], xgboost = y_pred[0], logistic = predictions_logistic[0], second_first = lr_predictions_first[150], second_second = predictions_second[150], cnn_logisticregr = predictions_cnn_logistic[150], lightgbm_catboo = y_predset[150], naive_bayes = naive_result[150] )


if __name__ == '__main__':
    app.run(debug=True)
