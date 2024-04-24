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


# Second Branch Final Model
second_branch_final_cnn_model = load_model('second_branch_final_cnn_model.keras')


# Both Branch Final Model
both_branch_final_model = joblib.load('both_branch_final_logistic_model.pkl')



# Final Model
Final_last_cnn_5_layer_model = load_model('Final_last_CNN_5_LAYER_model.keras')


















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
        print("User Input Provided", user_input_scaled[150])
        first_branch_final_test_predictions = []
        second_branch_final_test_predictions = []
        both_branch_final_test_predictions = []
        final_layer_predictions = []


        start_2_layer = cnn_2_layer.predict(user_input_scaled)
        result_2_layer = svm_2_layer.predict(start_2_layer)
        first_branch_final_test_predictions.append([result_2_layer[150]])
        both_branch_final_test_predictions.append([result_2_layer[150]])


        start_4_layer = cnn_4_layer.predict(user_input_scaled)
        result_4_layer = svm_4_layer.predict(start_4_layer)

        first_branch_final_test_predictions.append([result_4_layer[150]])
        both_branch_final_test_predictions.append([result_4_layer[150]])


        start_8_layer = cnn_8_layer.predict(user_input_scaled)
        result_8_layer = svm_8_layer.predict(start_8_layer)

        first_branch_final_test_predictions.append([result_8_layer[150]])
        both_branch_final_test_predictions.append([result_8_layer[150]])


        start_16_layer = cnn_16_layer.predict(user_input_scaled)
        result_16_layer = svm_16_layer.predict(start_16_layer)

        first_branch_final_test_predictions.append([result_16_layer[150]])
        both_branch_final_test_predictions.append([result_16_layer[150]])


        start_24_layer = cnn_24_layer.predict(user_input_scaled)
        result_24_layer = svm_24_layer.predict(start_24_layer)

        first_branch_final_test_predictions.append([result_24_layer[150]])
        both_branch_final_test_predictions.append([result_24_layer[150]])


        y_pred = xgmodel.predict(pd1data)

        first_branch_final_test_predictions.append(y_pred)
        both_branch_final_test_predictions.append(y_pred)


        predictions_logistic = logistic_model.predict(pd1data)

        first_branch_final_test_predictions.append(predictions_logistic)
        both_branch_final_test_predictions.append(predictions_logistic)

        #svm_model_test_predictions = [[result_2_layer[150],result_4_layer[150],result_8_layer[150],result_16_layer[150],result_24_layer[150],y_pred[0],predictions_logistic[0]]]
        print("First branch final Result sheet", first_branch_final_test_predictions)
        first_branch_combined_outputs_test = np.array(first_branch_final_test_predictions).T
        print(first_branch_combined_outputs_test)
        cnn_predictions_test = cnn_final.predict(first_branch_combined_outputs_test)
        final_result = svm_final.predict(cnn_predictions_test)

        final_layer_predictions.append([final_result[0]])







        #Second Branch

        # Load Logistic Regression model
        print("x_test Full input rows", x_test)
        predictions = loaded_rf_1_1.predict(x_test)
        x_test1 = x_test.copy()
        x_test1['rf_predictions'] = predictions
        lr_predictions_first = loaded_lr_1_2.predict(x_test1)
        second_branch_final_test_predictions.append([lr_predictions_first[150]])
        both_branch_final_test_predictions.append([lr_predictions_first[150]])


        #second

        predictions1 = loaded_dt_2_1.predict(x_test)
        x_test2 = x_test.copy()
        x_test2['second_predictions'] = predictions1
        predictions_second = loaded_lr_2_2.predict(x_test2)
        second_branch_final_test_predictions.append([predictions_second[150]])
        both_branch_final_test_predictions.append([predictions_second[150]])



        #cnn_logistic_regression
        cnn_logistic1 = cnn_logistic.predict(user_input_scaled)
        predictions_cnn_logistic = loaded_cnn_lr.predict(cnn_logistic1)
        second_branch_final_test_predictions.append([predictions_cnn_logistic[150]])
        both_branch_final_test_predictions.append([predictions_cnn_logistic[150]])

        # Lightgbm_catboost
        y_preds = lightgbm_catboost_loaded.predict(x_test_lightgbm, num_iteration=lightgbm_catboost_loaded.best_iteration)
        cattest1 = pd.DataFrame({'Lightgbm': y_preds})
        y_predset = catboost_lightgbm_loaded.predict(cattest1)
        second_branch_final_test_predictions.append([y_predset[150]])
        both_branch_final_test_predictions.append([y_predset[150]])



        #lIGHTGBM_naive
        y_predst = lightgbm_catboost_loaded.predict(x_test_lightgbm, num_iteration=lightgbm_catboost_loaded.best_iteration)
        cattest2 = pd.DataFrame({'Lightgbm': y_predst})
        naive_result = loaded_naive.predict(cattest2)
        second_branch_final_test_predictions.append([naive_result[150]])
        both_branch_final_test_predictions.append([naive_result[150]])


        print("Second Branch all result array", second_branch_final_test_predictions)
        print("Both branch full sheet of Result", both_branch_final_test_predictions)



        #Second Branch Final Model
        train_second_branch_array_combined = np.array(second_branch_final_test_predictions).T

        train_second_branch_result = second_branch_final_cnn_model.predict(train_second_branch_array_combined)

        train_final_result_binary = np.where(train_second_branch_result >= 0.5, 1, 0)
        train_final_result_binary = train_final_result_binary.flatten()
        final_layer_predictions.append(train_final_result_binary)






        #both branch final model


        train_both_branch_array_combined = np.array(both_branch_final_test_predictions).T
        train_both_branch_result = both_branch_final_model.predict(train_both_branch_array_combined)
        votesss_1 = train_both_branch_array_combined[0][1]+train_both_branch_array_combined[0][2]+train_both_branch_array_combined[0][3]+train_both_branch_array_combined[0][4]+train_both_branch_array_combined[0][5]+train_both_branch_array_combined[0][6]+train_both_branch_array_combined[0][7]+train_both_branch_array_combined[0][8]+train_both_branch_array_combined[0][9]+train_both_branch_array_combined[0][10]+train_both_branch_array_combined[0][11]+train_both_branch_array_combined[0][0]
        votesss_0 = 12 - votesss_1
        if votesss_1 >= 7 and train_both_branch_result[0] == 0:
            train_both_branch_result[0] = 1
        if votesss_0 >= 7 and train_both_branch_result[0] == 1:
            train_both_branch_result[0] = 0;
        final_layer_predictions.append(train_both_branch_result)
        print("Final layer of both branch" , final_layer_predictions)



        # Final Model
        train_final_combined = np.array(final_layer_predictions).T


        final_result_after_all_model = Final_last_cnn_5_layer_model.predict(train_final_combined)

        final_result_binary = np.where(final_result_after_all_model >= 0.5, 1, 0)
        final_result_binary = final_result_binary.flatten()
        votes_1 = train_final_combined[0][0]+train_final_combined[0][1]+train_final_combined[0][2]
        votes_0 = 3 - votes_1
        if votes_1 >= 2 and final_result_binary[0] == 0:
            final_result_binary[0] = 1
        if votes_0 >= 2 and final_result_binary[0] == 1:
            final_result_binary[0] = 0
        print(final_result_binary)






        final = result_2_layer[150] + result_4_layer[150] + result_8_layer[150] + result_16_layer[150] + result_24_layer[150] + y_pred[0] + predictions_logistic[0]+lr_predictions_first[150]+predictions_second[150]+predictions_cnn_logistic[150]+y_predset[150]+naive_result[150]

        accuracy = (final / 12) * 100


        print("First Branch----------")
        print("Layer 2 result", result_2_layer[150])
        print("Layer 4 result", result_4_layer[150])
        print("Layer 8 result", result_8_layer[150])
        print("Layer 16 result", result_16_layer[150])
        print("Layer 24 result", result_24_layer[150])
        print("Xgboost result", y_pred[0])
        print("Logistic Regression result", predictions_logistic[0])
        print("First Branch Final Result", final_result)


        print("Second Branch---------")

        print("Random Forest + Logistic Regression", lr_predictions_first[150])
        print("Decision Tree + Logistic Regression", predictions_second[150])
        print("CNN + Logistic Regression", predictions_cnn_logistic[150])
        print("LightGBM + Catboost", y_predset[150])
        print("LightGBM + Naive_bayes", naive_result[150])


        print("Second Branch Final Result", train_final_result_binary)
        print(">>>>>>>>>>>>")
        print("Both Branch Final Result", train_both_branch_result)



        print("Final Output:", final_result_binary)




        return render_template('result.html', prediction=final_result_binary[0], accurate=accuracy, layer_2=result_2_layer[150], layer_4=result_4_layer[150], layer_8=result_8_layer[150], layer_16=result_16_layer[150], layer_24=result_24_layer[150], xgboost = y_pred[0], logistic = predictions_logistic[0], second_first = lr_predictions_first[150], second_second = predictions_second[150], cnn_logisticregr = predictions_cnn_logistic[150], lightgbm_catboo = y_predset[150], naive_bayes = naive_result[150],first_branch_final =final_result[0], second_branch_final = train_final_result_binary[0], both_branch_final = train_both_branch_result[0] )


if __name__ == '__main__':
    app.run(debug=True)
