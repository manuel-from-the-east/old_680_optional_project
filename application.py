# save this as app.py
from flask import Flask, escape, request, render_template
import pandas as pd
import joblib

application = Flask(__name__)

# ---------------------------------------------------------------------------------------------------------------------#
# ---------------------------------------------- ML Model Code --------------------------------------------------------#
# ---------------------------------------------------------------------------------------------------------------------#

@application.route('/')
@application.route('/about')
def about():

    return render_template("about.html")

@application.route('/cafeOccupancyPredictor')
def cafeOccupancyPredictor():

    return render_template("cafeOccupancyPredictor.html")

def preprocessDataAndPredict(curr_day, curr_time, curr_occ, occ_24, occ_week, occ_month):
    # keep all inputs in array
    data = [curr_day, curr_time, curr_occ, occ_24, occ_week, occ_month]

    # Create Data Frame
    data = pd.DataFrame({'day': [curr_day], 'curr_time': [curr_time],
         'curr_occ': [curr_occ], 'occ_24': [occ_24], 'occ_week': [occ_week],
         'occ_month': [occ_month]})

    # open file
    file = open("finalModel.pkl", "rb")

    # load trained model
    trained_model = joblib.load(file)

    # predict
    prediction = trained_model.predict(data)

    return round(prediction[0], 0)

@application.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == "POST":
        # get form data
        curr_day = request.form.get('curr_day')
        curr_time = request.form.get('curr_time')
        curr_occ = request.form.get('curr_occ')
        occ_24 = request.form.get('occ_24')
        occ_week = request.form.get('occ_week')
        occ_month = request.form.get('occ_month')

        # call preprocessDataAndPredict and pass inputs
        try:
            prediction = preprocessDataAndPredict(curr_day, curr_time, curr_occ, occ_24, occ_week, occ_month)
            # pass prediction to template
            return render_template('predict.html', prediction=prediction)

        except ValueError:
            return "Please Enter valid values"

        pass
    pass


# Run on Correct Port
if __name__ == '__main__':
    # Setting debug to True enables debug output. This line should be
    # removed before deploying a production app.
    application.debug = True
    application.run(host="localhost", port=5000, debug=True)