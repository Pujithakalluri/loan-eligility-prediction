from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('loan_approval_model.pkl')

# Define a route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define a route to handle loan approval predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from the form
    input_data = [request.form['gender'], request.form['married'], request.form['dependents'],
                  request.form['education'], request.form['self_employed'], request.form['applicant_income'],
                  request.form['coapplicant_income'], request.form['loan_amount'], request.form['loan_amount_term'],
                  request.form['credit_history'], request.form['property_area']]

    # Convert input data to the same format as the training data
    input_data = [int(x) if i in [2, 5, 6, 7, 8, 9] else x for i, x in enumerate(input_data)]

    # Make a prediction
    prediction = model.predict([input_data])

    # Map the prediction to the corresponding label
    result = "Approved" if prediction[0] == 1 else "Not Approved"

    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
