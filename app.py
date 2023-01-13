# import Flask and create an instance of the Flask object
import flask
import pickle
import pandas as pd

# Use pickle to load in the trained model
with open(f'model/loan-model_log.pkl', 'rb') as f:
    model = pickle.load(f)

# Initialise the flask app
app = flask.Flask(__name__, template_folder='templates',
                  static_folder='static')


@app.route('/')
def home():
    return flask.render_template('main.html')
# add a function that returns content


@app.route("/submit", methods=['GET', 'POST'])
def form_submit():
    # if flask.request.method == 'POST':
    # Extract the input
    gender = flask.request.form['gender']
    marital_status = flask.request.form['marital_status']
    dependents = flask.request.form['dependents']
    level_of_education = flask.request.form['level_of_education']
    employment_status = flask.request.form['employment_status']
    t_income = flask.request.form['t_income']
    loan_amount = flask.request.form['loan_amount']
    term_of_loan = flask.request.form['term_of_loan']
    credit_history = flask.request.form['credit_history']
    property_area = flask.request.form['property_area']

    # Make inputs into DF for the model
    input_var = pd.DataFrame([[gender, marital_status, dependents, level_of_education, employment_status,
                               t_income, loan_amount, term_of_loan, credit_history, property_area]],
                             columns=['gender', 'marital_status', 'dependents', 'level_of_education', 'employment_status',
                                      't_income', 'loan_amount', 'term_of_loan', 'credit_history', 'property_area'],
                             index=['input'])

    # Get the model prediction
    prediction = model.predict(input_var)[0]

    # To render the form again, but add in the predictions and remind the user of previous input
    return flask.render_template('main.html', result=prediction)


if __name__ == '__main__':
    app.run()
