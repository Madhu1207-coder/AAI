import os
import json
import pickle
import facebook

from flask import Flask, render_template, jsonify, request, redirect
from wtforms import Form, TextAreaField, validators
from sklearn.ensemble import RandomForestClassifier  # Fix for missing module

app = Flask(__name__)
app.secret_key = "s0mth1ng_s3cr3t"

##### Preparing the Classifier
cur_dir = os.path.dirname(__file__)
model_path = os.path.join(cur_dir, 'ml_model', 'classifier.pkl')

try:
    with open(model_path, 'rb') as model_file:
        rf = pickle.load(model_file)
        if not isinstance(rf, RandomForestClassifier):
            raise ValueError("Loaded model is not a RandomForestClassifier.")
except FileNotFoundError:
    print("Error: classifier.pkl not found. Make sure the model file exists.")
    rf = None
except Exception as e:
    print(f"Error loading classifier: {str(e)}")
    rf = None

# Replace with your actual Facebook API token
token = "EAAD4wuQJXMQBAOeLbGArkUVA5rlu3VndjMmyBqlc33vTbXAl9uZB4fQqZCj1ByAAjqvHq1vfLx8MZBZAOc3Ll9kaRZAUZATXcDZB2bR2PsZALxfZCfKzvZC1XhBI4gCM2JWI6C5v9uc24mJByxteLGZBUzU7G9HFFYn1VtqCo9BXDz6cgZDZD"

class EventForm(Form):
    eventid = TextAreaField('', [validators.DataRequired(), validators.length(min=5)])

def classify(eventid):
    try:
        graph = facebook.GraphAPI(access_token=token, version='2.10')
        field_events = 'attending_count, can_guests_invite, guest_list_enabled, maybe_count, noreply_count, interested_count'
        temp_event = graph.get_object(id=eventid, fields=field_events)
        order_list = ['attending_count', 'can_guests_invite', 'guest_list_enabled', 'maybe_count', 'noreply_count']
        event = [temp_event[x] for x in order_list]
        print(event)

        if rf:
            return int(rf.predict([event])[0]), int(temp_event['interested_count'])
        else:
            return None, None
    except Exception as e:
        print(f"Error in classification: {str(e)}")
        return None, None

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/event')
def event():
    try:
        form = EventForm(request.form)
        return render_template('eventform.html', form=form)
    except Exception as e:
        return f"Error: {str(e)}"

@app.route('/results', methods=['POST'])
def results():
    form = EventForm(request.form)
    if request.method == 'POST' and form.validate():
        eventid = request.form['eventid']
        prediction, truth = classify(eventid)

        if prediction is not None:
            return render_template('results.html', prediction=prediction, truth=truth)
        else:
            return "Error in classification. Please check model or API key."

    return "Invalid form input."

@app.route('/geoplot')
def barchat():
    try:
        return render_template('geoplot.html')
    except Exception as e:
        return f"Error: {str(e)}"

@app.route('/popcat')
def popcat():
    try:
        return render_template('popcat.html')
    except Exception as e:
        return f"Error: {str(e)}"

@app.route('/wordcloud')
def wordcloud():
    try:
        return render_template('wordcloud.html')
    except Exception as e:
        return f"Error: {str(e)}"

@app.route('/posneg')
def posneg():
    try:
        return render_template('posneg.html')
    except Exception as e:
        return f"Error: {str(e)}"

@app.route('/get/<file>')
def get(file):
    try:
        if file in ['word_freq']:
            with open(f'static/data/{file}.json') as f:
                data = json.load(f)
            return jsonify(data)
        else:
            return 'File not Found !'
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    
    app.run(debug=True,use_reloader = False)
