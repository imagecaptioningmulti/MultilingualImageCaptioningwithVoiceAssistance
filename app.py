from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
app = Flask(__name__)


CORS(app)
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    name = request.form['name']
    return render_template('index.html', prediction_text='Hello, {}'.format(name))

@app.route('/get_example', methods=['GET'])
def get_example():
    '''
    Example GET method
    '''
    return jsonify(message="This is an example GET response")

@app.route('/post_example', methods=['POST'])
def post_example():
    '''
    Example POST method
    '''
    data = request.json
    return jsonify(message="This is an example POST response", received_data=data)

if __name__ == "__main__":
    app.run(debug=True)
