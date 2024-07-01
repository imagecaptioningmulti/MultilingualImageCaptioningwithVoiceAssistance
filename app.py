from flask import Flask, request, jsonify, render_template, url_for
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import pickle
import io
import os
app = Flask(__name__)
# Configuration




CORS(app)


# Load the encoder model
encoder_model_path = './FinalModels/encoder_model'
loaded_encoder = tf.keras.models.load_model(encoder_model_path)

# Load the decoder model
decoder_model_path = './FinalModels/decoder_model'
loaded_decoder = tf.keras.models.load_model(decoder_model_path)

# Load the tokenizer
with open('./FinalModels/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

max_length = 35
attention_features_shape = 64

def load_image(image_path):    
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)    
    return img, image_path

# Image feature extraction model
image_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
new_input = image_model.input 
hidden_layer = image_model.layers[-1].output 
image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

def evaluate_V1(image, loaded_encoder, loaded_decoder, max_length, tokenizer, attention_features_shape):
    attention_plot = np.zeros((max_length, attention_features_shape))

    # Manually initialize the hidden state for the Decoder
    hidden = tf.zeros((1, loaded_decoder.layers[2].units))  # Adjust units based on your GRU layer

    temp_input = tf.expand_dims(load_image(image)[0], 0)
    img_tensor_val = image_features_extract_model(temp_input)
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

    features = loaded_encoder(img_tensor_val)

    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    result = []

    for i in range(max_length):
        predictions, hidden, attention_weights = loaded_decoder(dec_input, features, hidden)

        attention_plot[i] = tf.reshape(attention_weights, (-1,)).numpy()

        predicted_id = tf.argmax(predictions[0]).numpy()
        result.append(tokenizer.index_word[predicted_id])

        if tokenizer.index_word[predicted_id] == '<end>':
            return result, attention_plot, predictions

        dec_input = tf.expand_dims([predicted_id], 0)

    attention_plot = attention_plot[:len(result), :]
    return result, attention_plot, predictions



      
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predictweb', methods=['POST'])
def predictweb():
    '''
    For rendering results on HTML GUI
    '''
    file = request.files['image']
    filename = os.path.join('./uploads', file.filename)
    file.save(filename)
    result, attention_plot, predictions = evaluate_V1(filename, loaded_encoder, loaded_decoder, max_length=20, tokenizer=tokenizer, attention_features_shape=64)
    pred_caption = ' '.join(result)   
    
    return render_template('index.html', prediction_text=':{}'.format(pred_caption))
   

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
@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    filename = os.path.join('./uploads', file.filename)
    file.save(filename)
    result, attention_plot, predictions = evaluate_V1(filename, loaded_encoder, loaded_decoder, max_length=20, tokenizer=tokenizer, attention_features_shape=64)
    pred_caption = ' '.join(result)   
    return jsonify({'caption': ' '.join(result)})
if __name__ == "__main__":
    app.run(debug=True)
