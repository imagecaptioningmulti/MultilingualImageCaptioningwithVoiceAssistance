from flask import Flask, request, jsonify, render_template, url_for
from flask_cors import CORS
import os
import io
from PIL import Image
import torch
from torchvision import transforms
from data_loader import get_loader
from model import DecoderRNN, EncoderCNN
from nlp_utils import clean_sentence

app = Flask(__name__)
# Configuration

# downloads_folder = os.path.expanduser('~/Downloads')
# cocoapi_dir = os.path.join(downloads_folder, 'COCOdataset')
cocoapi_dir = os.path.dirname(os.path.realpath(__file__))
print('>>>>>>>>>>>>',cocoapi_dir)

CORS(app)
# Defining a transform to pre-process the testing images.
transform_test = transforms.Compose(
    [
        transforms.Resize(256),  # smaller edge of image resized to 256
        transforms.CenterCrop(224),  # get 224x224 crop from center
        transforms.ToTensor(),  # convert the PIL Image to a tensor
        transforms.Normalize(
            (0.485, 0.456, 0.406),  # normalize image for pre-trained model
            (0.229, 0.224, 0.225),
        ),
    ]
)
# Creating the data loader.
data_loader = get_loader(transform=transform_test, mode="test", cocoapi_loc=cocoapi_dir)

device = torch.device("cpu")

# Specify the saved models to load.
encoder_file = "encoder-3.pkl"
decoder_file = "decoder-3.pkl"

# Select appropriate values for the Python variables below.
embed_size = 256
hidden_size = 512

# The size of the vocabulary.
vocab_size = len(data_loader.dataset.vocab)
print(vocab_size)
# Initialize the encoder and decoder, and set each to inference mode.
encoder = EncoderCNN(embed_size)
decoder = DecoderRNN(embed_size, hidden_size, vocab_size)
encoder.eval()
decoder.eval()

# Load the trained weights.
encoder.load_state_dict(torch.load(os.path.join("./captioning_models", encoder_file), map_location=device))
decoder.load_state_dict(torch.load(os.path.join("./captioning_models", decoder_file), map_location=device))

# Move models to CPU
encoder.to(device)
decoder.to(device)

def predict_caption(image):
    if image is None:
        return "Please select an image"

    image = transform_test(image).unsqueeze(0)
    with torch.no_grad():
        # Moving image Pytorch Tensor to CPU
        image = image.to(device)

        # Obtaining the embedded image features.
        features = encoder(image).unsqueeze(1)

        # Passing the embedded image features through the model to get a predicted caption.
        output = decoder.sample(features)

    sentence = clean_sentence(output, data_loader.dataset.vocab.idx2word)

    return sentence
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predictweb', methods=['POST'])
def predictweb():
    '''
    For rendering results on HTML GUI
    '''
    # name = request.form['name']
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    file = request.files['image']
    print(file.filename)
    # filename = os.path.join('./uploads', file.filename)
    # file.save(filename)
    if file and file.filename != '':
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
        caption = predict_caption(img)
        image_url = url_for('static', filename='uploads/' + file.filename)
        print(image_url)

    # Save the uploaded image to a static/uploads directory for display
        file.save(os.path.join('static/uploads', file.filename))
        return render_template('index.html', prediction_text=':{}'.format(caption),image_url=image_url)
    
    else:
        return jsonify({"error": "Invalid image"}), 400
    # return render_template('index.html', prediction_text='Hello, {}'.format(name))

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
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    file = request.files['image']
    print(file)
    # filename = os.path.join('./uploads', file.filename)
    # file.save(filename)
    if file and file.filename != '':
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
        caption = predict_caption(img)
        return jsonify({"caption": caption})
    
    else:
        return jsonify({"error": "Invalid image"}), 400
if __name__ == "__main__":
    app.run(debug=True)
