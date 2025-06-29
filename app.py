from flask import Flask, request, send_file, send_from_directory
from flask_cors import CORS
import os
from utils import load_model, process_image

# Initialize Flask
app = Flask(__name__, static_folder="frontend", static_url_path="")
CORS(app)

# Limit upload size to prevent system crashes
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB max

# Folder paths
UPLOAD_FOLDER = r'C:\Users\Anunanda\id_api\Uploads'
OUTPUT_FOLDER = r'C:\Users\Anunanda\id_api\Outputs'
MODEL_PATH = r'C:\Users\Anunanda\id_api\Backend\u2net\u2net.pth'

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load the U2NET model
model = load_model(MODEL_PATH)

# Route for main page
@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

# Route to process image
@app.route('/process', methods=['POST'])
def process():
    try:
        if 'image' not in request.files:
            return "No image uploaded", 400

        file = request.files['image']
        filename = file.filename
        input_path = os.path.join(UPLOAD_FOLDER, filename)
        output_path = os.path.join(OUTPUT_FOLDER, filename)

        file.save(input_path)
        process_image(model, input_path, output_path)

        return send_file(output_path, mimetype='image/png')

    except Exception as e:
        print(f"[ERROR] {e}")
        return f"Processing failed: {str(e)}", 500

if __name__ == '__main__':
    app.run(debug=True)
