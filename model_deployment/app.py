from flask import Flask, render_template, Response, jsonify, request, redirect
import time
from flask_cors import CORS
import os
from concurrent.futures import ThreadPoolExecutor
from wav2vec2_local import t1
from my_models import detection, t3

app = Flask(__name__)
CORS(app)  # This enables CORS for all routes

@app.route('/')
def index():
    return render_template('index.html')  # Render an HTML page with JavaScript

@app.route('/start_recording')
def start_audio_recording():
    result = t1.start_recording()
    return jsonify({'message': result})

@app.route('/stop_recording')
def stop_audio_recording():
    message, duration, transcription, detections = t1.stop_recording()
    duration = f'{duration:.2f}'
    return jsonify({'message': message,
                    'duration': duration,
                    'transcription': transcription,
                    'detection': detections,
    })

@app.route('/stream')
def stream():
    def generate():
        for i in range(10):
            data = t1.test_fun()
            time.sleep(1)
            yield f"data:{data} {i} \n\n"  # Send data as Server-Sent Events
    return Response(generate(), mimetype='text/event-stream')


# -------- U P L O A D --------

# Set up the upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'aac'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 *1024 # Max 16GB file size

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/upload', methods=['POST'])
def upload_audio():
    # Check if the post request has the file part
    if 'file' not in request.files:
        return 'File not present, please reupload it', 400
    
    file = request.files['file']

    # If no file is selected
    if file.filename == '':
        return 'File not uploaded', 400

    # If the file has a valid extension, process it and make a prediction
    if file and allowed_file(file.filename):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Call the functions sequentially
        result_1 = t1.process_uploaded_audio(filepath)
        result_2 = detection.detect_stuttering_in_audio(filepath)
        # result_3 = t3.final(filepath)  # Uncomment when needed
        
        # Gather results
        results = [result_1, result_2]  # Add result_3 if required
        
        return {'results': results}, 200

    return 'Invalid file format. Please upload an audio file (.mp3, .wav, .aac or .flac).', 400

if __name__ == '__main__':
    app.run(debug=True)