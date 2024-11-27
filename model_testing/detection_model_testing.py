import os
import numpy as np
import librosa
from keras.models import load_model
from sklearn.preprocessing import StandardScaler

# Load the trained stuttering detection model
model = load_model('detection_model_simplified.h5')

# Define fixed parameters used during training
FIXED_TIMESTEPS = 100
CHUNK_DURATION = 3  # Duration of each chunk in seconds
SAMPLE_RATE = 22050  # Audio sample rate (adjust if different)

# Function to detect stuttering in chunks of an audio file
def detect_stuttering_in_audio(file_path):
    # Load the audio file
    audio, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)

    # Calculate the number of samples per chunk
    chunk_samples = CHUNK_DURATION * SAMPLE_RATE

    # Text-based report
    report = []

    # Loop through the audio file in 3-second chunks
    for start_sample in range(0, len(audio), chunk_samples):
        # Get the end sample for the current chunk
        end_sample = min(start_sample + chunk_samples, len(audio))
        
        # Extract the chunk of audio
        chunk_audio = audio[start_sample:end_sample]
        
        # Extract MFCC features for the chunk
        mfccs = librosa.feature.mfcc(y=chunk_audio, sr=sample_rate, n_mfcc=40)

        # Pad or truncate MFCCs to the fixed number of time steps
        if mfccs.shape[1] < FIXED_TIMESTEPS:
            mfccs = np.pad(mfccs, ((0, 0), (0, FIXED_TIMESTEPS - mfccs.shape[1])), mode='constant')
        else:
            mfccs = mfccs[:, :FIXED_TIMESTEPS]

        # Prepare the data for prediction (add batch dimension)
        X = np.expand_dims(mfccs, axis=0)  # Shape: (1, time_steps, features)

        # Normalize the MFCCs
        scaler = StandardScaler()
        X = scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)

        # Predict stuttering for the current chunk
        prediction = model.predict(X)

        # Determine if stuttering is detected based on the prediction threshold
        if prediction >= 0.5:
            result = "stuttering detected"
        else:
            result = "no stuttering detected"

        # Calculate the start and end time of the current chunk in seconds
        start_time = start_sample / sample_rate
        end_time = end_sample / sample_rate

        # Add result to the report
        report.append(f"From {round(start_time, 2)}s to {round(end_time, 2)}s: {result}.")

    # Combine the report into a paragraph
    final_report = " ".join(report)

    # Print the report
    print("\nStuttering Detection Report:")
    print(final_report)

# Example usage: Path to the new .wav file
file_path = "pre_processing/test/M_0104_15y4m_1.wav"
detect_stuttering_in_audio(file_path)