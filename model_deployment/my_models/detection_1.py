import os
import numpy as np
import librosa
from keras.models import load_model
from sklearn.preprocessing import StandardScaler

# Load the trained stuttering detection model
model = load_model('D:/IIT R MTech/Assignment/DSML/advance/my_models/detection_model_simplified.h5')

# Define fixed parameters used during training
FIXED_TIMESTEPS = 100

# Function to detect stuttering in an entire audio file
def detect_stuttering_in_audio(file_path):
    # Load the audio file
    audio, sample_rate = librosa.load(file_path, sr=None)
    
    # Extract MFCCs from the audio file
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)

    # Pad or truncate MFCCs to the fixed number of time steps
    if mfccs.shape[1] < FIXED_TIMESTEPS:
        # Pad with zeros if the audio is shorter
        mfccs = np.pad(mfccs, ((0, 0), (0, FIXED_TIMESTEPS - mfccs.shape[1])), mode='constant')
    else:
        # Truncate if it's longer
        mfccs = mfccs[:, :FIXED_TIMESTEPS]

    # Prepare the data for prediction (add batch dimension)
    X = np.expand_dims(mfccs, axis=0)  # Shape: (1, time_steps, features)

    # Normalize the MFCCs
    scaler = StandardScaler()
    X = scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)

    # Predict stuttering using the trained model
    prediction = model.predict(X)

    # Print the prediction result
    # print(f"Prediction: {prediction}")

    # Determine if stuttering is detected based on the prediction threshold
    # if prediction >= 0.5:
    #     return "Stuttering detected in the audio.", prediction
    # else:
    #     return "No stuttering detected in the audio.", prediction
    return prediction

# # Example usage: Path to the new .wav file
# file_path = "sep28k/clips/stuttering-clips/clips/FluencyBank_017_53.wav"
# detect_stuttering_in_audio(file_path)