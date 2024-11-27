import os
import pandas as pd
import numpy as np
import librosa
import soundfile as sf
import random

CLIPS_DIR = "sep28k/clips/stuttering-clips/clips/"  

if not os.path.exists(CLIPS_DIR):
    raise FileNotFoundError(f"The directory {CLIPS_DIR} does not exist.")

file_path = 'data/sep28k/det_labels.csv'  
df = pd.read_csv(file_path)

result_counts = df['result'].value_counts()

print("Counts of result values before augmentation:")
print(result_counts)

samples_to_augment = df[df['result'] == 0].sample(n=1000, random_state=42)

augmented_samples = []
for idx, row in samples_to_augment.iterrows():

    original_file_path = os.path.join(CLIPS_DIR, row['name'] + '.wav')  
    
    try:
        audio, sr = librosa.load(original_file_path, sr=None)

        noise_factor = 0.005
        noisy_audio = audio + noise_factor * np.random.randn(len(audio))

        pitch_shift = random.randint(-2, 2)  
        pitched_audio = librosa.effects.pitch_shift(noisy_audio, n_steps=pitch_shift, sr=sr)

        speed_change = random.uniform(0.9, 1.1)  
        speed_adjusted_audio = librosa.effects.time_stretch(pitched_audio, rate=speed_change)

        new_file_name = f"new_{len(augmented_samples)}.wav"
        new_file_path = os.path.join(CLIPS_DIR, new_file_name)

        if os.path.exists(new_file_path):
            print(f"File already exists: {new_file_path}. Skipping...")
            continue
        
        sf.write(new_file_path, speed_adjusted_audio, sr)
        print(f"Saved augmented file: {new_file_path}")

        augmented_samples.append({'name': new_file_name, 'result': 0})
    
    except FileNotFoundError:
        print(f"File not found: {original_file_path}. Dropping this row from the DataFrame.")
        df = df.drop(idx)  
    except Exception as e:
        print(f"An error occurred while processing {original_file_path}: {e}")

augmented_df = pd.DataFrame(augmented_samples)
df = pd.concat([df, augmented_df], ignore_index=True)


num_to_delete = 2500
if df[df['result'] == 1].shape[0] >= num_to_delete:
    
    indices_to_delete = df[df['result'] == 1].sample(n=num_to_delete, random_state=42).index
    
    df = df.drop(indices_to_delete)
else:
    print(f"Not enough rows to delete. Available: {df[df['result'] == 1].shape[0]}")

df = df.sort_values(by='name')

result_counts = df['result'].value_counts()
print("Counts of result values after augmentation and deletion:")
print(result_counts)

df.to_csv('sep28k/detect_balanced.csv', index=False)
