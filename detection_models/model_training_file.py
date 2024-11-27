import os
import numpy as np
import pandas as pd
import librosa
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Bidirectional, Conv1D, BatchNormalization
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.regularizers import l2
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

CLIPS_DIR = "sep28k/clips/stuttering-clips/clips/"
file_path = "detection_model/training_data.csv"  

df = pd.read_csv(file_path)

X = []  
y = []  

count = 0
prev = 0
ck = 0
#MFCC Timestamps:
FIXED_TIMESTEPS = 100

for index, row in df.iterrows():
    audio_file_path = os.path.join(CLIPS_DIR, row['name'] + '.wav')
    
    try:
        audio, sample_rate = librosa.load(audio_file_path, sr=None)
        
        if audio.size == 0:
            print(f"Warning: {audio_file_path} is empty.")
            continue  # Skip if empty
        
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        if mfccs.shape[1] < FIXED_TIMESTEPS:
            mfccs = np.pad(mfccs, ((0, 0), (0, FIXED_TIMESTEPS - mfccs.shape[1])), mode='constant')
        else:
            mfccs = mfccs[:, :FIXED_TIMESTEPS]
        
        X.append(mfccs)
        y.append(row['result'])  

        count += 1
        ck = int(count / 2000)

        if ck != prev:
            prev = ck
            print('.')  
    
    except Exception as e:
        print(f"Error processing {audio_file_path}: {e}")


X = np.array(X)
y = np.array(y)

# Normalize MFCCs:
scaler = StandardScaler()
X = scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
X = X.reshape(X.shape[0], X.shape[1], X.shape[2])

# (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

class_weights = {0: 1., 1: 1.}  # 1:1 class weight (No imbalance present)

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.0001)

# Model:
model = Sequential()

model.add(Conv1D(16, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]), 
                 kernel_regularizer=l2(0.001)))  
model.add(BatchNormalization())  
model.add(Dropout(0.2))  

model.add(Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=l2(0.001))))  
model.add(BatchNormalization())  
model.add(Dropout(0.2))  

model.add(Bidirectional(LSTM(32, kernel_regularizer=l2(0.001))))
model.add(BatchNormalization())  
model.add(Dropout(0.2))  

model.add(Dense(64, activation="relu", kernel_regularizer=l2(0.001)))
model.add(BatchNormalization())  
model.add(Dense(32, activation="relu", kernel_regularizer=l2(0.001)))
model.add(Dense(1, activation="sigmoid"))

opt = Adam(learning_rate=0.0003)  
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, batch_size=64, epochs=5, validation_data=(X_test, y_test),
                    class_weight=class_weights, callbacks=[early_stopping, lr_scheduler])


model.save('detection_model_simplified.h5')

# Evaluation results:
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test Loss: {loss:.4f}")

y_pred = (model.predict(X_test) > 0.5).astype(int) 
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Stuttering', 'Stuttering'], yticklabels=['No Stuttering', 'Stuttering'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()