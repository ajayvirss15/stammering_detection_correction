import os
import pandas as pd


CLIPS_DIR = "data/sep28k/clips/stuttering-clips/clips/"

file_path = 'data/sep28k/SEP-28k_labels.csv'  
df = pd.read_csv(file_path)

df['name'] = df['Show'] + '_' + df['EpId'].astype(str) + '_' + df['ClipId'].astype(str)

df = df.sort_values(by='name')

ignore_list = []

for filename in os.listdir(CLIPS_DIR):
    file_path = os.path.join(CLIPS_DIR, filename)

    if 'FluencyBank' in filename:
        filename_without_ext = filename[:-4]
        ignore_list.append(filename)
        continue

    if os.stat(file_path).st_size == 44:
        ignore_list.append(filename)
        filename_without_ext = filename[:-4]
        df = df[df['name'] != filename_without_ext]
        continue

    filename_without_ext = filename[:-4]
    row = df[df['name'] == filename_without_ext]

    if not row.empty:
        if (row['Unsure'].values[0] == 1 or 
            row['PoorAudioQuality'].values[0] == 1 or 
            row['NoSpeech'].values[0] == 1 or 
            row['Music'].values[0] == 1):
            ignore_list.append(filename)
            df = df[df['name'] != filename_without_ext]


df.to_csv('data/sep28k/SEP-28k_labels_filtered.csv', index=False)

print(len(ignore_list))
print(df.head())
print(ignore_list[0])