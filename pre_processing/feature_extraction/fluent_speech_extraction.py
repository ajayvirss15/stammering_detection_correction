import pandas as pd
import os

file_path1="test/shuvam_tr/d1/fluencybank_labels.csv"
file_path2="labelv1.csv"
clip_path="sep28k/clips/stuttering-clips/clips"

df_fluency = pd.read_csv(file_path1)
df_sep28 = pd.read_csv(file_path2)

df_fluency['name'] = df_fluency['Show'] + '_' + df_fluency['EpId'].apply(lambda x: str(x).zfill(3)) + '_' + df_fluency['ClipId'].astype(str)
df_fluency['result'] = 0

# condition for fluent speech:
fluent_condition = (
    (df_fluency['Prolongation'] == 0) & 
    (
        (df_fluency['Block'] <= 1) & 
        (
            (df_fluency['SoundRep'] <= 1) | 
            (df_fluency['WordRep'] <= 1)
        )          
    )         
)

df_fluency.loc[fluent_condition, 'result'] = 0

df_result_0 = df_fluency[df_fluency['result'] == 0]

df_result_0 = df_result_0[df_result_0['name'].apply(
    lambda x: os.path.exists(os.path.join(clip_path, f"{x}.wav")) and os.stat(os.path.join(clip_path, f"{x}.wav")).st_size != 44
)]

df_result_0 = df_result_0[~df_result_0[['Unsure', 'PoorAudioQuality', 'NoSpeech', 'Music']].eq(1).any(axis=1)]
df_result_0 = df_result_0[['name', 'result']]

df_result_0.to_csv('labelv0.csv', index=False)
print(df_result_0.head())