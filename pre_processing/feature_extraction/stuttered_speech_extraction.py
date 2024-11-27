import pandas as pd
import os

file_path = 'SEP-28k_labels_filtered.csv'  
df = pd.read_csv(file_path)

df['result'] = 0

stuttering_condition = (
    (df['Prolongation'] >= 1) & 
    (
        (df['Block'] == 1) |         
        (df['SoundRep'] >= 1) |      
        (df['WordRep'] >= 1)
    )         
)

df.loc[stuttering_condition, 'result'] = 1

df_result_1 = df[df['result'] == 1]
df_result_1 = df_result_1[['name', 'result']]

df_result_1.to_csv('labelv1.csv', index=False)

print(df_result_1.head())