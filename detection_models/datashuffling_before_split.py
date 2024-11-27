import pandas as pd

input_csv = 'combined_labels.csv'  
output_csv = 'shuffled_output_file.csv'  

df = pd.read_csv(input_csv)
df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

df_shuffled.to_csv(output_csv, index=False)
