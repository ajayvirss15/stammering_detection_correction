import pandas as pd

file_path1 = "labelv0.csv"
file_path2 = "labelv1.csv"

df_fluency = pd.read_csv(file_path1)
df_sep28 = pd.read_csv(file_path2)

df_combined = pd.concat([df_fluency, df_sep28], ignore_index=True)

df_combined.to_csv("combined_labels.csv", index=False)

print(df_combined.head())