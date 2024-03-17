import pandas as pd
import seaborn as sns
data = pd.read_csv('healthcare.csv')
df.info()
pivot_table = df.pivot_table(index='Blood Group Type', columns='Gender', values='Medical Condition', aggfunc='count')
sns.heatmap(pivot_table,annot=True,fmt='g')
