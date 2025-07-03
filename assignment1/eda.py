import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv('assignment1/athletes.csv')

# Calculate total lift and save the updated DataFrame
df['total_lift'] = df['snatch'] + df['deadlift'] + df['backsq'] + df['candj']
df.to_csv('athletes.csv')

# Basic info
print('Shape:', df.shape)
print('Columns:', df.columns.tolist())
print('Data types:\n', df.dtypes)

# First few rows
print('\nFirst 5 rows:')
print(df.head())

# Summary statistics
print('\nSummary statistics:')
print(df.describe(include='all'))

# Missing values
print('\nMissing values:')
print(df.isnull().sum())

# Value counts for categorical columns
cat_cols = df.select_dtypes(include=['object', 'category']).columns
for col in cat_cols:
    print(f'\nValue counts for {col}:')
    print(df[col].value_counts())

# Plot distributions for numerical columns
num_cols = df.select_dtypes(include=['number']).columns
for col in num_cols:
    plt.figure(figsize=(6, 4))
    sns.histplot(df[col].dropna(), kde=True)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

# Bar plots for categorical columns
for col in cat_cols:
    plt.figure(figsize=(8, 4))
    df[col].value_counts().plot(kind='bar')
    plt.title(f'Value Counts of {col}')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()
