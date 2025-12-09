import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load data
df = pd.read_csv("medical_examination.csv")

# 1. Add 'overweight' column
# BMI = weight (kg) / height (m)^2
df['overweight'] = (df['weight'] / ((df['height'] / 100) ** 2)).apply(lambda x: 1 if x > 25 else 0)

# 2. Normalize data
# Cholesterol and gluc: 0 = good, 1 = bad
df['cholesterol'] = df['cholesterol'].apply(lambda x: 0 if x == 1 else 1)
df['gluc'] = df['gluc'].apply(lambda x: 0 if x == 1 else 1)

# 3. Draw Categorical Plot
def draw_cat_plot():
    # Prepare DataFrame for categorical plot
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol','gluc','smoke','alco','active','overweight'])
    
    # Draw the categorical plot
    fig = sns.catplot(
        data=df_cat,
        kind='count',
        x='variable',
        hue='value',
        col='cardio'
    ).fig

    # Save the figure
    fig.savefig('catplot.png')
    return fig

# 4. Draw Heat Map
def draw_heat_map():
    # Clean the data
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # Compute the correlation matrix
    corr = df_heat.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 12))

    # Draw the heatmap
    sns.heatmap(corr, mask=mask, annot=True, fmt=".1f", center=0, square=True, linewidths=1, cbar_kws={"shrink": .5})

    # Save the figure
    fig.savefig('heatmap.png')
    return fig

# Run functions if executed directly
if _name_ == "_main_":
    draw_cat_plot()
    draw_heat_map()
