import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Sample data
df = pd.DataFrame({
    "Base_Price": [100, 150, 200, 120],
    "Discount_Percent": [0.2, 0.1, 0.25, 0.15],
})

# Engineer derived columns



# Add Discounted_Price (redundant)
df["Discounted_Price"] = df["Base_Price"] * (1 - df["Discount_Percent"])

df["Profit"] = df["Base_Price"] - df["Discounted_Price"] 

print("Dataframe:")
print(df)

# Correlation matrix
corr_matrix = df.corr()

print("\nCorrelation matrix:")
print(corr_matrix)

# Plot heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix Showing Multicollinearity")
plt.show()