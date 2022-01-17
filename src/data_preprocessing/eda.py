import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def heatmap_corr_plot(df: pd.DataFrame):
    correlation_matrix = df.corr().round(2)
    plt.figure(figsize=(16, 9))
    sns.heatmap(data=correlation_matrix, annot=True)
    plt.savefig("plots/correlation.png")
