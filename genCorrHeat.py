import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def save_heatmap(name):
    cmap = sns.color_palette("rocket", as_cmap=True)
    df = pd.read_csv("raw_data.csv")
    sns.heatmap(df.corr(),annot = True, cmap=cmap)
    plt.savefig(f"./corr_figs/{name}.png")
    plt.clf()
