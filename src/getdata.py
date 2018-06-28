#Load in data and change to CSV readable format
import pandas as pd
import matplotlib.pyplot as plt

def get_df(i):
    df = pd.read_csv(i)
    return df
