import os
import pandas as pd


def clean_data_estim(file_path):
    """
    Nettoie le dataset en supprimant les lignes avec des valeurs manquantes.

    Parameters:
    file_path (str): le chemin du csv à nettoyer.

    Returns:
    pd.DataFrame: Le dataset propre.
    """
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)

    # Remove rows with any NaN values
    df_cleaned = df.dropna()

    # Reset the index of the cleaned DataFrame
    df_cleaned.reset_index(drop=True, inplace=True)

    return df_cleaned