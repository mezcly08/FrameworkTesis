import pandas as pd

def load_data_from_url(url):
    """Carga el dataset desde una URL."""
    try:
        df = pd.read_csv(url, na_values='?')
        return df
    except Exception as e:
        print(f"Error al cargar el dataset: {e}")
        return None
