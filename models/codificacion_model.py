from sklearn.preprocessing import OrdinalEncoder
import pandas as pd

class CodificacionModel:
    def codificar_categoricas(self, df, variables_categoricas):
        encoder = OrdinalEncoder()
        df[variables_categoricas] = encoder.fit_transform(df[variables_categoricas])
        return df

    def identificar_variables(self, df):
        variables_categoricas = df.select_dtypes(include=['object', 'category']).columns.tolist()
        variables_continuas = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        return variables_categoricas, variables_continuas
    
    def revisar_codificacion_categoricas(self, df, variables_categoricas):
        for var in variables_categoricas:
            if pd.api.types.is_numeric_dtype(df[var]):
                continue  # Si ya está codificada, no se debe hacer nada
            else:
                encoder = OrdinalEncoder()
                df[var] = encoder.fit_transform(df[[var]])
        return df
