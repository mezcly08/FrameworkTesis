from sklearn.preprocessing import OrdinalEncoder

class CodificacionModel:
    def codificar_categoricas(self, df, variables_categoricas):
        encoder = OrdinalEncoder()
        df[variables_categoricas] = encoder.fit_transform(df[variables_categoricas])
        return df

    def identificar_variables(self, df):
        variables_categoricas = df.select_dtypes(include=['object', 'category']).columns.tolist()
        variables_continuas = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        return variables_categoricas, variables_continuas
