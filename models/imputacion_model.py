from sklearn.impute import KNNImputer
import numpy as np

class ImputacionModel:
    def imputar_continuas(self, df, variables_continuas):
        pipeline_continuas = KNNImputer(n_neighbors=5, weights='uniform')
        df[variables_continuas] = pipeline_continuas.fit_transform(df[variables_continuas])
        return df

    def imputar_categoricas(self, df, variables_categoricas, ajuste_aleatorio=True):
        try:
            knn_imputer = KNNImputer(n_neighbors=5, weights='uniform')
            df[variables_categoricas] = knn_imputer.fit_transform(df[variables_categoricas])
        except Exception:
            for var in variables_categoricas:
                moda = df[var].mode()
                if not moda.empty:
                    df[var].fillna(moda[0], inplace=True)
                    if ajuste_aleatorio:
                        n_missing = df[var].isnull().sum()
                        if n_missing > 0:
                            valores_aleatorios = np.random.choice(df[var].dropna(), size=n_missing)
                            df.loc[df[var].isnull(), var] = valores_aleatorios
        return df
