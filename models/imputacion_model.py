from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
import numpy as np

class ImputacionModel:
    def imputar_continuas_knn(self, df, variables_continuas):
        pipeline_continuas = make_pipeline(KNNImputer(n_neighbors=5, weights='uniform'))
        df[variables_continuas] = pipeline_continuas.fit_transform(df[variables_continuas])
        return df

    def imputar_categoricas_knn(self, df, variables_categoricas, ajuste_aleatorio=True):
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
    
    def imputar_continuas_regresion_lineal(self, df, variables_continuas):
        for target_variable in variables_continuas:
            if df[target_variable].isnull().sum() > 0:
                matrizX = df[variables_continuas].drop(columns=[target_variable]).fillna(0)
                SerieY = df[target_variable].dropna()
                X_train = matrizX.loc[SerieY.index]
                model = LinearRegression(fit_intercept=True, n_jobs=-1)
                model.fit(X_train, SerieY)

                df_missing = df.loc[df[target_variable].isnull()]
                y_pred = model.predict(df_missing[variables_continuas].drop(columns=[target_variable]).fillna(0))
                df.loc[df[target_variable].isnull(), target_variable] = y_pred
        return df
    
    def imputar_categoricas_moda(self, df, variables_categoricas):
        for var in variables_categoricas:
            moda = df[var].mode()[0]
            df[var].fillna(moda, inplace=True)
        return df

    def imputar_continuas_mediana(self, df, variables_continuas):
        for var in variables_continuas:
            mediana = df[var].median()
            df[var].fillna(mediana, inplace=True)
        return df
