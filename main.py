from utils.data_loader import load_data_from_url
from controllers.codificacion_controller import CodificacionController
from controllers.imputacion_controller import ImputacionController
from views.report_view import ReportView

def main():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00383/risk_factors_cervical_cancer.csv"
    df = load_data_from_url(url)
    if df is None:
        return

    # Mostrar resumen inicial
    print("Estadísticas descriptivas antes de la imputación:")
    ReportView.mostrar_resumen(df)

    # Codificación
    codificacion_controller = CodificacionController(df)
    df, variables_categoricas, variables_continuas = codificacion_controller.aplicar_codificacion()

    # Mostrar valores nulos antes de la imputación
    print("\nValores nulos antes de la imputación:")
    print(df.isnull().sum())

#---BASE
    # Imputación
    imputacion_controller = ImputacionController(df, variables_continuas, variables_categoricas)
    df_imputado = imputacion_controller.aplicar_imputacion()

    # Mostrar resumen después de la imputación
    print("\nEstadísticas descriptivas después de la imputación:")
    ReportView.mostrar_resumen(df_imputado)
#---

#KNN
    # Imputación con KNN
    imputacion_controller_knn = ImputacionController(df, variables_continuas, variables_categoricas)
    df_imputado_knn = imputacion_controller_knn.aplicar_imputacion()

    # Mostrar valores nulos después de la imputación con KNN
    print("\nValores nulos después de la imputación con KNN:")
    ReportView.mostrar_valores_nulos(df, df_imputado_knn)

    # Mostrar resumen después de la imputación con KNN
    print("\nEstadísticas descriptivas después de la imputación con KNN:")
    ReportView.mostrar_resumen(df_imputado_knn)
#kNN_FIN
#MODA_MEDIANA
    # Imputación con moda y mediana
    imputacion_controller_regresion = ImputacionController(df, variables_continuas, variables_categoricas)
    df_imputado_moda_mediana = imputacion_controller_regresion.aplicar_imputacion_moda_mediana()

    # Mostrar resumen después de la imputación con moda y mediana
    print("\nEstadísticas descriptivas después de la imputación con moda y mediana:")
    ReportView.mostrar_resumen(df_imputado_moda_mediana)

    # Mostrar valores nulos después de la imputación con moda y mediana
    print("\nValores nulos después de la imputación con moda y mediana:")
    ReportView.mostrar_valores_nulos(df, df_imputado_moda_mediana)
#MODA_MEDIANA_FIN
#REGRESION_LINEAL
    # Imputación con Regresión Lineal
    imputacion_controller_regresion = ImputacionController(df, variables_continuas, variables_categoricas)
    df_imputado_regresion = imputacion_controller_regresion.aplicar_imputacion_regresion_lineal()

    # Mostrar resumen después de la imputación con Regresión Lineal
    print("\nEstadísticas descriptivas después de la imputación con Regresión Lineal:")
    ReportView.mostrar_resumen(df_imputado_regresion)

    # Mostrar valores nulos después de la imputación con Regresión Lineal
    print("\nValores nulos después de la imputación con Regresión Lineal:")
    ReportView.mostrar_valores_nulos(df, df_imputado_regresion)

#REGRESION_LINEAL_FIN



if __name__ == "__main__":
    main()
