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

    # Imputación
    imputacion_controller = ImputacionController(df, variables_continuas, variables_categoricas)
    df_imputado = imputacion_controller.aplicar_imputacion()

    # Mostrar resumen después de la imputación
    print("\nEstadísticas descriptivas después de la imputación:")
    ReportView.mostrar_resumen(df_imputado)

if __name__ == "__main__":
    main()
