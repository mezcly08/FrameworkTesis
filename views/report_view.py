class ReportView:
    @staticmethod
    def mostrar_resumen(df):
        print("Resumen de datos:")
        print(df.describe())
        print("\nValores nulos por columna:")
        print(df.isnull().sum())
