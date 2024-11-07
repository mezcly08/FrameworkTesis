class ReportView:
    @staticmethod
    def mostrar_resumen(df):
        print("Resumen de datos:")
        print(df.describe())
        print("\nValores nulos por columna:")
        print(df.isnull().sum())
    
    @staticmethod
    def mostrar_valores_nulos(df, df_imputado):
        print("Valores nulos antes de la imputación:")
        print(df.isnull().sum())
        print("\nValores nulos después de la imputación:")
        print(df_imputado.isnull().sum())