from models.codificacion_model import CodificacionModel

class CodificacionController:
    def __init__(self, df):
        self.df = df
        self.codificacion_model = CodificacionModel()

    def aplicar_codificacion(self):
        variables_categoricas, variables_continuas = self.codificacion_model.identificar_variables(self.df)
        self.df = self.codificacion_model.codificar_categoricas(self.df, variables_categoricas)
        return self.df, variables_categoricas, variables_continuas
