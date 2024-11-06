from models.imputacion_model import ImputacionModel

class ImputacionController:
    def __init__(self, df, variables_continuas, variables_categoricas):
        self.df = df
        self.imputacion_model = ImputacionModel()
        self.variables_continuas = variables_continuas
        self.variables_categoricas = variables_categoricas

    def aplicar_imputacion(self):
        if self.variables_continuas:
            self.df = self.imputacion_model.imputar_continuas(self.df, self.variables_continuas)
        if self.variables_categoricas:
            self.df = self.imputacion_model.imputar_categoricas(self.df, self.variables_categoricas)
        return self.df
