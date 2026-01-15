import mlflow.pyfunc
import pandas as pd

class PyCaretModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, pycaret_model):
        self.model = pycaret_model

    def predict(self, context, model_input):
        model_input_copy = model_input.copy()
        if 'survived' in model_input_copy.columns:
            model_input_copy = model_input_copy.drop(columns=['survived'])
        return self.model.predict(model_input_copy)