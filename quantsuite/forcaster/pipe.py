from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.compose import TransformedTargetRegressor
from quantsuite.forcaster.transformers import Winsorizer, Identity
from quantsuite.config.auto_ml import auto_ml_config
from typing import List


class Pipe:
    def __init__(self, model_str: str, num_cols: List[str]):
        self.model_str = model_str
        self.num_cols = num_cols
        self.pipe = None
        self.params = None

    def build_pipline(self):
        num_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scale', RobustScaler()),
            ('cutoff', Winsorizer(1.5)),  # Cut off at 1.5 IQR)
        ])

        preprocessor_x = ColumnTransformer(
            transformers=[
                ('num', num_transformer, self.num_cols)
            ])
        pipeline_x = Pipeline(steps=[
            ('preprocessor', preprocessor_x),
            ('model', auto_ml_config[self.model_str]['model'])
        ])

        preprocessor_y = Pipeline(steps=[
            ('copy', Identity())
        ])
        self.pipe = TransformedTargetRegressor(regressor=pipeline_x, transformer=preprocessor_y)

    def get_params(self):
        model_params = auto_ml_config[self.model_str]['params']
        self.params = {'regressor__model__' + key: value for key, value in model_params.items()}




