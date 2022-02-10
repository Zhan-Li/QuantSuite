from typing import List

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from quantsuite.config.auto_ml import auto_ml_config
from quantsuite.forcaster.transformers import Winsorizer


class Pipe:
    def __init__(self, model_str: str, num_cols: List[str]):
        self.model_str = model_str
        self.num_cols = num_cols

    def preprocess(self):
        num_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scale', RobustScaler()),
            ('cutoff', Winsorizer(1.5)),  # Cut off at 1.5 IQR)
        ])

        return ColumnTransformer(
            transformers=[
                ('num', num_transformer, self.num_cols)
            ])

    def build_pipline(self):
        if self.model_str == 'tf_regressor':
            return self.preprocess()
        return Pipeline(steps=[
            ('preprocessor', self.preprocess()),
            ('model', auto_ml_config[self.model_str]['model'])
        ])

    def get_params(self):
        model_params = auto_ml_config[self.model_str]['params']
        if self.model_str == 'tf_regressor':
            return {key: value for key, value in model_params.items()}
        return {'model__' + key: value for key, value in model_params.items()}
