from sklearn.base import BaseEstimator, TransformerMixin
from feature_raster.Sensors.Landsat import Landsat5
from feature_raster.exceptions import InvalidImage
from feature_raster.project_enums import LandsatEnums
from .LandsatCommonTransformer import LandsatGeneralTransformer
from .indexes import *


class Landsat5Transformer(LandsatGeneralTransformer):
    def __init__(self):
        super().__init__()

    def fit(self, x, y=None):
        return self

    def transform(self, landsat5object):
        if not isinstance(landsat5object, Landsat5):
            raise InvalidImage("Image must be a landsat5 object")
        return super().transform(landsat5object)
