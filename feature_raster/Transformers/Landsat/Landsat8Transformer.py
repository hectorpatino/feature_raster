from .LandsatCommonTransformer import LandsatGeneralTransformer

from feature_raster.Sensors.Landsat import Landsat8
from feature_raster.exceptions.some_exceptions import InvalidImage
from feature_raster.project_enums.LandsatEnums import LandsatEnums

from .indexes import *


class Landsat8Transformer(LandsatGeneralTransformer):
    def __init__(self):
        super().__init__()

    def fit(self, x, y=None):
        return self

    def transform(self, landsat8object):
        if not isinstance(landsat8object, Landsat8):
            raise InvalidImage("Image MUST BE a Landsat8 object")
        x = super().transform(landsat8object)
        x[LandsatEnums.bgi.value] = bgi(x[LandsatEnums.coastal.value], x[LandsatEnums.green.value])
        return x


