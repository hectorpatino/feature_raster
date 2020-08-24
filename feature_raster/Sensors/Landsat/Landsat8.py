from feature_raster.Sensors import GeneralSensor
from feature_raster.project_enums import LandsatEnums


class Landsat8(GeneralSensor):
    def __init__(self, img_path):
        GeneralSensor.__init__(self, img_path)
        # Band names were found here
        # https://landsat.gsfc.nasa.gov/landsat-8/landsat-8-bands/
        # TODO all bands should be used here, maybe not pancromatic, at least for now
        self.dataframe.columns = [LandsatEnums.coastal.value, LandsatEnums.blue.value, LandsatEnums.green.value,
                                  LandsatEnums.red.value, LandsatEnums.nir.value, LandsatEnums.swir1.value,
                                  LandsatEnums.swir2.value, LandsatEnums.quality.value]
