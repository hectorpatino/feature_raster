from feature_raster.Sensors.GeneralSensor import GeneralSensor
from feature_raster.project_enums import LandsatEnums


class Landsat5(GeneralSensor):
    def __init__(self, img_path):
        GeneralSensor.__init__(self, img_path)
        # band names found here
        # https://www.usgs.gov/land-resources/nli/landsat/landsat-5
        self.dataframe.columns = [LandsatEnums.blue.value, LandsatEnums.red.value, LandsatEnums.green.value,
                                  LandsatEnums.nir.value, LandsatEnums.swir1.value, LandsatEnums.swir2.value,
                                  LandsatEnums.quality.value]

