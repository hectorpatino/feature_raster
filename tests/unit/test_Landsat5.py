import unittest
from feature_raster.Sensors.Landsat import Landsat5
from rasterio.crs import CRS
from tests.paths import small_2010_dataset


class Landsat5Test(unittest.TestCase):
    def setUp(self):
        self.landsat5 = Landsat5(small_2010_dataset)

    def test_meta_of_rasterio(self):
        expected_general = {
            'driver': 'GTiff', 'dtype': 'uint16', 'nodata': None,
            'width': 9, 'height': 8, 'count': 7,
            'crs': CRS.from_dict(init='epsg:32618'),
        }
        self.assertEqual(expected_general['driver'], self.landsat5.meta['driver'])
        self.assertEqual(expected_general['dtype'], self.landsat5.meta['dtype'])
        self.assertEqual(expected_general['nodata'], self.landsat5.meta['nodata'])
        self.assertEqual(expected_general['width'], self.landsat5.meta['width'])
        self.assertEqual(expected_general["height"], self.landsat5.meta["height"])
        self.assertEqual(expected_general["count"], self.landsat5.meta["count"])
        self.assertEqual(expected_general["crs"], self.landsat5.meta["crs"])
