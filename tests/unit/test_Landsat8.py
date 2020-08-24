import unittest
from feature_raster.Sensors.Landsat import Landsat8
from rasterio.crs import CRS
from tests.paths import small_2018_dataset


class Landsat8Test(unittest.TestCase):
    def setUp(self):
        self.landsat8 = Landsat8(small_2018_dataset)

    def test_meta_of_rasterio(self):
        expected_general = {'driver': 'GTiff', 'dtype': 'uint16', 'nodata': None,
                            'width': 10, 'height': 9, 'count': 8,
                            'crs': CRS.from_dict(init='epsg:32618'), }
        self.assertEqual(expected_general['driver'], self.landsat8.meta['driver'])
        self.assertEqual(expected_general['dtype'], self.landsat8.meta['dtype'])
        self.assertEqual(expected_general['nodata'], self.landsat8.meta['nodata'])
        self.assertEqual(expected_general['width'], self.landsat8.meta['width'])
        self.assertEqual(expected_general["height"], self.landsat8.meta["height"])
        self.assertEqual(expected_general["count"], self.landsat8.meta["count"])
        self.assertEqual(expected_general["crs"], self.landsat8.meta["crs"])

    def test_correct_parsing_from_raster_to_pandas(self):
        # TODO dont forget to create this test
        pass



class Landsat8InitTest(unittest.TestCase):

    def test_img_path_is_a_valid_file(self):
        img_path = small_2018_dataset
        self.landsat8 = Landsat8(img_path=img_path)
        self.assertEqual(img_path, self.landsat8.img_path)

    def test_img_path_is_not_a_valid_file(self):
        # TODO make a count of bands right now i only support 8 including quality
        with self.assertRaises(ValueError):
            landsat8 = Landsat8(img_path=r"data/20218.tif")









