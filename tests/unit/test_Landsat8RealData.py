import unittest

from tests import envelope_2018, coberture_2018_binary
from feature_raster.exceptions.some_exceptions import InvalidTypeOfGeom
from feature_raster.Sensors.Landsat import Landsat8


class Landsat8RealCobertureTest(unittest.TestCase):
    def setUp(self):
        self.landsat_8 = Landsat8(img_path=envelope_2018)
        self.string = coberture_2018_binary

    def test_geom_path_is_not_a_valid_file(self):
        with self.assertRaises(InvalidTypeOfGeom):
            self.landsat_8.coberture_to_pandas(self.string)
