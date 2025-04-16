import os
import unittest

import gnbg

DATA_PATH =  os.path.realpath(os.path.join(
    os.path.realpath(os.path.dirname(__file__)), "../data"
))

class TestCanLoad(unittest.TestCase):
    def test_load(self):
        gnbg.set_root(DATA_PATH + "/")
        print(gnbg.GNBG(1))
        
if __name__ == "__main__":
    unittest.main()