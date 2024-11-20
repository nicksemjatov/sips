import sys
from gui.sips_gui import SIPS
import numpy as np
import cv2
sys.path.insert(0, '..')

if __name__ == '__main__':
    start_image = cv2.imread("Startpage SIPS.tiff", cv2.IMREAD_UNCHANGED)
    start_gui = SIPS(start_image[..., np.newaxis], 1000, 1, ui_config_file="sips_config.toml", icon_file="SIPS_icon.png")
