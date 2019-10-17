"""
Display a points layer on top of an image layer using the add_points and
add_image APIs
"""

import numpy as np
from skimage import data
from skimage.color import rgb2gray
import napari


with napari.gui_qt():
    # add the image
    viewer = napari.view_image(rgb2gray(data.astronaut()))
    # add the points
    text_pos = np.array([[100, 100], [200, 200], [333, 111]])
    text = np.asarray(['hi', 'hola', 'bonjour'])
    viewer.add_text(text_pos, annotations=text)
