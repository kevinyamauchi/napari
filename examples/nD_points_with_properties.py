"""
Display one points layer ontop of one 4-D image layer using the
add_points and add_image APIs, where the markes are visible as nD objects
accross the dimensions, specified by their size
"""

from math import ceil

import numpy as np
from skimage import data
import napari


with napari.gui_qt():
    blobs = data.binary_blobs(
                length=128, blob_size_fraction=0.05, n_dim=3, volume_fraction=0.05
            )
    viewer = napari.view_image(blobs.astype(float))

    # create the points
    points = []
    for z in range(blobs.shape[0]):
        points += [
                [z, 43, 43],
                [z, 43, 86],
                [z, 86, 43],
                [z, 86, 86]
        ]

    face_property = np.array([True, True, True, True, False, False, False, False] * int((blobs.shape[0] / 2)))
    edge_property = np.array(['A', 'B', 'C', 'D', 'E'] * ceil(len(points) / 5))[0:len(points)]

    properties = {
        'face_property': face_property,
        'edge_property': edge_property
    }

    points_layer = viewer.add_points(
        points, properties=properties,
        size=3,
        edge_width=5,
        edge_color='edge_property',
        face_color='face_property',
        n_dimensional=False
    )

    # change the color cycles
    points_layer.face_color_cycle = ['white', 'black']
    points_layer.edge_color_cycle = ['c', 'm', 'y', 'k']