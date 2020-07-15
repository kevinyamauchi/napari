"""
Perform a segmentation and annotate the results with
bounding boxes and text
"""
import numpy as np
from skimage import data
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square, remove_small_objects
import napari


def segment(image):
    """Segment an image using an intensity threshold determined via
    Otsu's method.

    Parameters
    ----------
    image : np.ndarray
        The image to be segmented

    Returns
    -------
    label_image : np.ndarray
        The resulting image where each detected object labeled with a unique integer.
    """
    # apply threshold
    thresh = threshold_otsu(image)
    bw = closing(image > thresh, square(4))

    # remove artifacts connected to image border
    cleared = remove_small_objects(clear_border(bw), 20)

    # label image regions
    label_image = label(cleared)

    return label_image


def make_bbox(bbox_extents):
    """Get the coordinates of the corners of a
    bounding box from the extents

    Parameters
    ----------
    bbox_extents : Tuple[int]
        The extents of the bounding box.
        Should be ordered: min_row, min_column, max_row, max_column

    Returns
    -------
    bbox_rect : np.ndarray
        The corners of the bounding box. Can be input directly into a
        napari Shapes layer.
    """
    minr = bbox_extents[0]
    minc = bbox_extents[1]
    maxr = bbox_extents[2]
    maxc = bbox_extents[3]

    bbox_rect = np.array(
        [[minr, minc], [maxr, minc], [maxr, maxc], [minr, maxc]]
    )

    return bbox_rect


def calculate_circularity(perimeter, area):
    """Calculate the circularity of the region

    Parameters
    ----------
    perimeter : float
        the perimeter of the region
    area : float
        the area of the region

    Returns
    -------
    circularity : float
        The circularity of the region as defined by 4*pi*area / perimeter^2
    """
    circularity = 4 * np.pi * area / (perimeter ** 2)

    return circularity


# load the image and segment it
image = data.coins()[50:-50, 50:-50]
label_image = segment(image)

# get the properties of the segmentation
regions = regionprops(label_image)
bbox_rects = [make_bbox(reg.bbox) for reg in regions]
labels = [reg.label for reg in regions]
circularity = [
    calculate_circularity(reg.perimeter, reg.area) for reg in regions
]

# create the properties dictionary
properties = {
    'label': labels,
    'circularity': circularity,
}

# specify the display parameters for the text
text_parameters = {
    'text': 'label: {label}\ncirc: {circularity:.2f}',
    'size': 12,
    'color': 'green',
    'anchor': 'upper_left',
    'translation': [-3, 0],
}

with napari.gui_qt():
    # initialise viewer with coins image
    viewer = napari.view_image(image, name='coins', rgb=False)

    # add the labels
    label_layer = viewer.add_labels(label_image, name='segmentation')

    shapes_layer = viewer.add_shapes(
        bbox_rects,
        face_color='transparent',
        edge_color='green',
        properties=properties,
        text=text_parameters,
        name='bounding box',
    )
