from typing import Union
from xml.etree.ElementTree import Element
import numpy as np
import itertools
from copy import copy, deepcopy
from contextlib import contextmanager
from ..base import Layer
from ...util.event import Event
from ...util.misc import ensure_iterable
from ...util.status_messages import format_float
from vispy.color import get_color_names, Color
from ._constants import Symbol, SYMBOL_ALIAS, Mode


class Text(Layer):
    """Points layer.

    Parameters
    ----------
    data : array (N, D)
        Coordinates for N text labels in D dimensions.
    symbol : str
        Symbol to be used for the point markers. Must be one of the
        following: arrow, clobber, cross, diamond, disc, hbar, ring,
        square, star, tailed_arrow, triangle_down, triangle_up, vbar, x.
    size : float, array
        Size of the point marker. If given as a scalar, all points are made
        the same size. If given as an array, size must be the same
        broadcastable to the same shape as the data.
    edge_width : float
        Width of the symbol edge in pixels.
    edge_color : str
        Color of the point marker border.
    face_color : str
        Color of the point marker body.
    n_dimensional : bool
        If True, renders points not just in central plane but also in all
        n-dimensions according to specified point marker size.
    name : str
        Name of the layer.
    metadata : dict
        Layer metadata.
    scale : tuple of float
        Scale factors for the layer.
    translate : tuple of float
        Translation values for the layer.
    opacity : float
        Opacity of the layer visual, between 0.0 and 1.0.
    blending : str
        One of a list of preset blending modes that determines how RGB and
        alpha values of the layer visual get mixed. Allowed values are
        {'opaque', 'translucent', and 'additive'}.
    visible : bool
        Whether the layer visual is currently being displayed.

    Attributes
    ----------
    data : array (N, D)
        Coordinates for N text labels in D dimensions.
    color : str
        Font color for the text
    bold : bool
        Font is bold if set to true.
    italic : bool
        Font is italic if set to true
    font : str
        Font to use
    font_size : float
        Font size in points
    annotations : List[str]
        List of the annotations for the text labels. Indices should be matched to N in data
    rotation : float
        Angle of the text in degrees (counter clockwise is positive)
    anchor_x : str
        Alignment of the text in the x-axis
    anchor_y : str
        Alignment of the text in the y-axis
    render_method : str
        Method of rendering. Should be either 'cpu' or 'gpu'

    
    mode : str
        Interactive mode. The normal, default mode is PAN_ZOOM, which
        allows for normal interactivity with the canvas.

        In ADD mode clicks of the cursor add points at the clicked location.

        In SELECT mode the cursor can select points by clicking on them or
        by dragging a box around them. Once selected points can be moved,
        have their properties edited, or be deleted.

    Extended Summary
    ----------
    _data_view : array (M, 2)
        2D coordinates of points in the currently viewed slice.
    _sizes_view : array (M, )
        Size of the point markers in the currently viewed slice.
    _indices_view : array (M, )
        Integer indices of the points in the currently viewed slice.
    _selected_view :
        Integer indices of selected points in the currently viewed slice within
        the `_data_view` array.
    _selected_box : array (4, 2) or None
        Four corners of any box either around currently selected points or
        being created during a drag action. Starting in the top left and
        going clockwise.
    _drag_start : list or None
        Coordinates of first cursor click during a drag action. Gets reset to
        None after dragging is done.
    """

    # The max number of points that will ever be used to render the thumbnail
    # If more points are present then they are randomly subsampled
    _max_points_thumbnail = 1024

    def __init__(
        self,
        data=None,
        *,
        color='black',
        bold=False,
        italic=False,
        font_size=12,
        annotations=None,
        rotation=0,
        anchor_x='center',
        anchor_y='center',
        render_method='cpu',
        n_dimensional=False,
        name=None,
        metadata=None,
        scale=None,
        translate=None,
        opacity=1,
        blending='translucent',
        visible=True,
    ):
        if data is None:
            data = np.empty((0, 2))
        ndim = data.shape[1]
        super().__init__(
            ndim,
            name=name,
            metadata=metadata,
            scale=scale,
            translate=translate,
            opacity=opacity,
            blending=blending,
            visible=visible,
        )

        self.events.add(n_dimensional=Event)

        # Save the text coordinates
        self._data = data
        self.dims.clip = False

        # Save the text style params
        self.annotations = np.asarray(annotations)
        self._n_dimensional = n_dimensional
        self._color = color
        self._bold = bold
        self._italic = italic
        self._font_size = font_size
        self._rotation = rotation
        self._anchor_x = anchor_x
        self._anchor_y = anchor_y
        self._render_method = render_method

        # Trigger generation of view slice and thumbnail
        self._update_dims()

    @property
    def data(self) -> np.ndarray:
        """(N, D) array: coordinates for N points in D dimensions."""
        return self._data

    @data.setter
    def data(self, data: np.ndarray):
        cur_npoints = len(self._data)
        self._data = data

        # Adjust the size array when the number of points has changed
        if len(data) < cur_npoints:
            # If there are now less points, remove the sizes and colors of the
            # extra ones
            with self.events.set_data.blocker():
                self.edge_colors = self.edge_colors[: len(data)]
                self.face_colors = self.face_colors[: len(data)]
                self.sizes = self._sizes[: len(data)]

        elif len(data) > cur_npoints:
            # If there are now more points, add the sizes and colors of the
            # new ones
            with self.events.set_data.blocker():
                adding = len(data) - cur_npoints
                if len(self._sizes) > 0:
                    new_size = copy(self._sizes[-1])
                    for i in self.dims.displayed:
                        new_size[i] = self.size
                else:
                    # Add the default size, with a value for each dimension
                    new_size = np.repeat(self.size, self._sizes.shape[1])
                size = np.repeat([new_size], adding, axis=0)
                self.edge_colors += [self.edge_color for i in range(adding)]
                self.face_colors += [self.face_color for i in range(adding)]
                self.sizes = np.concatenate((self._sizes, size), axis=0)
        self._update_dims()
        self.events.data()

    def _get_ndim(self):
        """Determine number of dimensions of the layer."""
        return self.data.shape[1]

    def _get_extent(self):
        """Determine ranges for slicing given by (min, max, step)."""
        if len(self.data) == 0:
            maxs = np.ones(self.data.shape[1], dtype=int)
            mins = np.zeros(self.data.shape[1], dtype=int)
        else:
            maxs = np.max(self.data, axis=0)
            mins = np.min(self.data, axis=0)

        return [(min, max, 1) for min, max in zip(mins, maxs)]

    @property
    def n_dimensional(self) -> str:
        """bool: renders points as n-dimensionsal."""
        return self._n_dimensional

    @n_dimensional.setter
    def n_dimensional(self, n_dimensional: bool) -> None:
        self._n_dimensional = n_dimensional
        self.events.n_dimensional()
        self._set_view_slice()

    @property
    def selected_data(self):
        """list: list of currently selected points."""
        return self._selected_data

    @selected_data.setter
    def selected_data(self, selected_data):
        self._selected_data = list(selected_data)
        selected = []
        for c in self._selected_data:
            if c in self._indices_view:
                ind = list(self._indices_view).index(c)
                selected.append(ind)
        self._selected_view = selected
        self._selected_box = self.interaction_box(self._selected_view)

        # Update properties based on selected points
        index = self._selected_data
        edge_colors = list(set([self.edge_colors[i] for i in index]))
        if len(edge_colors) == 1:
            edge_color = edge_colors[0]
            with self.block_update_properties():
                self.edge_color = edge_color

        face_colors = list(set([self.face_colors[i] for i in index]))
        if len(face_colors) == 1:
            face_color = face_colors[0]
            with self.block_update_properties():
                self.face_color = face_color

        size = list(
            set([self.sizes[i, self.dims.displayed].mean() for i in index])
        )
        if len(size) == 1:
            size = size[0]
            with self.block_update_properties():
                self.size = size

    def _slice_data(self, indices):
        """Determines the slice of points given the indices.

        Parameters
        ----------
        indices : sequence of int or slice
            Indices to slice with.

        Returns
        ----------
        in_slice_data : (N, 2) array
            Coordinates of points in the currently viewed slice.
        slice_indices : list
            Indices of points in the currently viewed slice.
        scale : float, (N, ) array
            If in `n_dimensional` mode then the scale factor of points, where
            values of 1 corresponds to points located in the slice, and values
            less than 1 correspond to points located in neighboring slices.
        """
        # Get a list of the data for the points in this slice
        not_disp = list(self.dims.not_displayed)
        disp = list(self.dims.displayed)
        indices = np.array(indices)
        if len(self.data) > 0:
            if self.n_dimensional is True and self.ndim > 2:
                distances = abs(self.data[:, not_disp] - indices[not_disp])
                # sizes = self.sizes[:, not_disp] / 2
                matches = np.all(distances <= 0.5, axis=1)
                in_slice_data = self.data[np.ix_(matches, disp)]
                indices = np.where(matches)[0].astype(int)
                return in_slice_data, indices, scale
            else:
                data = self.data[:, not_disp].astype('int')
                matches = np.all(data == indices[not_disp], axis=1)
                in_slice_data = self.data[np.ix_(matches, disp)]

                annotations = self.annotations[matches]
                indices = np.where(matches)[0].astype(int)
                return in_slice_data, indices, annotations
        else:
            return [], [], []

    def get_value(self):
        """Determine if points at current coordinates.

        Returns
        ----------
        selection : int or None
            Index of point that is at the current coordinate if any.
        """
        in_slice_data = self._data_view

        # Display text if there are any in this slice
        if len(self._data_view) > 0:
            # Get the point sizes
            distances = abs(
                self._data_view
                - [self.coordinates[d] for d in self.dims.displayed]
            )
            in_slice_matches = np.all(distances <= 0.5, axis=1)
            indices = np.where(in_slice_matches)[0]
            if len(indices) > 0:
                selection = self._indices_view[indices[-1]]
            else:
                selection = None
        else:
            selection = None

        return selection

    def _set_view_slice(self):
        """Sets the view given the indices to slice with."""

        in_slice_data, indices, annotations = self._slice_data(
            self.dims.indices
        )

        # Display points if there are any in this slice
        if len(in_slice_data) > 0:

            # Update the points node
            data = np.array(in_slice_data)

        else:
            # if no points in this slice send dummy data
            data = np.zeros((0, self.dims.ndisplay))

        self._data_view = data
        self._annotations_view = annotations
        self._indices_view = indices

        # self._update_thumbnail()
        self._update_coordinates()
        self.events.set_data()

    def _update_thumbnail(self):
        """Update thumbnail with current points and colors."""
        colormapped = np.zeros(self._thumbnail_shape)
        colormapped[..., 3] = 1
        if len(self._data_view) > 0:
            min_vals = [self.dims.range[i][0] for i in self.dims.displayed]
            shape = np.ceil(
                [
                    self.dims.range[i][1] - self.dims.range[i][0] + 1
                    for i in self.dims.displayed
                ]
            ).astype(int)
            zoom_factor = np.divide(
                self._thumbnail_shape[:2], shape[-2:]
            ).min()
            if len(self._data_view) > self._max_points_thumbnail:
                inds = np.random.randint(
                    0, len(self._data_view), self._max_points_thumbnail
                )
                points = self._data_view[inds]
            else:
                points = self._data_view
            coords = np.floor(
                (points[:, -2:] - min_vals[-2:] + 0.5) * zoom_factor
            ).astype(int)
            coords = np.clip(
                coords, 0, np.subtract(self._thumbnail_shape[:2], 1)
            )
            for i, c in enumerate(coords):
                col = self.face_colors[self._indices_view[i]]
                colormapped[c[0], c[1], :] = Color(col).rgba
        colormapped[..., 3] *= self.opacity
        self.thumbnail = colormapped

    def to_xml_list(self):
        """Convert the points to a list of xml elements according to the svg
        specification. Z ordering of the points will be taken into account.
        Each point is represented by a circle. Support for other symbols is
        not yet implemented.

        Returns
        ----------
        xml : list
            List of xml elements defining each point according to the
            svg specification
        """
        xml_list = []
        width = str(self.edge_width)
        opacity = str(self.opacity)
        props = {'stroke-width': width, 'opacity': opacity}

        for i, d, s in zip(
            self._indices_view, self._data_view, self._sizes_view
        ):
            d = d[::-1]
            cx = str(d[0])
            cy = str(d[1])
            r = str(s / 2)
            face_color = (255 * Color(self.face_colors[i]).rgba).astype(np.int)
            fill = f'rgb{tuple(face_color[:3])}'
            edge_color = (255 * Color(self.edge_colors[i]).rgba).astype(np.int)
            stroke = f'rgb{tuple(edge_color[:3])}'

            element = Element(
                'circle', cx=cx, cy=cy, r=r, stroke=stroke, fill=fill, **props
            )
            xml_list.append(element)

        return xml_list
