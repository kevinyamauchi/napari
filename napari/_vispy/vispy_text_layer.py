import numpy as np
from copy import copy
from vispy.scene.visuals import Text as TextNode
from vispy.visuals.transforms import ChainTransform

from ..layers import Text
from .vispy_base_layer import VispyBaseLayer
import numpy as np


class VispyTextLayer(VispyBaseLayer):
    _highlight_color = (0, 0.6, 1)
    _highlight_width = 1.5

    def __init__(self, layer):
        # Create a compound visual with the following four subvisuals:
        # Lines: The lines of the interaction box used for highlights.
        # Markers: The the outlines for each point used for highlights.
        # Markers: The actual markers of each point.
        node = TextNode()
        super().__init__(layer, node)

        self.layer.dims.events.ndisplay.connect(
            lambda e: self._on_display_change()
        )

        self._on_display_change()

    def _on_display_change(self):
        parent = self.node.parent
        self.node.transforms = ChainTransform()
        self.node.parent = None

        if self.layer.dims.ndisplay == 2:
            self.node = TextNode()
        else:
            self.node = TextNode()

        self.node.parent = parent
        self.layer._update_dims()
        self.layer._set_view_slice()
        self.reset()

    def _on_data_change(self):

        # Set vispy data, noting that the order of the points needs to be
        # reversed to make the most recently added point appear on top
        # and the rows / columns need to be switch for vispys x / y ordering
        if len(self.layer._data_view) == 0:
            data = np.zeros((1, self.layer.dims.ndisplay))
            size = [0]
        else:
            data = self.layer._data_view
            annotations = self.layer._annotations_view

        print(data)
        print(annotations)

        self.node.text = annotations
        self.node.pos = data

        # self.node.set_data(
        #     annotations,
        #     color=color,
        #     bold=self.layer.bold,
        #     italic=self.layer.italic,
        #     face=self.layer.font,
        #     face_size=self.layer.face_size,
        #     pos = data,
        #     anchor_x=self.layer.anchor_x,
        #     anchor_y=self.layer.anchor_y,
        #     method=self.layer.render_method,
        #     font_manager=None
        # )
        self.node.update()

    def _on_highlight_change(self):
        if self.layer.dims.ndisplay == 3:
            return

        if len(self.layer._highlight_index) > 0:
            # Color the hovered or selected points
            data = self.layer._data_view[self.layer._highlight_index]
            if data.ndim == 1:
                data = np.expand_dims(data, axis=0)
            size = self.layer._sizes_view[self.layer._highlight_index]
            face_color = [
                self.layer.face_colors[i]
                for i in self.layer._indices_view[self.layer._highlight_index]
            ]
        else:
            data = np.zeros((1, self.layer.dims.ndisplay))
            size = 0
            face_color = 'white'

        self.node._subvisuals[1].set_data(
            data[:, ::-1] + 0.5,
            size=size,
            edge_width=self._highlight_width,
            symbol=self.layer.symbol,
            edge_color=self._highlight_color,
            face_color=face_color,
            scaling=True,
        )

        if 0 in self.layer._highlight_box.shape:
            pos = np.zeros((1, 2))
            width = 0
        else:
            pos = self.layer._highlight_box
            width = self._highlight_width

        self.node._subvisuals[2].set_data(
            pos=pos[:, ::-1] + 0.5, color=self._highlight_color, width=width
        )

    def reset(self):
        self._reset_base()
        self._on_data_change()
        # self._on_highlight_change()
