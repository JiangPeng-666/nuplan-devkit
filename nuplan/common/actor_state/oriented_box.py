from sqlalchemy import null
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.actor_state.transform_state import translate_position
from nuplan.common.actor_state.utils import lazy_property
from shapely.geometry import Polygon

from nuplan.database.utils.boxes.box3d import Box3D
from nuplan.database.utils.geometry import yaw_to_quaternion


class OrientedBox:
    """ Represents the physical space occupied by agents on the plane. """

    def __init__(self, center: StateSE2, length: float, width: float, height: float):
        """
        :param center: The pose of the geometrical center of the box
        :param length: The length of the OrientedBox
        :param width: The width of the OrientedBox
        :param height: The height of the OrientedBox
        """
        self._center = center
        self._length = length
        self._width = width
        self._height = height

    @property
    def width(self) -> float:
        """
        Returns the width of the OrientedBox
        :return: The width of the OrientedBox
        """
        return self._width

    @property
    def length(self) -> float:
        """
        Returns the length of the OrientedBox
        :return: The length of the OrientedBox
        """
        return self._length

    @property
    def height(self) -> float:
        """
        Returns the height of the OrientedBox
        :return: The height of the OrientedBox
        """
        return self._height

    @property
    def center(self) -> StateSE2:
        """
        Returns the pose of the center of the OrientedBox
        :return: The pose of the center
        """
        return self._center

    @lazy_property
    def geometry(self) -> Polygon:
        """
        Returns the Polygon describing the OrientedBox, if not done yet it will build it lazily.
        :return: The Polygon of the OrientedBox
        """
        return self._make_polygon()

    @lazy_property
    def box3d(self) -> Box3D:
        '''
        Returns the Box3D describing the OrientedBox
        :return: The Box3D of the OrientedBox
        '''
        return self._make_box3d()

    def _make_polygon(self) -> Polygon:
        """  Creates a polygon which is a rectangle centered on the oriented box center, with given width and length """
        half_width = self.width / 2.0
        half_length = self.length / 2.0
        corners = [tuple(translate_position(self.center, half_length, half_width)),
                   tuple(translate_position(self.center, -half_length, half_width)),
                   tuple(translate_position(self.center, -half_length, -half_width)),
                   tuple(translate_position(self.center, half_length, -half_width))]
        return Polygon(corners)
    
    def _make_box3d(self) -> Box3D:
        return Box3D(
            center = (self.center.x, self.center.y, 0.0), 
            size = (self.width, self.length, self.height), 
            orientation = yaw_to_quaternion(self.center.heading)
        )