import numpy as np
import scipy.ndimage as sndi
from . import helper


def ras_to_vtk(ijk_to_ras):
    """
    Compute the transform between RAS and VTK space

    Args:
        ijk_to_ras (np.array(dtype=np.float64)): Volume's IJK to RAS transformation

    Returns:
        np.array(dtype=np.float64): Volume's RAS to VTK transformation
    """

    res = np.linalg.norm(ijk_to_ras[:3, :3], axis=0)
    ras_to_ijk = np.linalg.inv(ijk_to_ras)

    m = np.array([[0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]]) @ (np.diag(np.append(res, 1)) @ ras_to_ijk)

    return m


def vtk_to_ras(ijk_to_ras):
    """
    Compute the transform between VTK and RAS space, given transf_ijk_to_ras transform

    Args:
        ijk_to_ras (np.array(dtype=np.float64)): Volume's IJK to RAS transformation

    Returns:
        np.array(dtype=np.float64): Volume's VTK to RAS transformation
    """

    res = np.linalg.norm(ijk_to_ras[:3, :3], axis=0)

    return (ijk_to_ras @ np.diag(1 / np.append(res, 1))) @ np.array(
        [[0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]])


class volume:
    """
    Class to represent a volume
    """

    def __init__(self, data=np.zeros((0, 0, 0)), ijk_to_ras=np.eye(4)):
        """
        Initializes a volume

        Args:
            data (np.array(dtype=np.float64), optional): Volume's data. Defaults to np.zeros((0, 0, 0)).
            ijk_to_ras (np.array(dtype=np.float64), optional): Volume's IJK to RAS transformation.
                                                               Defaults to np.eye(4).

        Raises:
            ValueError: If volume's data or matrix transformation isn't good shape
        """

        if len(data.shape) != 3 or ijk_to_ras.shape != (4, 4):
            raise ValueError

        self._vol, self.ijk_to_ras = data, ijk_to_ras

        # Default to linear interpolation
        self._order = 1

    def __call__(self, p):
        """
        Compute the values at positions p by interpolation.
        See order property for the interpolation (scipy.ndimages.map_coordinates is used)

        Args:
            p (np.array(dtype=np.float64)): 3D points in RAS coordinates

        Returns:
            np.array(dtype=np.float64): Volume's data mapped to new coordinates
        """

        return sndi.map_coordinates(self._vol, self.transf_ras_to_ijk(p).T, order=self.order, prefilter=False)

    def transf_ijk_to_ras(self, p):
        """
        Transforms a set of IJK coordinates into RAS coordinates

        Args:
            p (np.array(dtype=np.float64)): Points in IJK coordinates

        Returns:
            np.array(dtype=np.float64): Points in RAS coordinates
        """

        return helper.homogenize(p) @ self._ijk_to_ras.T[:, :3]

    def transf_ras_to_ijk(self, p):
        """
        Transforms a set of RAS coordinates into IJK coordinates

        Args:
            p (np.array(dtype=np.float64)): Points in RAS coordinates

        Returns:
            np.array(dtype=np.float64): Points in IJK coordinates
        """

        return helper.homogenize(p) @ self._ras_to_ijk.T[:, :3]

    def transf_vtk_to_ras(self, p):
        """
        Transforms a set of VTK coordinates into RAS coordinates

        Args:
            p (np.array(dtype=np.float64)): Points in VTK coordinates

        Returns:
            np.array(dtype=np.float64): Points in RAS coordinates
        """

        return helper.homogenize(p) @ self._vtk_to_ras.T[:, :3]

    def transf_ras_to_vtk(self, p):
        """
        Transforms a set of RAS coordinates into VTK coordinates

        Args:
            p (np.array(dtype=np.float64)): Points in RAS coordinates

        Returns:
            np.array(dtype=np.float64): Points in VTK coordinates
        """

        return helper.homogenize(p) @ self._ras_to_vtk.T[:, :3]

    def get_line(self, start, end, n_samples=128):
        """
        Extract n_samples values along a line going from start to end (both points are included). Points are expressed
        in RAS coordinates. Their coordinates can be retrieved by a simple call to np.linspace(start,end,n_samples).
        See also the order property to tune the order of the spline for the interpolation
        It defaults to one for a linear interpolation

        Args:
            start (np.array(dtype=np.float64)): Samples start position
            end (np.array(dtype=np.float64)): Samples end position
            n_samples (int, optional): Number of samples. Defaults to 128.

        Returns:
            np.array(dtype=np.float64): Interpolated coordinates values
            np.array(dtype=np.float64): Source coordinate values
        """

        coord = np.linspace(start, end, n_samples)

        return self(coord), coord

    def get_patch(self, center, size, dim):
        """
        Extract a 3D patch from the volume with faces parallel to RAS frame coordinate directions
        Note: uses self.order to perform interpolation using scipy.map_coordinates

        Args:
            center (np.array(dtype=np.float64)): Center (in RAS) of the patch to extract
            size (np.array(dtype=np.float64)): Size in millimeters of the patch to extract
            dim (np.array(dtype=np.float64)): Voxel dimension of the patch

        Returns:
            volume: Patch as a volume, including the new transf_ras_to_ijk transform
        """

        if np.isscalar(dim):
            dim = dim * np.ones(3, dtype=np.uint32)

        if np.isscalar(size):
            size = size * np.ones(3, dtype=np.float32)

        center = np.asarray(center)
        dim = np.asarray(dim)
        size = np.asarray(size)

        # Compute new ras_to_ijk transform associated with patch
        res = size / dim
        trans_patch = np.eye(4)

        # The patch axes are parallel to the metric space axes
        trans_patch[:3, :3] = np.diag(res)
        trans_patch[:3, 3] = center - size / 2 + res / 2

        # Express this transform in global ijk frame
        t = self._ras_to_ijk @ trans_patch

        # RAS coordinates of voxels in the patch
        # Voxel coordinates (wrt patch), in homogeneous space
        p = np.vstack((np.mgrid[0:dim[0], 0:dim[1], 0:dim[2]].reshape((3, -1)), np.ones(np.prod(dim))))

        # Express the points in ras, then global ijk coordinates
        c = (t @ p)[:3]

        # Note: could avoid computations by avoiding multiple transforms (TB confirmed)
        return volume(sndi.map_coordinates(self._vol, c, order=self.order, prefilter=False).reshape(dim), trans_patch)

    # Need to validate this axis parameter. Not sure this is working with a different setup

    def mip_axes(self, axis=(0, 1, 2)):
        """
        Compute MIP images along given axes (by default, compute axial, coronal and sagittal MIP views)

        Args:
            axis (tuple, optional): Axis index. Defaults to (0, 1, 2).

        Returns:
            np.array(dtype=np.float64): MIP images along axial axe
            np.array(dtype=np.float64): MIP images along coronal axe
            np.array(dtype=np.float64): MIP images along sagittal axe
        """

        v = np.flip(self._vol, (0, 1, 2)).transpose((2, 1, 0))
        ax, cor, sag = (np.max(v, axis=i) for i in axis)

        return ax, cor, sag

    @property
    def n_planes(self):
        """
        Getter for number of planes

        Returns:
            int: Number of planes
        """

        return self._vol.shape[0]

    @property
    def n_lines(self):
        """
        Getter for number of lines

        Returns:
            int: Number of lines
        """

        return self._vol.shape[1]

    @property
    def n_columns(self):
        """
        Getter for number of columns

        Returns:
            int: Number of columns
        """

        return self._vol.shape[2]

    @property
    def dimensions(self):
        """
        Getter for dimensions

        Returns:
            tuple: Dimensions
        """

        return self._vol.shape

    @property
    def voxel_size(self):
        """
        Getter for voxel size

        Returns:
            np.array(dtype=np.float64): Voxel size
        """

        return np.linalg.norm(self._ijk_to_ras[:3, :3], axis=0)

    @property
    def ijk_to_ras(self):
        """
        Getter for IJK to RAS transform

        Returns:
            np.array(dtype=np.float64): IJK to RAS transform
        """

        return self._ijk_to_ras

    @ijk_to_ras.setter
    def ijk_to_ras(self, m):
        """
        Setter for IJK to RAS transform.
        Modify the other transforms too.

        Args:
            m (np.array(dtype=np.float64)): New IJK to RAS transform
        """

        self._ijk_to_ras = m.copy()
        self._ras_to_ijk = np.linalg.inv(self._ijk_to_ras)
        self._vtk_to_ras = vtk_to_ras(self._ijk_to_ras)
        self._ras_to_vtk = ras_to_vtk(self._ijk_to_ras)

    @property
    def ras_to_ijk(self):
        """
        Getter for RAS to IJK transform

        Returns:
            np.array(dtype=np.float64): RAS to IJK transform
        """

        return self._ras_to_ijk

    @ras_to_ijk.setter
    def ras_to_ijk(self, m):
        """
        Setter for RAS to IJK transform.
        Modify the other transforms too.

        Args:
            m (np.array(dtype=np.float64)): New RAS to IJK transform
        """

        self._ras_to_ijk = m.copy()
        self._ijk_to_ras = np.linalg.inv(self._ras_to_ijk)
        self._vtk_to_ras = vtk_to_ras(self._ijk_to_ras)
        self._ras_to_vtk = ras_to_vtk(self._ijk_to_ras)

    @property
    def data(self):
        """
        Getter for data

        Returns:
            np.array(dtype=np.float64): Volume's data
        """

        return self._vol

    @data.setter
    def data(self, d):
        """
        Setter for data

        Args:
            d (np.array(dtype=np.float64)): New volume's data

        Raises:
            TypeError: Bad dimension (New data dimension not equal to 3)
        """

        if len(d.shape) != 3:
            raise TypeError

        self._vol = d

    @property
    def order(self):
        """
        Getter for order

        Returns:
            int: Order
        """

        return self._order

    @order.setter
    def order(self, value):
        """
        Setter for order

        Args:
            value (int): New order value

        Raises:
            ValueError: Negative order value
        """

        if value < 0:
            raise ValueError('Volume.order value must be positive or zero')

        self._order = value

    @property
    def bbox(self):
        """
        Bounding box in RAS coordinates.
        The min pos corresponds to the center of the voxel in the lower, posterior, left corner.
        The max pos corresponds to the center of the voxel in the upper, anterior, right corner.

        Returns:
            np.array(dtype=np.float64): Bounding box in RAS coordinates. The first line is the min pos, and
                                        the second line is the max pos. It ensures that for each i in [0,2],
                                        min_pos[i] <= max_pos[i]
        """

        return np.sort(self.transf_ijk_to_ras(np.vstack((np.zeros(3), np.asarray(self._vol.shape) - 1))), axis=0)

    @property
    def min(self):
        """
        Getter for min value

        Returns:
            float: Min value
        """

        return np.min(self._vol.ravel())

    @property
    def max(self):
        """
        Getter for max value

        Returns:
            float: Max value
        """

        return np.max(self._vol.ravel())
