import numpy as np
import math
import scipy.optimize as scopt

from .helper import (cross, homogenize)
from .segment import segment


class cylinder:
    """
    Class to represent a cylinder
    """

    def __init__(self, center=np.array([0, 0, 0], dtype=np.float64), radius=1,
                 direction=np.array([0, 0, 1], dtype=np.float64), height=-1, name=''):
        """
        Generate a new instance of cylinder.
        The radius should be positive (ValueError raised if negative or zero, see radius property setter)
        Direction is ensured to be of norm 1: ValueError raised if a zero-length vector is provided as input
        (see direction property setter)
        A negative height is always translated as -1 and encodes a cylinder of infinite length.
        Zero-height cylinders are handled

        Args:
            center (np.array(dtype=np.float64), optional): Cylinder's center position.
                                                           Defaults to np.array([0, 0, 0], dtype=np.float64).
            radius (int, optional): Cylinder's radius. Defaults to 1.
            direction (np.array(dtype=np.float64), optional): Cylinder's direction.
                                                              Defaults to np.array([0, 0, 1], dtype=np.float64).
            height (int, optional): Cylinder's height. Defaults to -1.
            name (str, optional): Cylinder's name. Defaults to ''.
        """

        self.center = np.asarray(center[0:3], dtype=np.float64)
        self.radius = radius
        self.direction = direction
        self.height = height
        self.name = str(name)

    def copy(self):
        """
        Copy cylinder

        Returns:
            cylinder: Copy of current cylinder
        """

        return cylinder(self.center, self.radius, self.direction, self.height, self.name)

    @property
    def center(self):
        """
        Getter of cylinder's center

        Returns:
            np.array(dtype=np.float64): Cylinder's center
        """

        return self._center

    @center.setter
    def center(self, c):
        """
        Setter of cylinder's center

        Args:
            c (np.array(dtype=np.float64)): New cylinder's center
        """

        self._center = np.asarray(c)

    @property
    def radius(self):
        """
        Getter of cylinder's radius

        Returns:
            int: Cylinder's radius
        """

        return self._radius

    @radius.setter
    def radius(self, r):
        """
        Setter of cylinder's radius

        Args:
            r (int): New cylinder's radius

        Raises:
            ValueError: r <= 0
        """

        if r <= 0:
            raise ValueError
        else:
            self._radius = r

    @property
    def direction(self):
        """
        Getter of cylinder's direction

        Returns:
            np.array(dtype=np.float64): Cylinder's direction
        """

        return self._direction

    @direction.setter
    def direction(self, d):
        """
        Setter of cylinder's direction

        Args:
            d (np.array(dtype=np.float64)): New cylinder's direction

        Raises:
            ValueError: norm(d) == 0
        """

        n = np.linalg.norm(d)

        if math.isclose(n, 0):
            raise ValueError
        else:
            self._direction = d / n

    @property
    def height(self):
        """
        Getter of cylinder's height

        Returns:
            int: Cylinder's height
        """

        return self._height

    @height.setter
    def height(self, h):
        """
        Setter of cylinder's height
        Sets height to -1 (aka infinite length) if h<0

        Args:
            h (int): New cylinder's height
        """

        if h < 0:
            self._height = -1
        else:
            self._height = h

    def __repr__(self):
        """
        Represent cylinder object as a string

        Returns:
            str: Representation of cylinder object
        """

        return f'{self.center[0]} {self.center[1]} {self.center[2]} {self.radius} {self.direction[0]} ' \
               f'{self.direction[1]} {self.direction[2]} {self.height}'

    def display(self):
        """
        Show current cylinder's information
        """

        print(self)

    def distance(self, p):
        """
        Signed distance to a set of points p. Positive outside the cylinder.

        Args:
            p (np.array(dtype=np.float64)): Nx3 array containing set of points

        Returns:
            int: Distance between cylinder's center and the set of points
        """

        if len(p.shape) == 1:
            pa = p.reshape((1, 3))
        else:
            pa = p

        # Compute distance to segment of length self.height
        if self.height >= 0:
            d = self.height / 2 * self.direction
            dist_to_axis = segment(self.center - d, self.center + d).distance_sqr(pa)

        # Distance to infinite line
        else:
            d = pa - self.center
            li = d @ self.direction
            dist_to_axis = np.sum(d * d, axis=1) - li * li

        # Handle case of numerical errors causing dist_to_axis to be negative
        dist_to_axis[dist_to_axis < 0] = 0
        dist = np.sqrt(dist_to_axis)

        if len(p.shape) == 1:
            return dist[0] - self.radius

        return dist - self.radius

    def select_inliers(self, p, threshold):
        """
        Compute inliers from a point set p, that lies with threshold distance to the cylinder

        Args:
            p (np.array(dtype=np.float64)): Input point set Nx3
            threshold (float): Max distance to the cylinder (absolute value)

        Returns:
            (np.array(dtype=np.float64)): Inlier points selected
        """

        d = np.fabs(self.distance(p))
        i = np.where(d < threshold)

        return p[i]

    def fix_center(self, inliers):
        """
        Fixes the center location so that it lies at a median position along the axis, with respect to the inlier points

        Args:
            inliers (np.array(dtype=np.float64)): Inlier points
        """

        li = np.sort(self.direction @ (inliers - self.center).T)
        self.center += li[len(li) // 2] * self.direction

    def fix_height(self, inliers):
        """
        Fixes the height of the cylinder.
        First, only 75% of the inlier points are kept: those that are closest to the center, as per the distance along
        the cylinder axis
        Then, compute the height so that it encompasses all the inlier points.

        Args:
            inliers (np.array(dtype=np.float64)): Input inlier point set

        Returns:
            np.array(dtype=np.float64): The inlier points that are kept according to the height
        """

        # Compute distances to center along direction
        li = np.abs(self.direction @ (inliers - self.center).T)
        rm = len(li) // 4

        # Sort and removes the rm farthest points, use - for decreasing order
        idx = np.argsort(-li)[rm:]

        # Height is twice of the greatest distance of kept points
        self.height = 2 * li[idx[0]]

        # Keep the closest points
        return inliers[idx]

    def refine(self, inliers):
        """
        Refines the cylinder axis so that the distance to the inlier points is minimized

        Args:
            inliers (np.array(dtype=np.float64)): Inlier point set to consider
        """

        def cyl_to_param(cyl):
            return np.append(cyl.center, cyl.radius * cyl.direction)

        def param_to_cyl(param):
            return cylinder(param[:3], np.linalg.norm(param[3:]), param[3:], self.height)

        def residue(param):
            cyl = param_to_cyl(param)
            d = cyl.distance(inliers)

            return d @ d / inliers.shape[0]

        prm = cyl_to_param(self)
        res = scopt.minimize(residue, prm)

        if res.success:
            c = param_to_cyl(res.x)
            self.center = c.center
            self.radius = c.radius
            self.direction = c.direction

    def is_redundant(self, b):
        """
        Determine if the cylinder is redundant with a branch.

        Args:
            b (list): The branch, as a list of cylinders, ordered to form a polygonal line (list of joint segments)

        Returns:
            bool: True if self is redundant, meaning that it lies with self.radius/10 of a segment in the branch,
                  else False
                  False if there are less than 2 points (strictly) in the branch
        """

        r_max2 = (self.radius / 10) ** 2

        return len(b) >= 2 and min(
            [segment(s.center, e.center).distance_sqr(self.center) for s, e in zip(b[:-1], b[1:])]) < r_max2


def from_string(s):
    """
    Reads cylinder info from a string. Space is used as a separator for the items. There can be between 1 and up
    to 8 items (if more, only the first 8 items are considered, and the other ones are discarded).

    The stored information is:
        CenterX CenterY CenterZ Radius DirectionX DirectionY DirectionZ Height
    (ie inverse of Cylinder.__repr__)

    For example a string with only 2 elements will read the first two coordinates of the center (CenterX and CenterY,
    CenterZ will be set to 0).

    Default center coordinates are 0
    Default Radius is 1
    Default direction is (0,0,1) (also used if given direction is zero length)
    Default height is -1 (infinite cylinder)

    Args:
        s (str): Cylinder description

    Returns:
        cylinder: Cylinder created from its description
    """

    v = np.fromstring(s, dtype=float, sep=' ')

    if len(v) == 0:
        return cylinder()
    elif len(v) == 1:
        return cylinder(center=[v[0], 0, 0])
    elif len(v) == 2:
        return cylinder(center=[v[0], v[1], 0])
    elif len(v) == 3:
        return cylinder(center=v[0:3])
    elif len(v) == 4:
        return cylinder(center=v[0:3], radius=v[3])
    elif len(v) == 5:
        return cylinder(center=v[0:3], radius=v[3], direction=[v[4], 0, 0])
    elif len(v) == 6:
        return cylinder(center=v[0:3], radius=v[3], direction=[v[4], v[5], 0])
    elif len(v) == 7:
        return cylinder(center=v[0:3], radius=v[3], direction=v[4:7])
    else:
        return cylinder(center=v[0:3], radius=v[3], direction=v[4:7], height=v[7])


def fit_3_points(p0, p1, p2, direction):
    """
    Determine the cylinder going through 3 points p0,p1 and p2 and whose axis is along direction.
    Note that if direction were not given, 5 points would be required.

    Args:
        p0 (np.array(dtype=np.float64)): First point to use
        p1 (np.array(dtype=np.float64)): Second point to use
        p2 (np.array(dtype=np.float64)): Third point to use
        direction (np.array(dtype=np.float64)): Direction to have cylinder axis

    Returns:
        cylinder: The cylinder or None in case of degeneracies (direction is zero-normed, direction is contained in the
                  plane defined by the 3 points)
    """

    try:
        # Normalize direction (if possible)
        n = np.linalg.norm(direction)

        if math.isclose(n, 0):
            return None

        direction /= n

        # Remove the component along direction
        q0 = p0 - (p0 @ direction) * direction
        q1 = p1 - (p1 @ direction) * direction
        q2 = p2 - (p2 @ direction) * direction

        d10 = q1 - q0
        d20 = q2 - q0
        d21 = q2 - q1

        # Direction is with plane (p0,p1,p2)
        s = np.fabs(cross(d10, d20) @ direction)

        if math.isclose(s, 0):
            return None

        radius = np.sqrt((d10 @ d10) * (d20 @ d20) * (d21 @ d21)) / (2 * s)

        m = np.vstack((d10, d20, direction))

        n0 = q0 @ q0
        n1 = q1 @ q1
        n2 = q2 @ q2

        b = np.array([0.5 * (n1 - n0), 0.5 * (n2 - n0), 0])
        c = np.linalg.inv(m) @ b

    except Exception:
        return None

    return cylinder(center=c, radius=radius, direction=direction)


def change_frame(in_cyl, t):
    """
    Change of coordinate frame.

    Args:
        in_cyl (cylinder): Input cylinder
        t (np.array(dtype=np.float64)): Transform to apply the change of coordinates

    Returns:
        cylinder: in_cyl expressed in the new coordinate frame
    """

    c = homogenize(in_cyl.center) @ t.T[:, :3]
    d = in_cyl.direction @ t.T[:3, :3]

    return cylinder(center=c, radius=in_cyl.radius, direction=d, height=in_cyl.height, name=f'{in_cyl.name} (transd)')


def load_branch(f_name):
    """
    Returns a list of cylinders read from a file

    Args:
        f_name (str): File's name containing list of cylinders

    Returns:
        list: Branch of cylinders
    """

    ret = []

    with open(f_name) as f:
        for li in f.readlines():
            ret.append(from_string(li))

    return ret


def dist_to_branch(p, b):
    """
    Find i such that [b[i].center,b[i+1].center] is the closest segment to p.

    Args:
        p (np.array(dtype=np.float64)): The query point
        b (list): A branch, as a list of cylinders

    Returns:
        np.array(dtype=np.float64): Minimum square distance
        int: Index of the closest cylinder with the minimum square distance
    """

    if len(b) == 0:
        return -1

    if len(b) == 1:
        d = np.asarray(p, dtype=float) - b[0].center
        return np.dot(d, d)

    cp = b[0].center
    cn = b[1].center

    d_min = segment(cn, cp).distance_sqr(p)
    closest = 0

    for i, cyl_n in enumerate(b[2:]):
        cp = cn
        cn = cyl_n.center
        d = segment(cn, cp).distance_sqr(p)

        if d < d_min:
            d_min = d

            # Starts at i=0 for cyl_n of index 2
            closest = i + 1

    return d_min, closest


def closest_branch(p, ba):
    """
    Returns the closest branch to point p, within a list of branches ba

    Args:
        p (np.array(dtype=np.float64)): The query point
        ba (list): An array of branches

    Returns:
        np.array(dtype=np.float64): Minimum square distance
        list: Closest branch
        int: Index returned by dist_to_branch
    """

    if len(ba) == 0:
        return []

    d_min, closest = dist_to_branch(p, ba[0])
    bc = ba[0]

    for b in ba[1:]:
        d, cp = dist_to_branch(p, b)

        if d < d_min:
            bc = b
            d_min = d
            closest = cp

    return d_min, bc, closest
