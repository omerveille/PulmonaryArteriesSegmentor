import numpy as np
import math


class segment:
    """
    Class to represent a segment
    """

    def __init__(self, start=[0, 0, 0], end=[0, 0, 0]):
        """
        Generate a new segment instance.
        The default is set in 3D but all functions work in N-D

        Args:
            start (list, optional): Segment start position. Defaults to [0, 0, 0].
            end (list, optional): Segment end position. Defaults to [0, 0, 0].
        """

        self.start = np.asarray(start, dtype=np.float64)
        self.end = np.asarray(end, dtype=np.float64)

    def __repr__(self):
        """
        Represent a segment object as a string

        Returns:
            str: Segment object representation
        """

        return f"Segment ::\n\tStart: {self.start} -- End: {self.end}"

    def as_vector(self):
        """
        Compute the segment's vector

        Returns:
            np.array(dtype=np.float64): Segment's vector
        """

        return self.end - self.start

    def len_sqr(self):
        """
        Compute the squared segment length

        Returns:
            np.array(dtype=np.float64): Squared segment length
        """

        d = self.as_vector()

        return np.dot(d, d)

    def distance_sqr(self, p):
        """
        Squared distance to a point p.
        Correctly handles a zero-length segment (point-to-point distance)

        Args:
            p (np.array(dtype=np.float64)): Point to compute distance

        Returns:
            np.array(dtype=np.float64): Squared distance
        """

        if len(p.shape) == 1:
            pa = p.reshape((1, 3))
        else:
            pa = p

        d1 = pa - self.start
        v1 = np.sum(d1 * d1, axis=-1)

        li = self.as_vector()
        v2 = li @ li

        if math.isclose(v2, 0):
            return v1

        v3 = d1 @ li / v2
        v3[v3 <= 0] = 0
        v3[v3 >= 1] = 1
        d2 = v1 - v3 * v3 * v2

        return d2[0] if len(p.shape) == 1 else d2
