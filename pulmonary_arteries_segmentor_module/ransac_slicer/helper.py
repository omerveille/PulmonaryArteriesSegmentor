import numpy as np
import trimesh.primitives as tp


def _nv_to_geo_level(nv):
    """
    Returns the number of subdivisions necessary to go from an icosahedron to a regular polytope with at least nv
    vertices. Returns 0 is nv <= 12, (in particular if nv <= 0)

    Args:
        nv (int): Number of vertices

    Returns:
        int: Number of subdivisions necessary to go from an icosahedron to a regular polytope with at least nv vertices
    """

    if nv <= 12:
        return 0

    return int(np.ceil(np.log2((nv - 2) / 10) / 2))


def sample_gauss_sphere(n_vertices, center=[0, 0, 0], radius=1):
    """
    Sample the Gaussian sphere on at least n_vertices regularly spaced vertices. In other words, provides a set of
    regularly sampled directions. Works by recursive subdivision of an icosahedron (see _nv_to_geo_level to compute the
    number of subdivisions).
    Thereafter, the returned array may not have exactly n_vertices vertices, but will have at least this number, with a
    minimum of 42 (esp. if n_vertices is <= 0).
    Returns a Nx3 array where N is the actual number of vertices.

    Args:
        n_vertices (int): Number of vertices
        center (list, optional): Sphere's center. Defaults to [0, 0, 0].
        radius (int, optional): Sphere's radius. Defaults to 1.

    Raises:
        ValueError: radius <= 0

    Returns:
        np.array(dtype=np.float64): Sphere's vertices
    """

    if radius <= 0:
        raise ValueError

    m = tp.Sphere(radius=1, center=(0, 0, 0), subdivisions=_nv_to_geo_level(n_vertices))

    return np.asarray(m.vertices)


def sample_half_gauss_sphere(n_vertices):
    """
    Sample the Gaussian half sphere on at least n_vertices regularly spaced vertices. In other words, provides a set of
    regularly sampled orientations, regardless of the direction. Works by recursive subdivision of an icosahedron
    (see _nv_to_geo_level to compute the number of subdivisions).
    Thereafter, the returned array may not have exactly n_vertices vertices, but will have at least this number, with a
    minimum of 21 (esp. if n_vertices is <=0).

    Args:
        n_vertices (int): Number of vertices

    Returns:
        np.array(dtype=np.float64): Sphere's vertices
    """

    v = sample_gauss_sphere(2 * n_vertices)

    m = v @ v.T

    i, j = np.where(np.isclose(m, -1))
    k = [a for a, b in zip(i, j) if a < b]

    return v[k]


def gradient_central_dif(a):
    """
    _summary_

    Args:
        a (_type_): _description_

    Returns:
        _type_: _description_
    """

    g = np.zeros(a.shape)
    g[1:-1] = 0.5 * (a[2:] - a[:-2])

    return g


def homogenize(p):
    """
    _summary_

    Args:
        p (_type_): _description_

    Returns:
        _type_: _description_
    """

    if len(p.shape) == 1:
        return np.append(p, 1)

    return np.hstack((p, np.ones((p.shape[0], 1))))


def cross(a, b):
    """
    Custom cross product to counter poor performance of numpy cross for single vectors

    Args:
        a (np.array(dtype=np.float64)): 3-arrays
        b (np.array(dtype=np.float64)): 3-arrays

    Returns:
        np.array(dtype=np.float64): 3-arrays result
    """

    return np.array(
        [
            a[1] * b[2] - b[1] * a[2],
            b[0] * a[2] - a[0] * b[2],
            a[0] * b[1] - b[0] * a[1],
        ]
    )
