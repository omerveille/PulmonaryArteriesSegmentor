import numpy as np
import math


from . import cylinder, helper


class config:
    """
    Configuration for RANSAC algorithm
    """

    _min_radius = 0.05
    _max_radius = 5

    def __init__(self, nb_test_min=0, nb_test_max=1000, percent_inliers=0.5, threshold=0.1, radius_min=0.5,
                 radius_max=1.5, angle_max=np.pi / 3, nb_cyl_dirs=81, nb_ray_dirs=162, n_samples=128, ray_length=2,
                 nb_iter=1000):
        """
        Initialize algorithm's configuration

        Args:
            nb_test_min (int, optional): Min number of RANSAC tests. Clamped to [0,+infinity). Defaults to 0.
            nb_test_max (int, optional): Max number of RANSAC tests. Clamped to [0,+infinity). Defaults to 1000.
            percent_inliers (float, optional): Percentage of inliers to accept a candidate. Clamped to [0,1].
                                               Defaults to 0.5.
            threshold (float, optional): Proportion of the previous radius to take as distance threshold to elect
                                         inliers. Absolute value taken, so that it is positive. Defaults to 0.1.
            radius_min (float, optional): Proportion of the previous radius to take as min values of acceptable
                                          radius. Clamped to [_min_radius,_max_radius]. Defaults to 0.5.
            radius_max (float, optional): Proportion of the previous radius to take as max values of acceptable
                                          radius. Clamped to [_min_radius,_max_radius]. Defaults to 1.5.
            angle_max (float, optional): Maximum angle. Clamped to [0,Pi/2]. Defaults to np.pi/2.
            nb_cyl_dirs (int, optional): Number of directions to consider for the cylinder axis in RANSAC (clamped
                                         to 21 as a minimum, see helper.sample_half_gauss_sphere). Defaults to 81.
            nb_ray_dirs (int, optional): Number of directions to consider to cast rays from the vessel center
                                         (clamped to 42 as a minimum, see helper.sample_gauss_sphere).
                                         Defaults to 162.
            n_samples (int, optional): Number of samples to extract on each ray cast. Defaults to 128.
            ray_length (int, optional): Length of a cast ray, as a proportion of the previous radius. Defaults to 3.
            nb_iter (int, optional): Number of iterations for the algorithm. Defaults to 1000.
        """

        self.nb_test_min = nb_test_min
        self.nb_test_max = nb_test_max
        self.pct_inl = percent_inliers
        self.threshold = threshold
        self.r_min = radius_min
        self.r_max = radius_max
        self.a_max = angle_max
        self.cyl_dir_set = nb_cyl_dirs
        self.ray_dir_set = nb_ray_dirs
        self.n_samples = n_samples
        self.ray_len = ray_length
        self.nb_iter = nb_iter

        # To be done: proportion of the previous height to advance to get the new center
        self.advance_ratio = 0.5

    @property
    def nb_test_min(self):
        """
        Getter for min number of RANSAC tests

        Returns:
            int: Min number of RANSAC tests
        """

        return self._nb_test_min

    @nb_test_min.setter
    def nb_test_min(self, n):
        """
        Set nb_test_min to n
        Clamped to [0,+infinity).
        Also set nb_test_max to n if nb_test_max < n, to ensure that
        nb_test_min <= nb_test_max

        Args:
            n (int): New min number of RANSAC tests
        """

        self._nb_test_min = n if n > 0 else 0

        if not hasattr(self, '_nb_test_max') or n > self._nb_test_max:
            self._nb_test_max = n

    @property
    def nb_test_max(self):
        """
        Getter for max number of RANSAC tests

        Returns:
            int: Max number of RANSAC tests
        """

        return self._nb_test_max

    @nb_test_max.setter
    def nb_test_max(self, n):
        """
        Set nb_test_max to n. Clamped to [0,+infinity)
        Also set nb_test_min to n if nb_test_min > n, to ensure that nb_test_min <= nb_test_max

        Args:
            n (int): New max number of RANSAC tests
        """

        self._nb_test_max = n if n > 0 else 0

        if not hasattr(self, '_nb_test_min') or n < self._nb_test_min:
            self._nb_test_min = n

    @property
    def pct_inl(self):
        """
        Getter for percentage of inliers to accept a candidate.

        Returns:
            float: Percentage of inliers to accept a candidate
        """

        return self._pct_inl

    @pct_inl.setter
    def pct_inl(self, p):
        """
        Set pct_inl to p, the minimum percentage of inliers to select a candidate cylinder. Clamped to [0,1]

        Args:
            p (float): New percentage of inliers to accept a candidate
        """

        if p < 0:
            p = 0
        elif p > 1:
            p = 1

        self._pct_inl = p

    @property
    def threshold(self):
        """
        Getter of proportion of the previous radius to take as distance threshold to elect inliers

        Returns:
            float: Proportion of the previous radius to take as distance threshold to elect inliers
        """

        return self._threshold

    @threshold.setter
    def threshold(self, t):
        """
        Sets threshold to t, the largest distance to the cylinder for a point to be selected as inlier, expressed as a
        proportion of the current radius.
        Take the absolute value

        Args:
            t (float): New proportion of the previous radius to take as distance threshold to elect inliers
        """

        self._threshold = math.fabs(t)

    @property
    def r_min(self):
        """
        Getter of proportion of the previous radius to take as min values of acceptable radius

        Returns:
            float: Proportion of the previous radius to take as min values of acceptable radius
        """

        return self._r_min

    @r_min.setter
    def r_min(self, r):
        """
        Set r_min to r, the proportion of the smallest acceptable candidate radius to the current one.
        Clamped to [_min_radius,_max_radius]
        Also set r_max to r if r_max < r

        Args:
            r (float): Proportion of the smallest acceptable candidate radius
        """

        if r < config._min_radius:
            r = config._min_radius
        elif r > config._max_radius:
            r = config._max_radius

        self._r_min = r

        if not hasattr(self, '_r_max') or self.r_max < r:
            self.r_max = r

    @property
    def r_max(self):
        """
        Getter of proportion of the previous radius to take as max values of acceptable radius

        Returns:
            float: Proportion of the previous radius to take as max values of acceptable radius
        """

        return self._r_max

    @r_max.setter
    def r_max(self, r):
        """
        Set r_max to r, the proportion of the largest acceptable candidate radius to the current one
        Clamped to [_min_radius,_max_radius]
        Also set r_min to r if r_min > r, and set ray_len to r if ray_len < r

        Args:
            r (float): Proportion of the largest acceptable candidate radius to the current one
        """

        if r < config._min_radius:
            r = config._min_radius
        elif r > config._max_radius:
            r = config._max_radius

        self._r_max = r

        if not hasattr(self, '_r_min') or self.r_min > r:
            self.r_min = r

        if not hasattr(self, '_ray_len') or self.ray_len < r:
            self.ray_len = r

    @property
    def a_max(self):
        """
        Getter of maximum angle

        Returns:
            float: Maximum angle
        """

        return self._a_max

    @a_max.setter
    def a_max(self, a):
        """
        Set a_max to a, the maximum angle between the candidate direction and the current one
        Clamped to [0,Pi/2].
        Setting a_max to anything larger that pi consists in accepting any direction. Therefore, a_max is set to pi
        in such cases

        Args:
            a (float): New maximum angle
        """

        if a < 0:
            a = 0
        elif a > np.pi / 2:
            a = np.pi / 2

        self._a_max = a

    @property
    def cyl_dir_set(self):
        """
        Getter of directions to consider for the cylinder axis in RANSAC

        Returns:
            np.array(dtype=np.float64): Directions to consider for the cylinder axis in RANSAC
        """

        return self._cyl_dirs

    @cyl_dir_set.setter
    def cyl_dir_set(self, nb_cyl_dirs):
        """
        Setter of directions to consider for the cylinder axis in RANSAC

        Args:
            nb_cyl_dirs (int): Number of directions to consider for the cylinder axis in RANSAC
        """

        # Only update _cyl_dirs if necessary
        if not hasattr(self, '_cyl_dirs') or self._cyl_dirs.shape[0] < nb_cyl_dirs:
            # Regular sampling of the half Gaussian sphere
            self._cyl_dirs = helper.sample_half_gauss_sphere(nb_cyl_dirs)

    @property
    def nb_cyl_dirs(self):
        """
        Getter of number of directions to consider for the cylinder axis in RANSAC

        Returns:
            int: Number of directions to consider for the cylinder axis in RANSAC
        """

        return self.cyl_dir_set.shape[0]

    @property
    def ray_dir_set(self):
        """
        Getter of directions to consider to cast rays from the vessel center

        Returns:
            np.array(dtype=np.float64): Directions to consider to cast rays from the vessel center
        """

        return self._ray_dirs

    @ray_dir_set.setter
    def ray_dir_set(self, nb_ray_dirs):
        """
        Setter of directions to consider to cast rays from the vessel center

        Args:
            nb_ray_dirs (int): Number of directions to consider to cast rays from the vessel center
        """

        # Only update _ray_dirs if necessary
        if not hasattr(self, '_ray_dirs') or self._ray_dirs.shape[0] < nb_ray_dirs:
            # Regular sampling of the half Gaussian sphere
            self._ray_dirs = helper.sample_gauss_sphere(nb_ray_dirs)

    @property
    def nb_ray_dirs(self):
        """
        Getter of number of directions to consider to cast rays from the vessel center

        Returns:
            int: Number of directions to consider to cast rays from the vessel center
        """

        return self.ray_dir_set.shape[0]

    @property
    def n_samples(self):
        """
        Getter of number of samples to extract on each ray cast

        Returns:
            int: Number of samples to extract on each ray cast
        """

        return self._n_samples

    @n_samples.setter
    def n_samples(self, ns):
        """
        Setter of number of samples to extract on each ray cast

        Args:
            ns (int): New number of samples to extract on each ray cast
        """

        if ns <= 0:
            self._n_samples = 1
        else:
            self._n_samples = ns

    @property
    def ray_len(self):
        """
        Getter of length of a cast ray

        Returns:
            int: Length of a cast ray
        """

        return self._ray_len

    @ray_len.setter
    def ray_len(self, rl):
        """
        Sets the length of a cast ray, as a proportion of the current radius estimate
        Must be larger than self.r_max.
        If not, then self.ray_len is set to self.r_max, discarding rl

        Args:
            rl (int): The value to set. If below r_max, it is discarded and ray_len is set to r_max
        """

        if rl <= self.r_max:
            self._ray_len = self.r_max
        else:
            self._ray_len = rl

    @property
    def nb_iter(self):
        """
        Getter of number of iterations for the algorithm

        Returns:
            int: Number of iterations for the algorithm
        """

        return self._nb_iter

    @nb_iter.setter
    def nb_iter(self, n):
        """
        Setter of number of iterations for the algorithm

        Args:
            n (int): New number of iterations for the algorithm
        """

        self._nb_iter = n if n > 0 else 0


def fit_cylinder_ransac(p, axis, nb_test_min, nb_test_max, pct_inl, r_min, r_max, err):
    """
    Fits a cylinder to a set of points using RANSAC, given the direction for the cylinder's axis
    The percentage of inliers might be below pct_inl if nb_test_max is reached.
    In that case, the cylinder with the best percentage of inliers is returned.

    Args:
        p (np.array(dtype=np.float64)): Input set of points
        axis (np.array(dtype=np.float64)): Cylinder's axis
        nb_test_min (int): Min number of RANSAC tests
        nb_test_max (int): Max number of RANSAC tests
        pct_inl (float): Minimum allowable percentage of inliers
        r_min (float): Min radius allowed for returned cylinder
        r_max (float): Max radius allowed for returned cylinder
        err (float): Maximum allowable distance to cylinder for an inlier point

    Returns:
        cylinder: Fitted cylinder
        np.array(dtype=np.float64): Fitted cylinder's inlier set
    """

    max_cyl = cylinder.cylinder(direction=axis)
    max_p_inl = -1
    max_inliers = np.empty((0, 3))

    if p.shape[0] < 3:
        return max_cyl, max_inliers, 0

    # Do this at least nb_test_min times
    for _ in range(nb_test_min):

        # Pick 3 points at random
        q = p[np.random.choice(p.shape[0], 3, replace=False)]

        # Fit cylinder
        c = cylinder.fit_3_points(q[0], q[1], q[2], axis)

        if c is not None and r_min < c.radius < r_max:
            # Compute inliers
            inliers = c.select_inliers(p, err)

            # Percentage of inliers
            p_inl = inliers.shape[0] / p.shape[0]

            # Update max_cyl to current one if we found a better percentage of inliers
            if p_inl > max_p_inl:
                max_cyl = c
                max_inliers = inliers
                max_p_inl = p_inl

    # Now go up to nb_test_max tries, but return as soon as a correct cylinder has been found
    # (ie percentage of inliers is sufficient)
    for _ in range(nb_test_min, nb_test_max):

        # Return if percentage of inliers is sufficient
        if max_p_inl >= pct_inl:
            return max_cyl, max_inliers, max_p_inl

        # No adequate cylinder has been found, try with one more
        q = p[np.random.choice(p.shape[0], 3, replace=False)]
        c = cylinder.fit_3_points(q[0], q[1], q[2], axis)

        if c is not None and r_min < c.radius < r_max:
            inliers = c.select_inliers(p, err)

            # Percentage of inliers
            p_inl = inliers.shape[0] / p.shape[0]

            if p_inl > max_p_inl:
                max_cyl = c
                max_inliers = inliers
                max_p_inl = p_inl

    # Here: no cylinder was found after nb_test_max tries, with an adequate percentage of inliers. Return the best
    # cylinder that was found. In that case, the percentage of inliers could be below pct_inl
    return max_cyl, max_inliers, max_p_inl


def sample(vol, center, radius, n_samples, dirs):
    """
    Cast rays in a volume from center along each direction in dirs with length radius.
    Retrieves the point of minimum directional gradient along each ray.

    Args:
        vol (volume): Input volume
        center (np.array(dtype=np.float64)): A 3-vector, start of each ray
        radius (float): The length of each cast ray
        n_samples (int): Number of samples to extract along each ray
        dirs (np.array(dtype=np.float64)): The directions to cast rays towards, as a Nx3 array. Each direction has
                                           to be normalized.

    Returns:
        np.array(dtype=np.float64): One point along each ray, all assembled in a Nx3 array.
    """

    p = np.empty((dirs.shape[0], 3))

    for i, d in enumerate(dirs):
        start = center
        end = center + d * radius
        interpolated_coords, c = vol.get_line(start, end, n_samples)

        # Exclude first and last points whose gradient is invalid
        k = np.argmin(helper.gradient_central_dif(interpolated_coords)[1:-1]) + 1
        p[i] = c[k]

    return p


def filter_points(p, center, r_min, r_max):
    """
    Filters points in p to only keep those that at least at r_min distance from center and at most r_max distance from
    center (r_min and r_max excluded)

    Args:
        p (np.array(dtype=np.float64)): Points to filter, as a Nx3 array
        center (np.array(dtype=np.float64)): Reference point to compute distances from
        r_min (float): Only points with ]r_min,r_max[ distance to center are kept
        r_max (float): Only points with ]r_min,r_max[ distance to center are kept

    Returns:
        np.array(dtype=np.float64): Points that are kept
    """

    d = np.linalg.norm(p - center, axis=1)
    i = np.where(np.logical_and(d > r_min, d < r_max))[0]

    return p[i]


def next_cylinder(vol, cyl, cfg):
    """
    Compute next cylinder

    Args:
        vol (volume): Input volume
        cyl (cylinder): Current cylinder
        cfg (config): Tracking configuration

    Returns:
        cylinder: Next cylinder
        np.array(dtype=np.float64): Next cylinder's inlier point set
    """

    # Make sure 3rd order interpolation is used
    order = vol.order
    vol.order = 3

    # We try with original cfg.advance_ration, and if it fails, try again with 2*cfg.advance_ratio (typical case when
    # the artery is highly curved)
    for li in [1, 2]:

        # Compute guess for next cylinder center: advance from current one by cfg.advance_ratio times its height along
        # direction li is here in case the original advance_ratio (0.5 by default) was not large enough to ensure
        # sufficient progress in the tracking
        next_center = cyl.center + li * cfg.advance_ratio * cyl.height * cyl.direction

        # Sets some parameters from cfg
        r_min = cyl.radius * cfg.r_min
        r_max = cyl.radius * cfg.r_max
        ray_len = cyl.radius * cfg.ray_len
        err_threshold = cyl.radius * cfg.threshold

        # Extract points. Remove (filter out) points at both extremities because the 3rd order interpolation might
        # provide tainted gradient values
        p = sample(vol, next_center, ray_len, cfg.n_samples, cfg.ray_dir_set)
        p = filter_points(p, next_center, 0.1 * ray_len, 0.9 * ray_len)

        # Stop if less than 3 points remain after filtering
        if p.shape[0] < 3:
            return None, None

        # Sort cylinder directions starting with the one most aligned with cyl.direction
        idx = np.argsort(-np.abs(cyl.direction @ cfg.cyl_dir_set.T))

        # Look for best cylinder:
        #   - review axes in idx-based order
        #   - as soon as a cylinder is found with more than cfg.pct_inl inlier rate, keep it and stop searching -
        #     otherwise list cylinders with an inlier rate larger than cfg.pct_inl/2:
        #       the cylinder with the best inlier rate will be kept, if any
        p_max = 0
        c_max = cylinder.cylinder()
        i_max = np.empty((0, 3))
        # print("cyl direction:", cyl.direction)

        for axis in cfg.cyl_dir_set[idx]:
            if np.abs(cyl.direction @ axis) < np.cos(cfg.a_max):
                continue
            # print("axis", axis, "calcul:", cyl.direction @ axis)
            c, inliers, p_inl = fit_cylinder_ransac(p, axis, cfg.nb_test_min, cfg.nb_test_max, cfg.pct_inl, r_min,
                                                    r_max, err_threshold)

            # p_inl = inliers.shape[0]/p.shape[0]

            if p_inl > cfg.pct_inl:
                p_max = p_inl
                c_max = c
                i_max = inliers
                break

            # Also list cylinder with inlier rate > cfg.pct_inl/2
            elif p_inl > cfg.pct_inl / 2 and p_max < p_inl:
                # Update cyl with max inliers
                p_max = p_inl
                c_max = c
                i_max = inliers

        # Need at least 3 points to fit a cylinder.
        # Else: stop, no valid cylinder was found
        if i_max.shape[0] < 3:
            return None, None

        # Now refine the cylinder that was found
        r_min = c_max.radius * cfg.r_min
        r_max = c_max.radius * cfg.r_max
        ray_len = c_max.radius * cfg.ray_len
        err_threshold = c_max.radius * cfg.threshold

        # Update c_max, with same direction, but recompute points and other parameters
        # Update center at a median position along axis wrt inliers.
        c_max.fix_center(i_max)

        # Re-extract and filter points from this new center
        p = sample(vol, c_max.center, ray_len, cfg.n_samples, cfg.ray_dir_set)
        p = filter_points(p, c_max.center, 0.1 * ray_len, 0.9 * ray_len)

        # Select the inliers
        i_max = c_max.select_inliers(p, err_threshold)

        # Need at least 3 points to fit a cylinder. Else: stop
        if i_max.shape[0] < 3:
            return None, None

        # Re-update center wrt new inliers
        c_max.fix_center(i_max)

        # Compute height and keep inscribed inlier points
        i_max = c_max.fix_height(i_max)

        # Refine cylinder parameters, in particular to smooth its direction that was originally picked with a discrete
        # set
        c_max.refine(i_max)

        # Update inlier set according to new cylinder
        i_max = c_max.select_inliers(p, err_threshold)

        # Need at least 3 points to fit a cylinder. Else: stop
        if i_max.shape[0] < 3:
            return None, None

        # Re-update center wrt inliers
        c_max.fix_center(i_max)

        # Fixes cylinder direction so that it points in the same direction as the tracking advance necessary for next
        # step (computation of next guess)
        if c_max.direction @ cyl.direction < 0:
            c_max.direction = -c_max.direction

        # Fixes height, and select inliers with this height
        i_max = c_max.fix_height(i_max)

        # Test if a sufficient advance was made.
        # If not, we might try with a double advance_ratio (initial loop over li)
        if c_max.radius > 4:
            c_max.height = c_max.radius
        dist_centers = np.linalg.norm(c_max.center - cyl.center)
        if dist_centers >= cyl.height / 2 and (cyl.height == 0 or dist_centers <= cyl.height * 2):
            # Restore interpolation order
            vol.order = order
            # print("found cyl with direction:", c_max.direction, "and angle:",cyl.direction @ c_max.direction)
            return c_max, i_max

    # No new cylinder was found
    vol.order = order

    return None, None


def track_cylinder(vol, cyl, cfg):
    """
    Generator that tracks cylinders in a volume, starting with an initial guess

    Args:
        vol (volume): Input volume
        cyl (cylinder): Initial cylinder guess
        cfg (config): Configuration for the tracking

    Yields:
        cylinder: Next cylinder
        np.array(dtype=np.float64): Next cylinder's inlier set
    """

    for _ in range(cfg.nb_iter):
        c_max, i_max = next_cylinder(vol, cyl, cfg)

        if c_max is not None:
            yield c_max, i_max
            cyl = c_max.copy()
        else:
            break


def track_branch(vol, cyl, cfg, centers_line, center_line_radius, contour_points, branch, progress_dialog):
    """
    Performs the tracking in a volume, given an input cylinder and a configuration

    Args:
        vol (volume): Input volume
        cyl (cylinder): Input cylinder
        cfg (config): Configuration for the tracking
        centers_curve (np.ndarray): Centers curve's data
        contour_point (np.ndarray): Contour points' data
    """

    centerline_cpt = 0
    contour_points_cpt = 0

    for _, (_cylinder, current_contour_points) in enumerate(track_cylinder(vol, cyl, cfg)):
        # Criteria for acceptance: Need to be better justified especially third one
        #   1- Valid cylinder (i.shape[0] > 0)
        #   2- Sufficient advance: |c.center-cyl.center| > cyl.height/4
        #   3- Not redundant: d(c,branch) > cyl.radius/10
        # (Note: what if cyl.height/4 < cyl.radius/10?)

        if current_contour_points.shape[0] > 0 and not _cylinder.is_redundant(branch):
            branch.append(_cylinder)

            centers_line = np.vstack((centers_line, _cylinder.center))
            contour_points.append(current_contour_points.tolist())

            radius = np.linalg.norm(current_contour_points - _cylinder.center, axis=1).min()
            center_line_radius.append(radius)

            centerline_cpt += 1
            contour_points_cpt += len(current_contour_points)

            progress_dialog.setText(f"Centerline points found: {centerline_cpt}\nContour points found: {contour_points_cpt}")

        else:
            break

    return centers_line, contour_points, center_line_radius
