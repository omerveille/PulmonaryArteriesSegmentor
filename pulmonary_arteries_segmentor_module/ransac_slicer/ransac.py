#!/usr/bin/env python-real
from .io import (copy_curve)
from .cylinder_ransac import (track_branch, config)
from .cylinder import cylinder, closest_branch
from .volume import volume
import numpy as np
import json


def run_ransac(input_volume_path, centers_line, contour_points, starting_point, direction_point, starting_radius, pct_inlier_points, threshold, isNewBranch):
    # Input volume
    vol = volume.from_nrrd(input_volume_path)

    # for i in range(len(centers_line)):
    #     centers_line[i]= (np.array(centers_line[i]) * np.array([-1, -1, 1])).tolist()

    # for i in range(len(contour_points)):
    #     contour_points[i]= (np.array(contour_points[i]) * np.array([-1, -1, 1])).tolist()

    branch = []

    # Input info for branch tracking (in RAS coordinates)
    if isNewBranch:
        # Add existing cylinders in the branch
        for cp in centers_line:
            branch.append(cylinder(center=np.array(cp)))

        _, cb, idx_cyl = closest_branch(direction_point, [branch])
        branch = cb[:idx_cyl]
        starting_point = cb[idx_cyl].center
    direction_point = direction_point - starting_point

    init_radius = starting_radius

    # Tracking configuration
    pct_inl = pct_inlier_points / 100
    err = threshold / 100
    cfg = config(percent_inliers=pct_inl, threshold=err)

    # Initialize tracking
    cyl = cylinder(starting_point, init_radius, direction_point, height=0)

    # Perform tracking
    return track_branch(vol, cyl, cfg, centers_line, contour_points, branch)
