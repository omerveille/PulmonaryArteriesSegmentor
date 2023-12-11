#!/usr/bin/env python-real
from .io import (copy_curve)
from .cylinder_ransac import (track_branch, config)
from .cylinder import cylinder, closest_branch
from .volume import volume
import numpy as np
import json


def run_ransac(input_volume_path, input_centers_curve_path, output_centers_curve_path, input_contour_point_path,
         output_contour_point_path, starting_point, direction_point, starting_radius, pct_inlier_points, threshold, isNewBranch):
    # Input volume
    vol = volume.from_nrrd(input_volume_path)

    # JSON file describing the curve
    if input_centers_curve_path != output_centers_curve_path:
        centers_curve = copy_curve(input_centers_curve_path, vol)

        f = open(output_centers_curve_path, 'w')
        json.dump(centers_curve, f, indent=4)
        f.close()

    else:
        f = open(input_centers_curve_path)
        centers_curve = json.load(f)
        f.close()

    centers_curve['markups'][0]['coordinateSystem'] = 'RAS'

    # FCSV file describing contour points
    f = open(input_contour_point_path)
    contour_points = json.load(f)
    f.close()

    contour_points['markups'][0]['display']['opacity'] = 0.4
    contour_points['markups'][0]['display']['pointLabelsVisibility'] = False
    contour_points['markups'][0]['coordinateSystem'] = 'RAS'

    f = open(output_contour_point_path, 'w')
    json.dump(contour_points, f, indent=4)
    f.close()

    # Input info for branch tracking (in RAS coordinates)

    branch = []
    # Add existing cylinders in the branch
    for cp in centers_curve['markups'][0]['controlPoints']:
        branch.append(cylinder(center=np.array(cp['position'])))

    # Input info for branch tracking (in RAS coordinates)
    if isNewBranch:
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
    track_branch(vol, cyl, cfg, centers_curve, output_centers_curve_path, contour_points, output_contour_point_path, branch)
