#!/usr/bin/env python-real
from .io import (copy_curve)
from .cylinder_ransac import (track_branch, config)
from .cylinder import cylinder
from .volume import volume
import json


def run_ransac(input_volume_path, input_centers_curve_path, output_centers_curve_path, input_contour_point_path,
         output_contour_point_path, starting_point, direction_point, starting_radius, pct_inlier_points, threshold):
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

    # FCSV file describing contour points
    f = open(input_contour_point_path)
    contour_points = json.load(f)
    f.close()

    contour_points['markups'][0]['display']['opacity'] = 0.4
    contour_points['markups'][0]['display']['pointLabelsVisibility'] = False

    f = open(output_contour_point_path, 'w')
    json.dump(contour_points, f, indent=4)
    f.close()

    # Input info for branch tracking (in RAS coordinates)
    direction_point = direction_point - starting_point

    init_radius = starting_radius

    # Tracking configuration
    pct_inl = pct_inlier_points / 100
    err = threshold / 100
    cfg = config(percent_inliers=pct_inl, threshold=err)

    # Initialize tracking
    cyl = cylinder(starting_point, init_radius, direction_point, height=0)

    # Perform tracking
    track_branch(vol, cyl, cfg, centers_curve, output_centers_curve_path, contour_points, output_contour_point_path)
