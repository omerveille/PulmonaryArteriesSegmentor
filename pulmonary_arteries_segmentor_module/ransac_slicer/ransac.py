#!/usr/bin/env python-real
from .io import (copy_curve, fcsv_to_json)
from .cylinder_ransac import (track_branch, config)
from .cylinder import cylinder
from .volume import volume
import sys
import json
import numpy as np


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
    contour_points = fcsv_to_json(input_contour_point_path, vol)

    contour_points['markups'][0]['display']['opacity'] = 0.4
    contour_points['markups'][0]['display']['pointLabelsVisibility'] = False

    f = open(output_contour_point_path, 'w')
    json.dump(contour_points, f, indent=4)
    f.close()

    # Input info for branch tracking (in RAS coordinates)
    init_center = np.array(list(map(float, starting_point.split(','))))

    init_direction = np.array(list(map(float, direction_point.split(','))))
    init_direction = init_direction - init_center

    init_radius = starting_radius

    # Tracking configuration
    pct_inl = pct_inlier_points / 100
    err = threshold / 100
    cfg = config(percent_inliers=pct_inl, threshold=err)

    # Initialize tracking
    cyl = cylinder(init_center, init_radius, init_direction, height=0)

    # Perform tracking
    track_branch(vol, cyl, cfg, centers_curve, output_centers_curve_path, contour_points, output_contour_point_path)


if __name__ == '__main__':

    if len(sys.argv) < 13:
        sys.exit('Usage: ransac --starting_point starting_point --direction_point direction_point input_volume '
                 'input_centers_curve output_centers_curve input_contour_points output_contour_points starting_radius '
                 'pct_inlier_points threshold')

    run_ransac(sys.argv[5], sys.argv[6], sys.argv[7], sys.argv[8], sys.argv[9], sys.argv[2], sys.argv[4], float(sys.argv[10]),
         float(sys.argv[11]), float(sys.argv[12]))
