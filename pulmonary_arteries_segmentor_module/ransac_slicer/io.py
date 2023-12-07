import numpy as np
import pandas as pd
from datetime import date
import json
import nibabel as ni
import nrrd


def read_nrrd_from_file(f_name):
    """
    Read a volume from nrrd file

    Args:
        f_name (str): Nrrd filepath

    Returns:
        np.array(dtype=np.float64): Volume's data
        dict: File's header
    """

    data, header = nrrd.read(f_name)
    return data, header


def read_nii_from_file(f_name):
    """
    Read a volume from Nifti file

    Args:
        f_name (str): Nifti filepath

    Returns:
        np.array(dtype=np.float64): Volume's data
        np.array(dtype=np.float64): Volume's IJK to RAS transformation
    """

    nv = ni.load(f_name)

    return nv.get_fdata(), nv.affine


def save_nii_to_file(f_name, vol, vox_to_met):
    """
    Save a volume to Nifti file

    Args:
        f_name (str): Nifti filepath
        vol (np.array(dtype=np.float64)): Volume's data
        vox_to_met (np.array(dtype=np.float64)): Volume's IJK to RAS transformation
    """

    ni.save(ni.Nifti1Image(vol, vox_to_met), f_name)


def read_points_from_csv(f_name):
    """
    Read points from CSV file
    Coordinates are stored in columns named 'x', 'y' and 'z'

    Args:
        f_name (str): CSV filepath

    Returns:
        np.array(dtype=np.float64): Read points
    """

    try:
        d = pd.read_csv(f_name, usecols=['x', 'y', 'z'])
        return d.to_numpy()
    except Exception:
        return None


def to_markups(f_name, p):
    """
    Saves a set of points in csv file that can be imported in Slicer as Markups

    Args:
        f_name (str): CSV filepath
        p (np.array(dtype=np.float64)): Points to save
    """

    d = pd.DataFrame(p, columns=['r', 'a', 's'])
    d.insert(0, 'label', [' '] * p.shape[0])
    d.to_csv(f_name)


def from_line_markups(f_name):
    """
    Read markups from file

    Args:
        f_name (str): Markups filepath

    Returns:
        np.array(dtype=np.float64): Markups' center
        np.array(dtype=np.float64): Markups' radius
        np.array(dtype=np.float64): Markups' direction
    """

    with open(f_name) as f:
        d = json.load(f)

    d = d['markups'][0]
    center = np.array(d['controlPoints'][0]['position'])
    e = np.array(d['controlPoints'][1]['position'])

    if d['coordinateSystem'] == 'LPS':
        center *= np.array([-1, -1, 1])
        e *= np.array([-1, -1, 1])

    direction = e - center
    radius = d['display']['glyphSize']

    return center, radius, direction


def init_markups_json(vol, markups_type):
    """
    Initialize JSON object for markups

    Args:
        vol (volume): Input volume
        markups_type (str): Type of markups to add in this object

    Returns:
        dict: Markups data
    """

    markups_json = {}
    markups_json[
        '@schema'] = 'https://raw.githubusercontent.com/slicer/slicer/master/Modules/Loadable/Markups/Resources/' \
                     'Schema/markups-schema-v1.0.3.json#'

    markups_json['markups'] = [{}]

    markups_json['markups'][0]['type'] = markups_type
    markups_json['markups'][0]['coordinateSystem'] = 'RAS'
    markups_json['markups'][0]['coordinateUnits'] = 'mm'
    markups_json['markups'][0]['locked'] = False
    markups_json['markups'][0]['fixedNumberOfControlPoints'] = False
    markups_json['markups'][0]['labelFormat'] = '%N-%d'
    markups_json['markups'][0]['lastUsedControlPointNumber'] = 0

    markups_json['markups'][0]['controlPoints'] = []

    markups_json['markups'][0]['measurements'] = []

    markups_json['markups'][0]['display'] = {}

    markups_json['markups'][0]['display']['visibility'] = True
    markups_json['markups'][0]['display']['opacity'] = 1.0
    markups_json['markups'][0]['display']['color'] = [0.4, 1.0, 0.0]
    markups_json['markups'][0]['display']['selectedColor'] = [1.0, 0.5000076295109483, 0.5000076295109483]
    markups_json['markups'][0]['display']['activeColor'] = [0.4, 1.0, 0.0]
    markups_json['markups'][0]['display']['propertiesLabelVisibility'] = False
    markups_json['markups'][0]['display']['pointLabelsVisibility'] = True
    markups_json['markups'][0]['display']['textScale'] = 3.0
    markups_json['markups'][0]['display']['glyphType'] = 'Sphere3D'
    markups_json['markups'][0]['display']['glyphScale'] = 3.0
    markups_json['markups'][0]['display']['glyphSize'] = 5.0
    markups_json['markups'][0]['display']['useGlyphScale'] = True
    markups_json['markups'][0]['display']['sliceProjection'] = False
    markups_json['markups'][0]['display']['sliceProjectionUseFiducialColor'] = True
    markups_json['markups'][0]['display']['sliceProjectionOutlinedBehindSlicePlane'] = False
    markups_json['markups'][0]['display']['sliceProjectionColor'] = [1.0, 1.0, 1.0]
    markups_json['markups'][0]['display']['sliceProjectionOpacity'] = 0.6
    markups_json['markups'][0]['display']['lineThickness'] = 0.2
    markups_json['markups'][0]['display']['lineColorFadingStart'] = 1.0
    markups_json['markups'][0]['display']['lineColorFadingEnd'] = 10.0
    markups_json['markups'][0]['display']['lineColorFadingSaturation'] = 1.0
    markups_json['markups'][0]['display']['lineColorFadingHueOffset'] = 0.0
    markups_json['markups'][0]['display']['handlesInteractive'] = False
    markups_json['markups'][0]['display']['translationHandleVisibility'] = True
    markups_json['markups'][0]['display']['rotationHandleVisibility'] = True
    markups_json['markups'][0]['display']['scaleHandleVisibility'] = True
    markups_json['markups'][0]['display']['interactionHandleScale'] = 3.0
    markups_json['markups'][0]['display']['snapMode'] = 'toVisibleSurface'

    markups_json['metadata'] = {}

    markups_json['metadata']['name'] = 'Unknown'
    markups_json['metadata']['date'] = date.today().isoformat()
    markups_json['metadata']['dimensions'] = vol.dimensions
    markups_json['metadata']['origin'] = []  # TODO
    markups_json['metadata']['ras_to_ijk'] = vol.ras_to_ijk.tolist()
    markups_json['metadata']['spacing'] = []  # TODO

    return markups_json


def copy_curve(curve_path, vol):
    """
    Read and copy curve from a file

    Args:
        curve_path (str): Curve filepath
        vol (volume): Volume related to this curve

    Returns:
        dict: Copied curve data
    """

    f = open(curve_path)
    input_data = json.load(f)
    f.close()

    output_data = input_data.copy()

    output_data['metadata'] = {}

    output_data['metadata']['name'] = 'Unknown'
    output_data['metadata']['date'] = date.today().isoformat()
    output_data['metadata']['dimensions'] = vol.dimensions
    output_data['metadata']['origin'] = []  # TODO
    output_data['metadata']['ras_to_ijk'] = vol.ras_to_ijk.tolist()
    output_data['metadata']['spacing'] = []  # TODO

    return output_data


def fcsv_to_json(contour_point_path, vol):
    """
    Read contour points from a FCSV file to JSON object

    Args:
        contour_point_path (str): Input contour points filepath (FCSV)
        vol (volume): Input volume

    Returns:
        dict: Points data (JSON)
    """

    f = open(contour_point_path)
    lines = f.read().splitlines()
    f.close()

    contour_point = init_markups_json(vol, 'Fiducial')

    lines = lines[3:]

    for line in lines:
        contour_point['markups'][0]['lastUsedControlPointNumber'] += 1

        data = line.split(',')
        point_dict = {'id': str(contour_point['markups'][0]['lastUsedControlPointNumber']), 'label': data[11],
                      'description': '', 'associatedNodeID': '',
                      'position': [float(data[1]), float(data[2]), float(data[3])],

                      # TODO : Check how to get the true orientation
                      'orientation': [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],

                      'selected': True, 'locked': False, 'visibility': True, 'positionStatus': 'defined', }

        contour_point['markups'][0]['controlPoints'].append(point_dict)

    return contour_point
