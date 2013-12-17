import numpy as np
import scipy.linalg as sla
from scipy.io import loadmat

import cv2

import warnings

class IntrinsicErrors(object):
    """Just stores the uncertainty (standard deviation)"""
    def __init__(self, focal_length_error, principal_point_error,
                 skew_error, distortion_error):
        self.focal_length = focal_length_error
        self.principal_point = principal_point_error
        self.skew = skew_error
        self.distortion = distortion_error

    def __repr__(self):
        return "%s(%r)" % (self.__class__, self.__dict__)

class ExtrinsicErrors(object):
    """Store the uncertainty for extrinsics"""
    def __init__(self, trans_error, rot_error):
        self.trans = trans_error
        self.rot = rot_error

class Intrinsics(object):
    """Encompasses everything we need to know about the internal workings
    of a camera."""
    def __init__(self, matrix, distortion,
                 focal_length_error, principal_point_error, skew_error, distortion_error,
                 image_width, image_height, num_calibration_images):
        self.matrix = matrix
        self.distortion = distortion
        if distortion is None:
            self.distortion = np.zeros(5)  # default - no distortion correct

        self.errors = IntrinsicErrors(focal_length_error, principal_point_error,
                                      skew_error, distortion_error)

        self.image_width = image_width
        self.image_height = image_height
        self.num_calibration_images = num_calibration_images

    def __repr__(self):
        return "%s(%r)" % (self.__class__, self.__dict__)

    def inv_matrix(self):
        return sla.inv(self.matrix)

    def zero_out_distortion(self):
        """For testing purposes, remove distortion"""
        self.actual_distortion = self.distortion
        self.distortion[:] = 0

    def produce_smaller_intrinsics(self, cur_w, cur_h, new_w, new_h):
        """Downsize an intrinsic matrix (urk). Pretty sure this makes
        the distortion not work"""

        warnings.warn('this function is most definitely not tested! use at your own risk!')

        w_scale = float(new_w) / cur_w
        h_scale = float(new_h) / cur_h
        # avg_scale = (w_scale + h_scale) / 2.

        m = self.matrix

        w2 = m[0, 2]
        h2 = m[1, 2]

        x_fact = m[0, 0] / w2
        y_fact = m[1, 1] / h2

        new_w2 = w2 * w_scale
        new_h2 = h2 * h_scale

        new_matrix = np.eye(3)
        new_matrix[0, 2] = new_w2
        new_matrix[1, 2] = new_h2

        new_matrix[0, 0] = x_fact * new_w2
        new_matrix[1, 1] = y_fact * new_h2

        return Intrinsics(matrix=new_matrix)


class Extrinsics(object):
    """Stores the extrinsics computed from each checkerboard image,
    i.e. the extrinsics of the camera as related to the checkerboard
    corner"""
    def __init__(self, extrinsics_dict, errors_dict, indices):
        self.extrinsics = extrinsics_dict
        self.errors = errors_dict
        self.indices = indices

    def __getitem__(self, index):
        """Overload to make access to extrinsic matrices easier"""
        return self.extrinsics[index]

class CalibrationResults(object):
    def __init__(self, intrinsics, extrinsics):
        self.intrinsics = intrinsics
        self.extrinsics = extrinsics

    @classmethod
    def from_file(cls, calib_results_mat_fname):
        """Load the calib_results.mat file into a useful
        python object"""
        # Maps from the matlab variable names to our variables names
        _intrinsic_mapping_dict = {
            'KK': 'matrix',
            'kc': 'distortion',
            'fc_error': 'focal_length_error',
            'cc_error': 'principal_point_error',
            'alpha_c_error': 'skew_error',
            'kc_error': 'distortion_error',
            'nx': 'image_width',
            'ny': 'image_height',
            'n_ima': 'num_calibration_images',
        }

        mat_contents = loadmat(calib_results_mat_fname, squeeze_me=True)

        # first things to get: intrinsics stuff
        intrinsics_dict = {}
        for matlab_name, python_name in _intrinsic_mapping_dict.items():
            intrinsics_dict[python_name] = mat_contents[matlab_name]

        intrinsics = Intrinsics(**intrinsics_dict)

        # Now we need to get the extrinsics for each calibration
        # image. Note that they don't all necessarily exist - if we provide
        # [image_001.bmp, image_010.bmp, image_100.bmp] to the calibration
        # procedure it will set n_ima to 100 instead of 3!! In this case the 
        # extrinsics in the file will be stored as NaN, so we don't bother loading
        # them.
        extrinsics_dict = {}
        extrinsics_errors_dict = {}
        valid_indices = []
        for i in range(1, mat_contents['n_ima']+1):
            trans_vec = mat_contents['Tc_%d' % i]
            if not np.all(np.isfinite(trans_vec)):
                # this image must not have existed
                continue

            rot_vec = mat_contents['omc_%d' % i]
            trans_error = mat_contents['Tc_error_%d' % i]
            rot_error = mat_contents['omc_error_%d' % i]

            # Need to make the extrinsic matrix. how do we convert this axis
            # angle to rotation? Using rodrigues.
            rot_mtx, _ = cv2.Rodrigues(rot_vec)
            extrinsic_matrix = np.vstack([np.hstack([rot_mtx, trans_vec.reshape(3,1)]),
                                          [0, 0, 0, 1]])

            extrinsic_errors = ExtrinsicErrors(trans_error, rot_error)

            # Need i-1 here because of matlab 1 indexing - if the first image is
            # image000.bmp matlab still calls it image 1
            extrinsics_dict[i - 1] = extrinsic_matrix
            extrinsics_errors_dict[i - 1] = extrinsic_errors
            valid_indices.append(i - 1)

        # Fix the number of images
        intrinsics.num_calibration_images = len(valid_indices)

        all_ext = Extrinsics(extrinsics_dict, extrinsics_errors_dict, valid_indices)

        return cls(intrinsics, all_ext)

def load_calib_results(fname):
    return CalibrationResults.from_file(fname)

