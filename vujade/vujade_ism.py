"""
Developer: vujadeyoon
Email: vujadeyoon@gmail.com
Github: https://github.com/vujadeyoon/vujade

Title: vujade_imgcv.py
Description: Image Statistics Matching

Acknowledgement:
    1. This implementation is highly inspired from continental.
    2. Github: https://github.com/continental/image-statistics-matching
"""


import sys
import numpy as np
from dataclasses import dataclass
from typing import Tuple
from vujade.vujade_debug import printd, pprintd


@dataclass
class PredefinedValue(object):
    MATCH_ZERO: float = 0.0
    MATCH_FULL: float = 1.0


class FeatureDistributionMatching(object):
    def __init__(self, _channels: tuple = (0,), _channel_ranges: tuple = ((0.0, 1.0),)):
        super().__init__()
        self.channels = _channels
        self.channel_ranges = _channel_ranges

    def apply_mean_cov(self, source: np.ndarray, _ndarr_meanref: np.ndarray, _ndarr_covref: np.ndarray) -> np.ndarray:
        matching_result = self._matching_mean_cov(source[:, :, self.channels], _ndarr_meanref, _ndarr_covref)
        result = np.copy(source)
        # Replace selected channels with matching result
        result[:, :, self.channels] = matching_result

        # Replace selected channels
        for channel in self.channels:
            result[:, :, channel] = np.clip(result[:, :, channel], self.channel_ranges[channel][0],
                                            self.channel_ranges[channel][1])

        return result.astype(np.float32)

    def apply(self, source: np.ndarray, reference: np.ndarray) -> np.ndarray:
        matching_result = self._matching(source[:, :, self.channels], reference[:, :, self.channels])
        result = np.copy(source)
        # Replace selected channels with matching result
        result[:, :, self.channels] = matching_result

        # Replace selected channels
        for channel in self.channels:
            result[:, :, channel] = np.clip(result[:, :, channel], self.channel_ranges[channel].min,
                                            self.channel_ranges[channel].max)

        return result.astype(np.float32)

    @staticmethod
    def _matching_mean_cov(source: np.ndarray, _ndarr_meanref: np.ndarray, _ndarr_covref: np.ndarray) -> np.ndarray:
        """ Run all transformation steps """
        # 1.) reshape to feature matrix (H*W,C)
        feature_mat_src = FeatureDistributionMatching._get_feature_matrix(source)
        # feature_mat_ref = FeatureDistributionMatching._get_feature_matrix(reference)

        # 2.) center (subtract mean)
        feature_mat_src, _ = FeatureDistributionMatching._center_image(feature_mat_src)
        # feature_mat_ref, reference_mean = FeatureDistributionMatching._center_image(feature_mat_ref)

        # 3.) whitening: cov(feature_mat_src) = I
        feature_mat_src_white = FeatureDistributionMatching._whitening(feature_mat_src)

        # 4.) transform covariance: cov(feature_mat_ref) = covariance_ref
        feature_mat_src_transformed = FeatureDistributionMatching._covariance_transformation_from_covref(
            feature_mat_src_white, _ndarr_covref)

        # 5.) Add reference mean
        feature_mat_src_transformed += _ndarr_meanref

        # 6.) Reshape
        result = feature_mat_src_transformed.reshape(source.shape)

        return result

    @staticmethod
    def _matching(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """ Run all transformation steps """
        # 1.) reshape to feature matrix (H*W,C)
        feature_mat_src = FeatureDistributionMatching._get_feature_matrix(source)
        feature_mat_ref = FeatureDistributionMatching._get_feature_matrix(reference)

        # 2.) center (subtract mean)
        feature_mat_src, _ = FeatureDistributionMatching._center_image(feature_mat_src)
        feature_mat_ref, reference_mean = FeatureDistributionMatching._center_image(feature_mat_ref)

        # 3.) whitening: cov(feature_mat_src) = I
        feature_mat_src_white = FeatureDistributionMatching._whitening(feature_mat_src)

        # 4.) transform covariance: cov(feature_mat_ref) = covariance_ref
        feature_mat_src_transformed = FeatureDistributionMatching._covariance_transformation(feature_mat_src_white,
                                                                                             feature_mat_ref)

        # 5.) Add reference mean
        feature_mat_src_transformed += reference_mean

        # 6.) Reshape
        result = feature_mat_src_transformed.reshape(source.shape)

        return result

    @staticmethod
    def _get_feature_matrix(image: np.ndarray) -> np.ndarray:
        """ Reshapes an image (H, W, C) to
        a feature vector (H * W, C)
        :param image: H x W x C image
        :return feature_matrix: N x C matrix with N samples and C features
        """
        feature_matrix = np.reshape(image, (-1, image.shape[-1]))
        return feature_matrix

    @staticmethod
    def _center_image(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """ Centers the image by removing mean
        :returns centered image and original mean
        """
        image = np.copy(image)
        image_mean = np.mean(image, axis=0)
        image -= image_mean
        return image, image_mean

    @staticmethod
    def _whitening(feature_mat: np.ndarray) -> np.ndarray:
        """
        Transform the feature matrix so that cov(feature_map) = Identity or
        if the feature matrix is one dimensional so that var(feature_map) = 1.
        :param feature_mat: N x C matrix with N samples and C features
        :return feature_mat_white: A corresponding feature vector with an
        identity covariance matrix or variance of 1.
        """
        if feature_mat.shape[1] == 1:
            variance = np.var(feature_mat)
            feature_mat_white = feature_mat / np.sqrt(variance)
        else:
            data_cov = np.cov(feature_mat, rowvar=False)
            u_mat, s_vec, _ = np.linalg.svd(data_cov)
            sqrt_s = np.diag(np.sqrt(s_vec))
            feature_mat_white = (feature_mat @ u_mat) @ np.linalg.inv(sqrt_s)
        return feature_mat_white

    @staticmethod
    def _covariance_transformation_from_covref(feature_mat_white: np.ndarray, covariance_ref: np.ndarray) -> np.ndarray:
        if feature_mat_white.shape[1] == 1:
            raise NotImplementedError()
        else:
            u_mat, s_vec, _ = np.linalg.svd(covariance_ref)
            sqrt_s = np.diag(np.sqrt(s_vec))

            feature_mat_transformed = (feature_mat_white @ sqrt_s) @ u_mat.T
        return feature_mat_transformed

    @staticmethod
    def _covariance_transformation(feature_mat_white: np.ndarray, feature_mat_ref: np.ndarray) -> np.ndarray:
        """
        Transform the white (cov=Identity) feature matrix so that
        cov(feature_mat_transformed) = cov(feature_mat_ref). In the 2d case
        this becomes:
        var(feature_mat_transformed) = var(feature_mat_ref)
        :param feature_mat_white: input with identity covariance matrix
        :param feature_mat_ref: reference feature matrix
        :return: feature_mat_transformed with cov == cov(feature_mat_ref)
        """
        if feature_mat_white.shape[1] == 1:
            variance_ref = np.var(feature_mat_ref)
            feature_mat_transformed = feature_mat_white * np.sqrt(variance_ref)
        else:
            covariance_ref = np.cov(feature_mat_ref, rowvar=False)
            u_mat, s_vec, _ = np.linalg.svd(covariance_ref)
            sqrt_s = np.diag(np.sqrt(s_vec))

            feature_mat_transformed = (feature_mat_white @ sqrt_s) @ u_mat.T
        return feature_mat_transformed


class HistogramMatching(object):
    """Histogram Matching operation class"""

    def __init__(self, _channels: tuple = (0,), _match_prop: float = PredefinedValue.MATCH_FULL):
        super().__init__()
        self.channels = _channels
        self.match_prop = float(_match_prop)

    @property
    def match_prop(self) -> float:
        """ Returns the matching proportion value """
        return self._match_prop

    @match_prop.setter
    def match_prop(self, matching_proportion: float) -> None:
        if PredefinedValue.MATCH_ZERO <= matching_proportion <= PredefinedValue.MATCH_FULL:
            self._match_prop = matching_proportion
        else:
            raise ValueError(
                f'matching proportion has to be in range [{PredefinedValue.MATCH_ZERO}, {PredefinedValue.MATCH_FULL}], the given value is {matching_proportion}')

    def apply(self, source: np.ndarray, reference: np.ndarray) -> np.ndarray:
        result = np.copy(source)
        for channel in self.channels:
            result[:, :, channel] = self._match_channel(source[:, :, channel], reference[:, :, channel])
        return result.astype(np.float32)

    def _match_channel(self, source: np.ndarray, reference: np.ndarray) -> np.ndarray:
        if self.match_prop == PredefinedValue.MATCH_ZERO:
            return source

        source_shape = source.shape
        source = source.ravel()
        reference = reference.ravel()

        # get unique pixel values (sorted),
        # indices of the unique array and counts
        _, s_indices, s_counts = np.unique(source, return_counts=True, return_inverse=True)
        r_values, r_counts = np.unique(reference, return_counts=True)

        # compute the cumulative sum of the counts
        s_quantiles = np.cumsum(s_counts).astype(float) / (source.size + sys.float_info.epsilon)
        r_quantiles = np.cumsum(r_counts).astype(float) / (reference.size + sys.float_info.epsilon)

        # interpolate linearly to find the pixel values in the reference
        # that correspond most closely to the quantiles in the source image
        interp_values = np.interp(s_quantiles, r_quantiles, r_values)

        # pick the interpolated pixel values using the inverted source indices
        result = interp_values[s_indices]

        # apply matching proportion
        if self.match_prop < PredefinedValue.MATCH_FULL:
            diff = source.astype(float) - result
            result = source.astype(float) - (diff * self.match_prop)

        return result.reshape(source_shape)
