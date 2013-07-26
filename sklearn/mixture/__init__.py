"""
The :mod:`sklearn.mixture` module implements mixture modeling algorithms.
"""

from .gmm import sample_gaussian, log_multivariate_normal_density
from .gmm import GMM, distribute_covar_matrix_to_match_covariance_type
from .gmm import _validate_covars
from .dpgmm import DPGMM, VBGMM
from .dpmfmm import DPMFMM, sample_vmf_3d, pdf_vmf_3d
from .vbmfmm import VBMFMM

__all__ = ['GMM',
           'DPGMM',
           'VBGMM',
           'DPMFMM',
           'VBMFMM',
           '_validate_covars',
           'distribute_covar_matrix_to_match_covariance_type',
           'log_multivariate_normal_density',
           'sample_gaussian',
           'sample_vmf_3d',
           'pdf_vmf_3d']
