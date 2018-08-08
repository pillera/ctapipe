# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Utilities to generate toymodel (fake) reconstruction inputs for testing 
purposes.

Example:

.. code-block:: python

    >>> from instrument import CameraGeometry
    >>> geom = CameraGeometry.make_rectangular(20,20)
    >>> showermodel = generate_2d_shower_model(centroid=[0.25, 0.0], 
    length=0.1,width=0.02, psi='40d')
    >>> image, signal, noise = make_toymodel_shower_image(geom, showermodel.pdf)
    >>> print(image.shape)
    (400,)
                                             
.. plot:: image/image_example.py
    :include-source:



"""
import numpy as np
from scipy.stats import multivariate_normal
from ctapipe.utils import linalg

__all__ = [
    'generate_2d_shower_model',
    'make_toymodel_shower_image',
]


def generate_2d_shower_model(centroid, width, length, psi):
    """Create a statistical model (2D gaussian) for a shower image in a
    camera. The model's PDF (`model.pdf`) can be passed to
    `make_toymodel_shower_image`.

    Parameters
    ----------
    centroid : (float,float)
        position of the centroid of the shower in camera coordinates
    width : float
        width of shower (minor axis)
    length : float
        length of shower (major axis)
    psi : convertable to `astropy.coordinates.Angle`
        rotation angle about the centroid (0=x-axis)

    Returns
    -------

    a `scipy.stats` object

    """
    aligned_covariance = np.array([[length, 0], [0, width]])
    # rotate by psi angle: C' = R C R+
    rotation = linalg.rotation_matrix_2d(psi)
    rotated_covariance = rotation.dot(aligned_covariance).dot(rotation.T)
    return multivariate_normal(mean=centroid, cov=rotated_covariance)


def make_toymodel_shower_image(geom, showerpdf, intensity=50, nsb_level_pe=50):
    """Generates a pedestal-subtracted shower image from a statistical
    shower model (as generated by `shower_model`). The resulting image
    will be in the same format as the given
    `~ctapipe.image.camera.CameraGeometry`.

    Parameters
    ----------
    geom : `ctapipe.instrument.CameraGeometry`
        camera geometry object 
    showerpdf : func
        PDF function for the shower to generate in the camera, e.g. from a 
    intensity : int
        factor to multiply the model by to get photo-electrons
    nsb_level_pe : type
        level of NSB/pedestal in photo-electrons

    Returns
    -------

    an array of image intensities corresponding to the given `CameraGeometry`

    """
    pos = np.empty(geom.pix_x.shape + (2,))
    pos[..., 0] = geom.pix_x.value
    pos[..., 1] = geom.pix_y.value
    

    model_counts = (showerpdf(pos) * intensity).astype(np.int32)
    signal = np.random.poisson(model_counts)
    noise = np.random.poisson(nsb_level_pe, size=signal.shape)
    image = (signal + noise) - np.mean(noise)

    return image, signal, noise


def gaussian(x, mean, sigma):
    return np.exp(-(x - mean)**2. / (2. * sigma * sigma))


def generate_muon_model(xy, radius, width, centre_x, centre_y):
    r_pix = np.sqrt((xy[..., 0] - centre_x)**2. + (xy[..., 1] - centre_y)**2.)
    Im_pix = gaussian(r_pix, radius, width)
    return Im_pix

def make_toymodel_shower_image_muons(geom, my_par, intensity=50, nsb_level_pe=50):
    """Generates a pedestal-subtracted shower image from a statistical
    shower model (as generated by `shower_model`). The resulting image
    will be in the same format as the given
    `~ctapipe.image.camera.CameraGeometry`.

    Parameters
    ----------
    geom : `ctapipe.instrument.CameraGeometry`
        camera geometry object 
    my_par : parameters of the muon ring
        radius, width, centre_x, centre_y
    intensity : int
        factor to multiply the model by to get photo-electrons
    nsb_level_pe : type
        level of NSB/pedestal in photo-electrons

    Returns
    -------

    an array of image intensities corresponding to the given `CameraGeometry`

    """
    pos = np.empty(geom.pix_x.shape + (2,))
    pos[..., 0] = geom.pix_x.value
    pos[..., 1] = geom.pix_y.value
    

    model_counts = (5*generate_muon_model(pos, my_par[0], my_par[1], my_par[2], my_par[3]) * intensity).astype(np.int32)
    signal = np.random.poisson(model_counts)
    noise = np.random.poisson(nsb_level_pe, size=signal.shape)
    image = (signal + noise) - np.mean(noise)

    return image, signal, noise

def phi_func(xy,r_pix,centre_y):
    n = xy.shape[0]
    phi = np.empty(n)
    for i in range(n):
        if xy[i,1] > centre_y:
            phi[i] = np.arccos((xy[i,1]-centre_y)/r_pix[i])
        else: 
            phi[i] = np.arccos(-1*(xy[i,1]-centre_y)/r_pix[i]) + np.pi
    return phi

def generate_muon_model2(xy,radius,wchange,wmean,centre_x,centre_y):
    """
    wchange is the maximal gaussian width variation along the ring
    wmean is the mean value of the gaussian width 
    """
    r_pix = np.sqrt((xy[..., 0] - centre_x)**2. + (xy[..., 1] - centre_y)**2.)
    phi = phi_func(xy,r_pix,centre_y)
    width = wchange*np.sin(phi - np.pi/2)+wmean
    Im_pix = gaussian(r_pix, radius, width)
    return Im_pix

def make_toymodel_shower_image_muons2(geom, param, intensity=50, nsb_level_pe=50):
    """Generates a pedestal-subtracted shower image from a statistical
    shower model (as generated by `shower_model`). The resulting image
    will be in the same format as the given
    `~ctapipe.image.camera.CameraGeometry`.

    Parameters
    ----------
    geom : `ctapipe.instrument.CameraGeometry`
        camera geometry object 
    param : parameters of the muon ring
        radius, wchange, wmean, centre_x, centre_y
    intensity : int
        factor to multiply the model by to get photo-electrons
    nsb_level_pe : type
        level of NSB/pedestal in photo-electrons

    Returns
    -------

    an array of image intensities corresponding to the given `CameraGeometry`

    """
    pos = np.empty(geom.pix_x.shape + (2,))
    pos[..., 0] = geom.pix_x.value
    pos[..., 1] = geom.pix_y.value
    
    radius, wchange, wmean, centre_x, centre_y = param
    model_counts = (4*generate_muon_model2(pos,radius, wchange, wmean, centre_x, centre_y) * intensity).astype(np.int32)
    signal = np.random.poisson(model_counts)
    noise = np.random.poisson(nsb_level_pe, size=signal.shape)
    image = (signal + noise) - np.mean(noise)

    return image, signal, noise