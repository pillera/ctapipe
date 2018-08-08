#!/usr/bin/env python3
"""
Example of drawing a Camera using a toymodel shower image.
"""

import matplotlib.pylab as plt
from matplotlib.pyplot import text
from numpy import empty

from ctapipe.image import toymodel, hillas_parameters, tailcuts_clean, psf_likelihood_fit
from ctapipe.instrument import CameraGeometry
from ctapipe.visualization import CameraDisplay


def draw_neighbors(geom, pixel_index, color='r', **kwargs):
    """Draw lines between a pixel and its neighbors"""

    neigh = geom.neighbors[pixel_index]  # neighbor indices (not pixel ids)
    x, y = geom.pix_x[pixel_index].value, geom.pix_y[pixel_index].value
    for nn in neigh:
        nx, ny = geom.pix_x[nn].value, geom.pix_y[nn].value
        plt.plot([x, nx], [y, ny], color=color, **kwargs)


if __name__ == '__main__':

    # Load the camera
#    geom = CameraGeometry.from_name("LSTCam")
    geom = CameraGeometry.from_name("CHEC")
    disp = CameraDisplay(geom)
    #disp.set_limits_minmax(0, 300)
    disp.add_colorbar()


    # Create a fake camera image to display:
    
    
    #My toy muon model 
    mr = 0.07
    mw = 0.005
    mcx = 0.0
    mcy = 0.0
    my_par = [mr,mw,mcx,mcy]
    """
    image, sig, bg = toymodel.make_toymodel_shower_image_muons(
        geom, my_par, intensity=50, nsb_level_pe=1000
    )
    """
    """
    #My toy muon model 2
    mr = 0.08
    mwchange = 0.004
    mwmean = 0.005
    mcx = 0.0
    mcy = 0.0
    my_par = [mr,mwchange,mwmean,mcx,mcy]
    
    image, sig, bg = toymodel.make_toymodel_shower_image_muons2(
        geom, my_par, intensity=50, nsb_level_pe=1000
    )
    """
    model = toymodel.generate_2d_shower_model(
        centroid=(0.02, 0.0), width=0.001, length=0.001, psi='35d'
    )
    image, sig, bg = toymodel.make_toymodel_shower_image(
            geom, model.pdf, intensity=30./model.pdf((0.02, 0.0)), nsb_level_pe=2
        )
    
    # Apply image cleaning
    cleanmask = tailcuts_clean(
        geom, image, picture_thresh=2, boundary_thresh=5
    )
    clean = image.copy()
    clean[~cleanmask] = 0.0
    
    

    
    # Show the camera image and overlay and clean pixels
    disp.image = image
    disp.cmap = 'PuOr'
    disp.highlight_pixels(cleanmask, color='black')


    pos = empty(geom.pix_x.shape + (2,))
    pos[..., 0] = geom.pix_x.value
    pos[..., 1] = geom.pix_y.value

    print("I'm fitting...")
    radius, cx, cy, ssgma = psf_likelihood_fit(pos[...,0],pos[...,1],image)
    print(radius)
    print(cx)
    print(cy)
    print(ssgma)
    """
    # Draw the neighbors of pixel 100 in red, and the neighbor-neighbors in
    # green
    for ii in geom.neighbors[130]:
        draw_neighbors(geom, ii, color='green')

    draw_neighbors(geom, 130, color='cyan', lw=2)
    """
    plt.show()
    
    
     