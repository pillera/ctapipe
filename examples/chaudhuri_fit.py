"""
Example of drawing a Camera using a toymodel shower image.
Taubin fit of the ring
"""

import matplotlib.pylab as plt
import matplotlib.lines as lines
import numpy as np
import time

from scipy.optimize import minimize
from astropy.units import Quantity

from ctapipe.image import toymodel, tailcuts_clean
from ctapipe.instrument import CameraGeometry
from ctapipe.visualization import CameraDisplay

def kundu_chaudhuri_circle_fit(x, y, weights):
    """
    Fast, analytic calculation of circle center and radius for 
    weighted data using method given in [chaudhuri93]_

    Parameters
    ----------
    x: array-like or astropy quantity
        x coordinates of the points
    y: array-like or astropy quantity
        y coordinates of the points
    weights: array-like
        weights of the points

    """

    weights_sum = np.sum(weights)
    mean_x = np.sum(x * weights) / weights_sum
    mean_y = np.sum(y * weights) / weights_sum

    a1 = np.sum(weights * (x - mean_x) * x)
    a2 = np.sum(weights * (y - mean_y) * x)

    b1 = np.sum(weights * (x - mean_x) * y)
    b2 = np.sum(weights * (y - mean_y) * y)

    c1 = 0.5 * np.sum(weights * (x - mean_x) * (x**2 + y**2))
    c2 = 0.5 * np.sum(weights * (y - mean_y) * (x**2 + y**2))

    center_x = (b2 * c1 - b1 * c2) / (a1 * b2 - a2 * b1)
    center_y = (a2 * c1 - a1 * c2) / (a2 * b1 - a1 * b2)

    radius = np.sqrt(np.sum(
        weights * ((center_x - x)**2 + (center_y - y)**2),
    ) / weights_sum)

    return radius, center_x, center_y


def _psf_neg_log_likelihood(params, x, y, weights):
    """  
    Negative log-likelihood for a gaussian ring profile

    Parameters
    ----------
    params: 4-tuple
        the fit parameters: (radius, center_x, center_y, std)
    x: array-like
        x coordinates
    y: array-like
        y coordinates
    weights: array-like
        weights for the (x, y) points

    This will usually be x and y coordinates and pe charges of camera pixels
    """
    radius, center_x, center_y, sigma = params
    pixel_distance = np.sqrt((center_x - x)**2 + (center_y - y)**2)
    
    ret = np.sum(
        (np.log(sigma) + 0.5 * ((pixel_distance - radius) / sigma)**2) * weights
    )
    return ret

def draw_circle(parr):
    global anim_i
    r, cx, cy, s = parr
    par = (cx, cy, r)
    disp.overlay_ringpar(par, color='cyan', linewidth=2)
    plt.show()
    plt.savefig('tmp%04d.png'%anim_i)
    anim_i += 1
    #time.sleep(0.5)

def psf_likelihood_fit(x, y, weights):
    """
    Do a likelihood fit using a ring with gaussian profile.
    Uses the kundu_chaudhuri_circle_fit for the initial guess



    Parameters
    ----------
    x: array-like or astropy quantity
        x coordinates of the points
    y: array-like or astropy quantity
        y coordinates of the points
    weights: array-like
        weights of the points

    This will usually be x and y coordinates and pe charges of camera pixels

    Returns
    -------
    radius: astropy-quantity
        radius of the ring
    center_x: astropy-quantity
        x coordinate of the ring center
    center_y: astropy-quantity
        y coordinate of the ring center
    std: astropy-quantity
        standard deviation of the gaussian profile (indicator for the ring width)
    """

    x = Quantity(x).decompose()
    y = Quantity(y).decompose()
    assert x.unit == y.unit
    unit = x.unit
    x = x.value
    y = y.value
   
    start_r, start_x, start_y = kundu_chaudhuri_circle_fit(x, y, weights)

    result = minimize(
        _psf_neg_log_likelihood,
        x0=(start_r, start_x, start_y, 5e-3),
        args=(x, y, (weights!=0)),
        method='L-BFGS-B',
        bounds=[
            (0, None),      # radius should be positive
            (None, None),
            (None, None),
            (1e-5, None),      # std should be positive
        ],
        #callback=draw_circle
        
    )

    if not result.success:
        result.x = np.full_like(result.x, np.nan)

    return result.x * unit
    

if __name__ == '__main__': 
    
    anim_i = 0

    np.random.seed(0)
    nIter = 1000
    
    #Cut
    logl_cut = -1000
    m_low = 0 #good muons counter i.e. events with xi < xi_cut
    m_high = 0 #bad muons counter i.e. events with xi > xi_cut
    s_low = 0 #good showers counter i.e. events with xi < xi_cut
    s_high = 0 #bad showers counter i.e. events with xi > xi_cut
    #Classification variable declaration
    logl_mu = []
    logl_shower = []
    time_mu = []
    time_shower = []
    
    #Randomize parameters
    mr = (.15 -.005)*np.random.random(nIter) + .05
    mwchange = (.005 -.0005)*np.random.random(nIter) + .0005
    mwmean = (.007 -.003)*np.random.random(nIter) + .003
    mcx = .3*np.random.random(nIter) - .15
    mcy = .3*np.random.random(nIter) - .15
    
    # Load the camera
    geom = CameraGeometry.from_name("CHEC")
    pos = np.empty(geom.pix_x.shape + (2,))
    pos[..., 0] = geom.pix_x.value
    pos[..., 1] = geom.pix_y.value
    
    print("Muons...")
    for l in range(nIter):
        
        if l%100 == 0:
            print("Iteration nr.%d"%l)
            #print("%f %f"%(m_low,s_low))
        
        
        """
        disp = CameraDisplay(geom)
        disp.set_limits_minmax(0, 40)
        disp.add_colorbar()
        """

        # Create a fake camera image to display:       
        params = [mr[l],mwchange[l],mwmean[l],mcx[l],mcy[l]]
    
        image, sig, bg = toymodel.make_toymodel_shower_image_muons2(
            geom, params, intensity=7, nsb_level_pe=2
        )
    
        # Apply image cleaning
        cleanmask = tailcuts_clean(
            geom, image, picture_thresh=2, boundary_thresh=5
            )
        clean = image.copy()
        clean[~cleanmask] = 0.0 #clean image: not useful pixels are put = 0
        clean = cleanmask.astype(int)
        
        """
        # Show the camera image and overlay and clean pixels
        disp.image = image
        disp.cmap = 'PuOr'
        disp.highlight_pixels(cleanmask, color='black')
        plt.show()
        plt.savefig('tmp%04d.png'%anim_i);
        anim_i += 1
        """
        start_time = time.clock()
        radius, ccx, ccy, ssgma = psf_likelihood_fit(pos[...,0],pos[...,1],clean)
        end_time = time.clock()
        exec_time = end_time - start_time
        time_mu.append(exec_time)
        
        ringpar = (radius, ccx, ccy, ssgma)
        logl = _psf_neg_log_likelihood(ringpar, pos[...,0], pos[...,1], clean)
        #print(logl)
        logl_mu.append(logl)
        #print('Minimal epsilon: %f'%epsilon_min)
        if logl < logl_cut:
            m_low += 1
        else:
            m_high += 1
        
        
        """
        #Plot fit result
        par = (ccx, ccy, radius)
        disp.overlay_ringpar(par, color='cyan', linewidth=2)
        plt.savefig('tmp%04d.png'%anim_i);
        anim_i += 1
        """
        """
        savename = "/afs/ifh.de/group/cta/scratch/pillera/Taubin/Chaudhuri/Muon_"
        savename += str(l)
        savename += ".png"
        plt.savefig(savename)
        """
        plt.clf()
        
   
   
    #Air showers
    #centroid = .3*np.random.random([20,2]) - .15
    cx = .3*np.random.random(nIter) - .15
    cy = .3*np.random.random(nIter) - .15
    width = (.001-.0001)*np.random.random(nIter) + .0001
    length = (.01-.001)*np.random.random(nIter) + .001
    angle = np.random.randint(0,360,size=nIter)
    #angle = str(angle)
    #print(angle)
    angle = [str(s) + 'd' for s in angle]
    
    
    
    print("Showers...")
    
    for l in range(nIter):
        
        if l%100 == 0:
            #print("%f %f"%(m_low,s_low))
            print("Iteration nr. %d"%l)
       
        """
        disp = CameraDisplay(geom)
        disp.set_limits_minmax(0, 40)
        disp.add_colorbar()
        """

        # Create a fake camera image to display:
        
        model = toymodel.generate_2d_shower_model(
            centroid=(cx[l],cy[l]), 
            width=width[l], 
            length=length[l], 
            psi=angle[l]
        )
        
        image, sig, bg = toymodel.make_toymodel_shower_image(
            geom, model.pdf, intensity=30./model.pdf((cx[l],cy[l])), nsb_level_pe=2
        )
        
        # Apply image cleaning
        cleanmask = tailcuts_clean(
            geom, image, picture_thresh=2, boundary_thresh=5
        )
        
        clean = image.copy()
        clean[~cleanmask] = 0.0 #clean image: not useful pixels are put = 0
        clean = cleanmask.astype(int)
        
        """
        # Show the camera image and overlay and clean pixels
        disp.image = image
        disp.cmap = 'PuOr'
        disp.highlight_pixels(cleanmask, color='black')
        plt.show()
        plt.savefig('tmp%04d.png'%anim_i);
        anim_i += 1
        """
        
        start_time = time.clock()
        radius, ccx, ccy, ssgma = psf_likelihood_fit(pos[...,0],pos[...,1],clean)
        end_time = time.clock()
        exec_time = end_time - start_time
        
        time_shower.append(exec_time)
        
        ringpar = (radius, ccx, ccy, ssgma)
        logl = _psf_neg_log_likelihood(ringpar, pos[...,0], pos[...,1], clean)
        #print(logl)
        logl_shower.append(logl)
        #print('Minimal epsilon: %f'%epsilon_min)
        if logl < logl_cut:
            s_low += 1
        else:
            s_high += 1
        """
        #Plot fit result
        par = (ccx, ccy, radius)
        disp.overlay_ringpar(par, color='cyan', linewidth=2)
        plt.savefig('tmp%04d.png'%anim_i);
        anim_i += 1
        """
        """
        savename = "/afs/ifh.de/group/cta/scratch/pillera/Taubin/Chaudhuri/Shower_"
        savename += str(l)
        savename += ".png"
        plt.savefig(savename)
       """
        plt.clf()
        
        
    #Classification
    fig, ax = plt.subplots()
    ax.hist(logl_shower, bins=50, range=[-2000,0],color='red',label='showers',alpha=0.5)
    ax.hist(logl_mu, bins=50, range=[-2000,0],color='orange',label='muons',alpha=0.5)
    
    ax.legend()
    ax.set_xlabel('Chaudhuri minimizer')
    ax.set_ylabel('number of events')
    #ax.set_title(r'Histogram of IQ: $\mu=100$, $\sigma=15$')
    
    
    ymin, ymax = plt.ylim()
    cutLine = [(logl_cut,ymin), (logl_cut,ymax)]
    (l1, l2) = zip(*cutLine)
    #ax.add_line(lines.Line2D(l1, l2, linewidth=3, color='blue'))
    fig.savefig("/afs/ifh.de/group/cta/scratch/pillera/Taubin/Chaudhuri_minimizer.png")
    
    eff = m_low / (m_low + m_high)
    print("Efficiency = %f %%"%(eff*100))
    pur = m_low / (m_low + s_high)
    print("Purity = %f %%"%(pur*100))
    
    
    #Time
    fig_t, ax_t = plt.subplots()
    ax_t.hist(time_mu, bins=50, range=[0.002,.05],color='orange',label='muons',alpha=0.5)
    ax_t.hist(time_shower, bins=50, range=[0.002,.05],color='red',label='showers',alpha=0.5)
    ax_t.legend()
    ax_t.set_xlabel('time (sec)')
    ax_t.set_ylabel('number of events')
    ax_t.set_title(r'Chaudhuri fit execution time')
    fig_t.savefig("/afs/ifh.de/group/cta/scratch/pillera/Taubin/Chaudhuri_exec_time.png")
    mtimeav = np.mean(time_mu)
    stimeav = np.mean(time_shower)
    timeav = np.mean([mtimeav,stimeav])
    freqav = 1./timeav
    print("Average computation time: muons   %f sec"%mtimeav)
    print("                        : showers %f sec"%stimeav)
    print("                        : total   %f sec"%timeav)
    print("Average frequency: %f Hz"%freqav)
    
