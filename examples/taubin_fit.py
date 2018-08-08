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

from ctapipe.image import toymodel, tailcuts_clean, psf_likelihood_fit, impact_parameter_chisq_fit
from ctapipe.instrument import CameraGeometry
from ctapipe.visualization import CameraDisplay



def taubin_func2(params,x,y,imagemask):
    """
    Taubin fit function to be minimized

    Parameters
    ----------
    params: 3-tuple
        the fit parameters: (radius, center_x, center_y)
    x: array-like
        x coordinates
    y: array-like
        y coordinates
    image: array-like
        cleaned image

    This will usually be x and y coordinates and pe charges of camera pixels
    (We don't care about the latter, we interpret it as 0,1 info)
    """
    r, cx, cy = params
    return (((x-cx)**2+(y-cy)**2-r**2)**2)[imagemask].sum()/(((x-cx)**2+(y-cy)**2)[imagemask].sum())

def draw_circle(parr):
    global anim_i
    r, cx, cy = parr
    par = (cx, cy, r)
    disp.overlay_ringpar(par, color='cyan', linewidth=2)
    plt.show()
    plt.savefig('tmp%04d.png'%anim_i)
    anim_i += 1    

def taubin_fit(x,y,image):
    """
    Do a Taubin fit using 
    Uses the sample mean and std for the initial guess
    Parameters
    ----------
    x: array-like or astropy quantity
        x coordinates of the points
    y: array-like or astropy quantity
        y coordinates of the points
    image: array-like
        weights of the points (0,1)

    This will usually be x and y coordinates and if pixel is on or off

    Returns
    -------
    radius: astropy-quantity
        radius of the ring
    center_x: astropy-quantity
        x coordinate of the ring center
    center_y: astropy-quantity
        y coordinate of the ring center
    """

    x = Quantity(x).decompose()
    y = Quantity(y).decompose()
    assert x.unit == y.unit
    unit = x.unit
    x = x.value
    y = y.value

    #Initial guess: data set centroid and std deviation from mean
    pos = np.empty(x.shape + (2,))
    pos[..., 0] = x
    pos[..., 1] = y
    centroid = np.average(pos,axis=0,weights=image)
    #print(centroid)
    rad = np.average(np.sqrt(image*(x-centroid[0])**2+image*(y-centroid[1])**2))
    #print(rad)
    
    
    start = (rad,centroid[0],centroid[1])
    result = minimize(
        taubin_func2,
        x0=(rad,centroid[0],centroid[1]),
        args=(x, y, (image!=0)),
        method='L-BFGS-B',
        bounds=[
            (0, None),      # radius should be positive
            (None, None),
            (None, None),
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
    xi_cut = 0.003
    m_low = 0 #good muons counter i.e. events with xi < xi_cut
    m_high = 0 #bad muons counter i.e. events with xi > xi_cut
    s_low = 0 #good showers counter i.e. events with xi < xi_cut
    s_high = 0 #bad showers counter i.e. events with xi > xi_cut
    #Classification variable declaration
    xi_mu = []
    xi_shower = []
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
            print("%f %f"%(m_low,s_low))
        
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
        
        """
        # Show the camera image and overlay and clean pixels
        disp.image = image
        disp.cmap = 'PuOr'
        disp.highlight_pixels(cleanmask, color='black')
        plt.show()
        plt.savefig('tmp%04d.png'%anim_i);
        anim_i += 1

        """

        
        #print("I'm fitting...")
        #radius, cx, cy, ssgma = psf_likelihood_fit(pos[...,0],pos[...,1],clean)
    
        #Plot fit result
        #ringpar = (cx, cy, radius)
        #disp.overlay_ringpar(ringpar, color='cyan', linewidth=2)
        """
        imp_par, phi_max = impact_parameter_chisq_fit(pos[...,0],pos[...,1],clean,
                 radius,
                 cx,
                 cy,
                 mirror_radius=0.15,
                 bins=30)
        
        print('Impact parameter = ')
        print(imp_par)
        print('Phi max = ')
        print(phi_max)
        """
    
      
        #Taubin fit
        start_time = time.clock()
        trad, tcx, tcy = taubin_fit(pos[...,0], pos[...,1], clean)
        end_time = time.clock()
        exec_time = end_time - start_time
        time_mu.append(exec_time)
        #print(exec_time)
        xi_min = taubin_func2((trad,tcx,tcy),pos[...,0], pos[...,1], cleanmask)
        xi_mu.append(xi_min)
        #print('Minimal epsilon: %f'%epsilon_min)
        if xi_min < xi_cut:
            m_low += 1
        else:
            m_high += 1
        """
        ringpar = (tcx, tcy, trad)
        disp.overlay_ringpar(ringpar, color='cyan', linewidth=2)
        plt.show()
        plt.savefig('tmp%04d.png'%anim_i);
        anim_i += 1
        """
        """
        savename = "/afs/ifh.de/group/cta/scratch/pillera/Taubin/Pics23_7/Muon_"
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
            
            print("Iteration nr. %d"%l)
       
        """
        disp = CameraDisplay(geom)
        disp.set_limits_minmax(0, 40)
        disp.add_colorbar()
        """

        # Create a fake camera image to display:
        #print(angle[l]+'d')
        
        model = toymodel.generate_2d_shower_model(
            centroid=(cx[l],cy[l]), width=width[l], length=length[l], psi=angle[l]
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
        
        """
        # Show the camera image and overlay and clean pixels
        disp.image = image
        disp.cmap = 'PuOr'
        disp.highlight_pixels(cleanmask, color='black')
        plt.show()
        plt.savefig('tmp%04d.png'%anim_i);
        anim_i += 1

        """
        
        
        #print("I'm fitting...")
        #radius, cx, cy, ssgma = psf_likelihood_fit(pos[...,0],pos[...,1],clean)
    
        #Plot fit result
        #ringpar = (cx, cy, radius)
        #disp.overlay_ringpar(ringpar, color='cyan', linewidth=2)
        """
        imp_par, phi_max = impact_parameter_chisq_fit(pos[...,0],pos[...,1],clean,
                 radius,
                 cx,
                 cy,
                 mirror_radius=0.15,
                 bins=30)
        print('Impact parameter = ')
        print(imp_par)
        print('Phi max = ')
        print(phi_max)
        """
        #Taubin fit
        
        start_time = time.clock()
        trad, tcx, tcy = taubin_fit(pos[...,0], pos[...,1], clean)
        end_time = time.clock()
        exec_time = end_time - start_time
        time_shower.append(exec_time)
        #print(exec_time)
        ringpar = (tcx, tcy, trad)
        xi_min = taubin_func2((trad,tcx,tcy),pos[...,0], pos[...,1], cleanmask)
        xi_shower.append(xi_min)
        #print('Minimal epsilon: %f'%epsilon_min)
        if xi_min < xi_cut:
            s_low += 1
        else:
            s_high += 1
        """
        disp.overlay_ringpar(ringpar, color='cyan', linewidth=2)
        plt.show()
        plt.savefig('tmp%04d.png'%anim_i);
        anim_i += 1
        """
        """
        savename = "/afs/ifh.de/group/cta/scratch/pillera/Taubin/Pics23_7/Shower_"
        savename += str(l)
        savename += ".png"
        plt.savefig(savename)
       """
        plt.clf()
        
        
    #Classification
    fig, ax = plt.subplots()
    ax.hist(xi_shower, bins=50, range=[0,.005],color='red',label='showers',alpha=0.5)
    ax.hist(xi_mu, bins=50, range=[0,.005],color='orange',label='muons',alpha=0.5)
    
    ax.legend()
    ax.set_xlabel('Taubin minimizer')
    ax.set_ylabel('number of events')
    #ax.set_title(r'Histogram of IQ: $\mu=100$, $\sigma=15$')
    
   
    ymin, ymax = plt.ylim()
    cutLine = [(xi_cut,ymin), (xi_cut,ymax)]
    (l1, l2) = zip(*cutLine)
    #ax.add_line(lines.Line2D(l1, l2, linewidth=3, color='blue'))
    fig.savefig("/afs/ifh.de/group/cta/scratch/pillera/Taubin/taubin_minimizer.png")
    
    eff = m_low / (m_low + m_high)
    print("Efficiency = %f %%"%(eff*100))
    pur = m_low / (m_low + s_high)
    print("Purity = %f %%"%(pur*100))
    
    
    #Time
    fig_t, ax_t = plt.subplots()
    ax_t.hist(time_mu, bins=50, range=[0.002,.02],color='orange',label='muons',alpha=0.5)
    ax_t.hist(time_shower, bins=50, range=[0.002,.02],color='red',label='showers',alpha=0.5)
    ax_t.legend()
    ax_t.set_xlabel('time (sec)')
    ax_t.set_ylabel('number of events')
    ax_t.set_title(r'Taubin fit execution time')
    fig_t.savefig("/afs/ifh.de/group/cta/scratch/pillera/Taubin/taubin_exec_time.png")
    mtimeav = np.mean(time_mu)
    stimeav = np.mean(time_shower)
    timeav = np.mean([mtimeav,stimeav])
    freqav = 1./timeav
    print("Average computation time: muons   %f sec"%mtimeav)
    print("                        : showers %f sec"%stimeav)
    print("                        : total   %f sec"%timeav)
    print("Average frequency: %f Hz"%freqav)
    
