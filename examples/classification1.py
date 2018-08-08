import matplotlib.pylab as plt
import matplotlib.lines as lines
import numpy as np
import time
from numpy import arange
from scipy.optimize import minimize
from astropy.units import Quantity

from ctapipe.image import toymodel, tailcuts_clean, psf_likelihood_fit, impact_parameter_chisq_fit
from ctapipe.instrument import CameraGeometry
from ctapipe.visualization import CameraDisplay

from sklearn.neural_network import MLPClassifier
from sklearn import svm

if __name__ == '__main__': 

    
    
    np.random.seed(0)
    nIter = 10000
    
    npix = 2048
    
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
    
    X = [] #sample
    y = [] #target
    X_test = []
    y_test = []
    
    print("Muons...")
    for l in range(nIter):
        if l%100 == 0:
            print("Iteration nr.%d"%l)
            
        
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
        data = cleanmask.astype(int)
        if l < float(nIter)/2.:
            X.append(data)
            y.append(0)
        else:
            X_test.append(data)
            y_test.append(0)
        
        """
        # Show the camera image and overlay and clean pixels
        disp.image = image
        disp.cmap = 'PuOr'
        disp.highlight_pixels(cleanmask, color='black')
        plt.show()
        plt.savefig('tmp%04d.png'%anim_i);
        anim_i += 1

        """
        """ Machine learning part"""
        
      
        
        
       
        
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
        #clean = image.copy()
        #clean[~cleanmask] = 0.0 #clean image: not useful pixels are put = 0
        
        data = cleanmask.astype(int)
        if l < float(nIter)/2.:
            X.append(data)
            y.append(1)
        else:
            X_test.append(data)
            y_test.append(1)
        
     
     
     
     
    print("________________________________________________________")
    print("MLP")
    #classification
    clf = MLPClassifier()
    clf.fit(X,y)
    start_time = time.clock()
    ypr = clf.predict(X_test)
    end_time = time.clock()
    exec_time = end_time - start_time
    print("MLP execution time for %d samples: %f sec"%((nIter/2.),exec_time))
    print("MLP frequency: %f Hz"%(float(ypr.shape[0])/exec_time))
    #for y in ypr:  print(y),
    #print(clf._label_binarizer)
    
    true_m = 0
    false_m = 0
    true_s = 0
    false_s = 0
    for i in range(ypr.shape[0]):
        if ypr[i] == y[i]:
            #wellclassified event
            if y[i] == 0:#muon
                true_m += 1
            else:#shower
                true_s += 1
        else: 
            #misclassified event
            if ypr[i] == 0:
                false_m += 1
            else:
                false_s += 1
    
    print("Total misclassifications: %f%%"%((false_s+false_m)/100.))
    print("Efficiency: %f%%"%(true_m/(float(true_m)+float(false_s))*100.))
    print("Purity: %f%%"%(true_m/(float(true_m)+float(false_m))*100.))
    
    print("________________________________________________________")
    print("SVM")
    #classification
    clf = svm.SVC()
    clf.fit(X,y)
    start_time = time.clock()
    ypr = clf.predict(X_test)
    end_time = time.clock()
    exec_time = end_time - start_time
    print("SVM execution time for %d samples: %f sec"%((nIter/2.),exec_time))
    print("SVM frequency: %f Hz"%(float(ypr.shape[0])/exec_time))
    #for y in ypr:  print(y),
    #print(clf._label_binarizer)
    
    true_m = 0
    false_m = 0
    true_s = 0
    false_s = 0
    for i in range(ypr.shape[0]):
        if ypr[i] == y[i]:
            #wellclassified event
            if y[i] == 0:#muon
                true_m += 1
            else:#shower
                true_s += 1
        else: 
            #misclassified event
            if ypr[i] == 0:
                false_m += 1
            else:
                false_s += 1
    
    print("Total misclassifications: %f%%"%((false_s+false_m)/100.))
    print("Efficiency: %f%%"%(true_m/(float(true_m)+float(false_s))*100.))
    print("Purity: %f%%"%(true_m/(float(true_m)+float(false_m))*100.))
    
    
    
    
    s = 0
    miscl = 0
    for i in range(ypr.shape[0]):
        if ypr[i] == y[i]:
            s += 1
        else :
            miscl += 1
    #rint("Purity: %d/1000"%miscl)  
    """
    print("SVM")
    clf = svm.SVC()
    clf.fit(X,y)
    ysvm = clf.predict(X_test)
    s = 0
    miscl = 0
    for i in np.range(ysvm.shape[0]):
        if ysvm[i] == y[i]:
            s += 1
        else :
            miscl += 1
    print("Misclassifications: %d/500"%miscl)   
    """
    #print(ypr)
    #print(y_test)
    
    """
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
    """