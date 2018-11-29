import numpy as np
import matplotlib.pyplot as plt
from ctapipe.io import event_source
from ctapipe.instrument import CameraGeometry
from ctapipe.visualization import CameraDisplay
import time
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import auc
from sklearn.externals import joblib
from scipy.optimize import minimize
from astropy.units import Quantity
from ctapipe.image.charge_extractors import GlobalPeakIntegrator, FullIntegrator
from ctapipe.calib.camera.r1 import CameraR1CalibratorFactory
from copy import deepcopy

def majority_func(image):
        """
        Returns the number of triggered (super)pixels
        """
        return image.sum()

if __name__ == '__main__':
    
    thr = 17 
    Npix = 2048
    datapath = []
    datapath.append("/Users/robertapillera/CTA2/Data/muon_0deg_0deg_run4___cta-prod3-sst-gct-s-1xnsb_desert-2150m-Paranal-1xNSB-default-trigger-sst-gct.simtel.gz")
    datapath.append("/Users/robertapillera/CTA2/Data/proton_0deg_0deg_run11___cta-prod3-sst-gct-s_desert-2150m-Paranal-1xNSB-default-trigger-sst-gct.simtel.gz")

    X = []
    y = []
    for i in arange(2):
        
        with event_source(datapath[i], max_events=200) as source:
            for event in source:
                if len(event.mc.tel)>1:
                    continue                
                for telid in event.r0.tels_with_data:
                    calibrator = CameraR1CalibratorFactory.produce(eventsource=source)
                    event_copy = deepcopy(event)
                    calibrator.calibrate(event_copy)
                    geom = event.inst.subarray.tel[telid].camera
                    integrator = FullIntegrator()
                    r1 = event_copy.r1.tel[telid].waveform
                    charge, peakpos, window = integrator.extract_charge(r1)
                    image = np.array(charge>thr,dtype=int) #select pixels above threshold
                    X.append(image)
                    y.append(i)

    xi_m = []
    xi_s = []
    for i in range(len(y)):
        if y[i] == 0: #it's a muon
            xi_m.append(majority_func(X[i]))
        else: #it's a shower
            xi_s.append(majority_func(X[i]))
        xlabel = 'number of pixels above threshold'
        bins = 150
        hist_range = (0,bins)

    xi_m = np.array(xi_m)
    xi_s = np.array(xi_s)
    fig1 = plt.figure()
    hist_m = plt.hist(xi_m,bins=bins,range=hist_range,label='muons',color='red',alpha=0.6)
    hist_s = plt.hist(xi_s,bins=bins,range=hist_range,label='showers',color='blue',alpha=0.6)       
    plt.legend()
    plt.title("Majority classifier")
    plt.xlabel(xlabel)
    plt.ylabel('entries')
    cut_val = hist_m[1][:-1]

    integral_m = hist_m[0].sum()
    integral_s = hist_s[0].sum()
    cs_m = np.cumsum(hist_m[0])
    cs_s = np.cumsum(hist_s[0])
    eff = cs_m/(float(integral_m))*100.
    pur = cs_m/(cs_m.astype(np.float)+cs_s.astype(np.float))*100.        
    true_pos = cs_m/(float(integral_m))
    false_pos = cs_s/(float(integral_s))

    fig = plt.figure(figsize=(7,6))
    ax = fig.add_subplot(111)
    area = auc(false_pos, true_pos)
    plt.plot(false_pos, true_pos, color='crimson',lw=2)
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    ax2 = ax.twinx()
    ax2.set_ylabel("Efficiency %")   
    ax2.set_ylim(0, 100)  
    ax3 = ax.twiny()   
    ax3.set_xlim(0,100)
    ax3.set_xlim(ax3.get_xlim()[::-1])
    ax3.set_xlabel("Purity %")
    plt.title("Majority classifier ROC curve with area = %0.2f"%area,y=1.09)
