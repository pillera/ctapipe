import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ctapipe.io import event_source
from ctapipe.utils import get_dataset_path
from ctapipe.instrument import CameraGeometry
from ctapipe.visualization import CameraDisplay
from ctapipe.image.charge_extractors import *
from ctapipe.calib.camera.r1 import CameraR1CalibratorFactory
from ctapipe.calib.camera.calibrator import CameraCalibrator
from copy import deepcopy
from dask.array.wrap import full
from astropy.io import ascii


if __name__ == '__main__':

    filename = "/Users/robertapillera/Downloads/muon_lst.simtel.gz"

    #dic = np.loadtxt("CHEC_pixel_dictionary.dat",dtype=int)
    #geom_trigpat = CameraGeometry.from_table("CHEC_superpix_tab.fits",format="fits")
    fig = plt.figure(1,figsize=(7,6))

    ii = np.array([8,71,81,114,167,190,196,242,477,532,564])
    ii = np.array([13,17,19,20,30,40,46])
    ii = np.array([289,333])
    i = 0
    e = 0
    with event_source(filename, max_events=2) as source:
        for event in source:
            e += 1
            if e != 40:
                continue
            print("Nr of telescopes: %d"%len(event.mc.tel))
            # if len(event.mc.tel)>1:
            #     continue #exclude images with more than one telescope
            for telid in event.r0.tels_with_data:
                
                if i == 0:
                    plt.clf() #Clear figure for next image
                    #Create camera image
                    calibrator = CameraCalibrator(r1_product="HESSIOR1Calibrator",eventsource=source)
                    event_copy = deepcopy(event)
                    calibrator.calibrate(event_copy)
                    geom = event.inst.subarray.tel[telid].camera
                    integrator = FullIntegrator()
                    dl0 = event_copy.dl0.tel[telid].waveform
                    charge, peakpos, window = integrator.extract_charge(dl0)
                    charge = charge[0]
                    print(charge)
                    #Plot camera image
                    axes = plt.subplot()
                    disp = CameraDisplay(geom,image=charge, ax=axes)
                    disp.add_colorbar() 
                    
                    # disp2 = CameraDisplay(geom_trigpat,image=trig_image, ax=axes)
                    # disp2.cmap = plt.cm.Greys_r
                    # disp2.set_limits_minmax(0,1.0)
                    # disp2.add_colorbar()
                    plt.show()

                #i += 1

                