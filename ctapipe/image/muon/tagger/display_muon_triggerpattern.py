import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ctapipe.io import event_source
from ctapipe.utils import get_dataset_path
from ctapipe.instrument import CameraGeometry
from ctapipe.visualization import CameraDisplay
from ctapipe.image.charge_extractors import GlobalPeakIntegrator, FullIntegrator
from ctapipe.calib.camera.r1 import CameraR1CalibratorFactory
from copy import deepcopy
from dask.array.wrap import full
from astropy.io import ascii


if __name__ == '__main__':

    filename = "Data/muon_0deg_0deg_run4___cta-prod3-sst-gct-s-1xnsb_desert-2150m-Paranal-1xNSB-default-trigger-sst-gct.simtel.gz"

    dic = np.loadtxt("CHEC_pixel_dictionary.dat",dtype=int)
    geom_trigpat = CameraGeometry.from_table("CHEC_superpix_tab.fits",format="fits")
    fig = plt.figure(1,figsize=(15,5))

    with event_source(filename, max_events=10) as source:
        for event in source:
            if len(event.mc.tel)>1:
                continue #exclude images with more than one telescope
            for telid in event.r0.tels_with_data:
                plt.clf() #Clear figure for next image
                #Create camera image
                calibrator = CameraR1CalibratorFactory.produce(eventsource=source)
                event_copy = deepcopy(event)
                calibrator.calibrate(event_copy)
                geom = event.inst.subarray.tel[telid].camera
                integrator = FullIntegrator()
                r1 = event_copy.r1.tel[telid].waveform
                charge, peakpos, window = integrator.extract_charge(r1)
                charge = charge[0]
                trig_image = np.zeros(512)
                trig_image[dic[event.r0.tel[telid].trig_pix_id]] = 1.0 
                #Plot camera image
                axes = plt.subplot(1,2,1)
                disp = CameraDisplay(geom,image=charge, ax=axes)
                disp.add_colorbar() 
                axes = plt.subplot(1,2,2)
                disp2 = CameraDisplay(geom_trigpat,image=trig_image, ax=axes)
                disp2.cmap = plt.cm.Greys_r
                disp2.set_limits_minmax(0,1.0)
                disp2.add_colorbar()
                plt.show()