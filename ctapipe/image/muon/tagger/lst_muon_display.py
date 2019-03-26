import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ctapipe.io import event_source
from ctapipe.instrument import CameraGeometry
from ctapipe.visualization import CameraDisplay
from ctapipe.calib.camera.calibrator import CameraCalibrator
from ctapipe.image.muon.muon_reco_functions import analyze_muon_event


if __name__ == '__main__':

    filename = "/fefs/aswg/workspace/MC_common/corsika6.9_simtelarray_2018-11-07/LST4_monotrigger/prod3/proton/proton_20190226/"
    filename += "North_pointing/Data/proton_20deg_0deg_run1___cta-prod3-demo-2147m-LaPalma-baseline-mono.simtel.gz"

    #dic = np.loadtxt("CHEC_pixel_dictionary.dat",dtype=int)
    #geom_trigpat = CameraGeometry.from_table("CHEC_superpix_tab.fits",format="fits")
    #fig = plt.figure(1,figsize=(7,6))

    e = 0
    source = event_source(filename, max_events=20)
    calibrator = CameraCalibrator(r1_product="HESSIOR1Calibrator",eventsource=source)

    
    for event in source:
        
        calibrator.calibrate(event)
        muon_evt = analyze_muon_event(event)
        print(muon_evt['MuonIntensityParams'])
        #for telid in event.r0.tels_with_data:

            # plt.clf() #Clear figure for next image
            # #Create camera image
            
            # geom = event.inst.subarray.tel[telid].camera
            # image = event.dl1.tel[telid].image[0]
            
            # #Plot camera image
            # axes = plt.subplot()
            # disp = CameraDisplay(geom,image=image, ax=axes)
            # disp.add_colorbar() 
            # plt.title("Event %d"%e)
            # plt.show()
            # plt.savefig("/lustrehome/pillera/CTA/Muons/Pictures/ProtonEvents/LST_proton_%d"%e)

                
        e += 1
                