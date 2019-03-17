#!/bin/env python
import sys, os
import numpy as np
import warnings
#from astropy.table import Table
from ctapipe.calib import CameraCalibrator
#from ctapipe.core import Tool
from ctapipe.core import traits as t
#from ctapipe.image.muon.muon_diagnostic_plots import plot_muon_event
#from ctapipe.image.muon.muon_reco_functions import analyze_muon_event
from ctapipe.io import event_source
#from ctapipe.utils import get_dataset

warnings.filterwarnings("ignore")  # Supresses iminuit warnings

if __name__ == '__main__':
    
    #print("Program started")   
    if sys.argv[1:]:
        arg = int(sys.argv[1]) #0 for first half of proton files, 1 for second
	#n_events = int(sys.argv[3])
    else:
        arg = 0
    nn = np.array([0,26])
    pix_thr = np.array([5,10,15,20,25,30,35])
    #sim_nr = np.array([1,4,5,7,8,9,11,12,14,16,17,18,22,23,24,25,27,29,30,31,32,34,36,37,38,40,41,43,44,45,46,47,48,49,51,52,54,55,56,57,58,60,64,65,66,68,69,71,72,73,74,77])
    sim_nr = np.array([[1,4,5,7,8,9,11,12,14,16,17,18,22,23,24,25,27,29,30,31,32,34,36,37,38,40],[41,43,44,45,46,47,48,49,51,52,54,55,56,57,58,60,64,65,66,68,69,71,72,73,74,77]])
    #sim_dir = "/lustrehome/divenere/CTA_LST/" for muons
    sim_dir = "/lustre/home/divenere/CTA/MC/LaPalma_proton_North_20deg_LaPalma3_qgs2/"
    s1 = "proton_20deg_0deg_run5000"
    s11 = "proton_20deg_0deg_run500"
    s2 = "___cta-prod3-lapalma3-2147m-LaPalma.simtel.gz"
    
    n_events = 10000
    tot_nr_ev = 0

    #file to store nr of events and usable images
    info_file_name = "/lustrehome/pillera/CTA/Muons/Proton_sample_info.txt"
    info_file = open(info_file_name,"a+")
    info_file.write("Run nr / total nr of events / total nr of images / nr of LST images\n")
    print("Run nr / total nr of events / total nr of images / nr of LST images")
    
    for i in range(len(sim_nr[arg])): #loop over files

        numev = 0
        num_img = 0
        num_us_img = 0

        if (i <= 5) and arg == 0:
            sim_name = sim_dir+s1+str(sim_nr[arg][i])+s2
        else:
            sim_name = sim_dir+s11+str(sim_nr[arg][i])+s2

        source = event_source(sim_name, max_events=n_events)
        calib = CameraCalibrator(r1_product="HESSIOR1Calibrator",eventsource=source)

        #Get size distribution
        #foutsize_name = "/lustrehome/pillera/CTA/Muons/ProtonFiles/Size/Proton_"+str(sim_nr[arg][i])+"_size.txt"
        foutsize_name = "/lustrehome/pillera/CTA/Muons/ProtonFiles/Size/Proton_"+str(nn[arg]+i)+"_size.txt"
        foutsize = open(foutsize_name,"w") #output file

        for event in source:
            
            #print("Event nr %d"%numev)
            
            for telid in event.r0.tels_with_data:
                num_img += 1
                #print(event.inst.subarray.tel[telid].camera.cam_id);
                if (event.inst.subarray.tel[telid].camera.cam_id != 'LSTCam'):
                    continue
                else:
                    calib.calibrate(event)
                    size = event.dl1.tel[telid].image[0].sum()
                    foutsize.write(str(size))
                    foutsize.write("\n")
                    num_us_img += 1
            numev += 1

        tot_nr_ev += num_us_img
        foutsize.close()

        #write file info
        outstr = str(sim_nr[arg][i])+" "+str(numev)+" "+str(num_img)+" "+str(num_us_img)+"\n"
        print(outstr)
        info_file.write(outstr)
        
        #Get threshold distributions
        for j in range(len(pix_thr)): #loop over thresholds
            
            #fileout_name = "/lustrehome/pillera/CTA/Muons/ProtonFiles/"+str(pix_thr[j])+"pe/Proton_"+str(sim_nr[arg][i])+"_thres_"+str(pix_thr[j])+".txt"
            fileout_name = "/lustrehome/pillera/CTA/Muons/ProtonFiles/"+str(pix_thr[j])+"pe/Proton_"+str(nn[arg]+i)+"_thres_"+str(pix_thr[j])+".txt"
            fileout = open(fileout_name,"w") #output file
            #numev = 0
            for event in source:
                
                #print("Event nr %d"%numev)
                
                for telid in event.r0.tels_with_data:
                    if (event.inst.subarray.tel[telid].camera.cam_id != 'LSTCam'):
                        continue
                    else:
                        calib.calibrate(event)
                        n_pix = np.array(event.dl1.tel[telid].image[0]>pix_thr[j],dtype=int).sum()
                        fileout.write(str(n_pix))
                        fileout.write("\n")
                
            fileout.close()
    
    info_file.write("Total number of usable events: %d"%tot_nr_ev)
    print("Total number of usable events: %d"%tot_nr_ev)
    info_file.close()
    
    
