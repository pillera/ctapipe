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
    
    print("Program started") 
    if sys.argv[1:]:
        ns = int(sys.argv[1]) #north ns = 1 or south pointing ns=2
        run = int(sys.argv[2])
    else:
        ns = 1
        run = 1

    sim_dir = "/fefs/aswg/workspace/MC_common/corsika6.9_simtelarray_2018-11-07/LST4_monotrigger/prod3/proton/proton_20190226/"
    filename = "proton_20deg_"
    if ns == 1:
        sim_dir += "North_pointing/Data/"
        filename += "0"
    elif ns == 2:
        sim_dir = "South_pointing/Data/"
        filename += "180"
    filename += "deg_run"
    filename += str(run)
    filename += "___cta-prod3-demo-2147m-LaPalma-baseline-mono.simtel.gz"

    n_events = 1
    tot_nr_ev = 0

    source = event_source(sim_name, max_events=n_events)
    calib = CameraCalibrator(r1_product="HESSIOR1Calibrator",eventsource=source)

    foutsize_name = "/home/roberta.pillera/MuonAnalysis/ProtonFiles/Run_"+str(run)+"_size.txt"
    foutsize = open(foutsize_name,"w") #output file

    numev = 0
    for event in source:
        
        print("Event nr %d"%numev)
        
        for telid in event.r0.tels_with_data:
            num_img += 1
            
            calib.calibrate(event)
            size = event.dl1.tel[telid].image[0].sum()
            foutsize.write(str(size))
            foutsize.write("\n")
            print("Size: %f"%size)
            
        numev += 1

    
    foutsize.close()

        
    
    
