#!/bin/env python
import sys, os
import warnings
import numpy as np
from ctapipe.calib import CameraCalibrator
from ctapipe.io import event_source
from ctapipe.image.muon.muon_reco_functions import analyze_muon_event
import time

warnings.filterwarnings("ignore")  # Supresses iminuit warnings

if __name__ == '__main__':
    
    #print("Program started") 
    if sys.argv[1:]:
        thr = int(sys.argv[1]) #north ns = 1 or south pointing ns=2
        stop = int(sys.argv[2])
        #run = int(sys.argv[2])
        #group = int(sys.argv[2])
    else:
        thr = 3500. #cut at 0.7 ringcompleteness
        stop = 5000
        #group = 1
        #run = 1
    """
    one group processes 500 runs
    the first job -> group = 1
    group -> 2 it starts from run 501 up to 1000
    """
    #north or south pointing
    ns = 1

    sim_dir = "/fefs/aswg/workspace/MC_common/corsika6.9_simtelarray_2018-11-07/LST4_monotrigger/prod3/proton/proton_20190226/"
    filename = sim_dir
    endstring = "___cta-prod3-demo-2147m-LaPalma-baseline-mono.simtel.gz"
    if ns == 1:
        filename += "North_pointing/Data/"
        filename += "proton_20deg_"
        filename += "0"
    elif ns == 2:
        filename += "South_pointing/Data/"
        filename += "proton_20deg_"
        filename += "180"
    filename += "deg_run"

    
    start = 1
    #stop = 5000

    tot_numev = 0
    #tot_numimg = 0
    taggedmuons = 0
    selectedmuons = 0
    info = {'Run': [],
            'Ev_nr': []}
    t_start = time.clock()
    for run in range(start,stop+1):
        
        sim_name = filename + str(run) + endstring
        n_events = 10000

        source = event_source(sim_name, max_events=n_events)
        calib = CameraCalibrator(r1_product="HESSIOR1Calibrator",eventsource=source)

        numev = 0
        for event in source:
            
            calib.calibrate(event)
            
            # tag = [False]*len(event.dl0.tels_with_data) 
            # i = 0
            # for telid in event.r0.tels_with_data:
            #     i += 1
            #     #tot_numimg += 1
            #     if event.dl1.tel[telid].image[0].sum() > thr :
            #         #muon was tagged!
            #         tag[i-1] = True
            #         taggedmuons += 1

            numev += 1
            tot_numev += 1
            # if np.array(tag).sum() == 0: # no image is preselected
            #     continue
            # else: #analyze
            #     muon_evt = analyze_muon_event(event)
            #     if muon_evt['MuonIntensityParams']: #Muon is selected
            #         selectedmuons += 1
            #         info['Run'].append(run)
            #         info['Ev_nr'].append(numev-1)

                
    
    t_end = time.clock()         
    
    t_total = t_end - t_start
    # tab = Table(info)
    # tab.write("PreselectionResults"+str(ns)+".fits",format='fits')   

    print("CALIBRATION ONLY")
    print("Processing time: %f sec"%t_total)
    print("Total number of events: %d"%tot_numev)
    # print("Total tagged muons: %d"%taggedmuons)
    # print("Total selected muons: %d"%selectedmuons)
    print("Processing rate (n_tot/t_tot): %f"%(float(tot_numev)/t_total))
    #print("Total number of images: %d"%tot_numimg)

    
    

        
    
    
