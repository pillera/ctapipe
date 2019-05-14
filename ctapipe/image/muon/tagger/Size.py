#!/bin/env python
import sys, os
import warnings
import numpy as np
from ctapipe.calib import CameraCalibrator
from ctapipe.io import event_source
from ctapipe.image.muon.muon_reco_functions import analyze_muon_event
import time
from astropy.table import Table

warnings.filterwarnings("ignore")  # Supresses iminuit warnings



if __name__ == '__main__':
    
    #print("Program started") 
    if sys.argv[1:]:
        ns = int(sys.argv[1]) 
        group = int(sys.argv[2])
        #run = int(sys.argv[2])
        #group = int(sys.argv[2])
    else:
        thr = 2500. #cut at 0.7 ringcompleteness
        stop = 5000
        #group = 1
        #run = 1
    """
    one group processes 500 runs
    the first job -> group = 1
    group -> 2 it starts from run 501 up to 1000
    """
    #north or south pointing
    #ns = 1
    thr = 3500.
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
    

    
    start = (group-1)*500+1
    stop = start + 500
    #stop = 5000

    tot_numev = 0
    #tot_numimg = 0
    taggedmuons = 0
    selectedmuons = 0
    info = {'Run': [],
            'Ev_nr': []}
    time_tab = {'Run_nr': [],
                'Ev_nr': [],
                'Time': [],
                'Energy': []
                }
    size_tab = {'Run_nr': [],
                'Ev_nr': [],
                'Size': []}
    
    for run in range(start,stop+1):
        
        sim_name = filename + str(run) + endstring
        n_events = 10000

        source = event_source(sim_name, max_events=n_events)
        calib = CameraCalibrator(r1_product="HESSIOR1Calibrator",eventsource=source)

        numev = 0
        for event in source:
            
            size_tab['Run_nr'].append(run)
            size_tab['Ev_nr'].append(numev)
            
            calib.calibrate(event)
            
            for telid in event.r0.tels_with_data:
                size_tab['Size'].append(event.dl1.tel[telid].image[0].sum())
                #tot_numimg += 1
                

            numev += 1
            tot_numev += 1
            
                      
             
    
        
    
    
    sizetable = Table(time_tab)
    sizetable.write("/home/roberta.pillera/MuonAnalysis/ProtonFiles/Size"+str(ns)+"_"+str(group)+".fits",format='fits')
    # print("MUON SELECTION")
    # print("Processing time: %f sec"%t_total)
    # print("Total number of events: %d"%tot_numev)
    # print("Total tagged muons: %d"%taggedmuons)
    # print("Total selected muons: %d"%selectedmuons)
    # print("Processing rate (n_tot/t_tot): %f"%(float(tot_numev)/t_total))
    #print("Total number of images: %d"%tot_numimg)

    
    

        
    
    
