#!/bin/env python
import sys, os
import numpy as np
from ctapipe.calib import CameraCalibrator
from ctapipe.io import event_source



if __name__ == '__main__':
    
    #print("Program started") 
    if sys.argv[1:]:
        ns = int(sys.argv[1]) #north ns = 1 or south pointing ns=2
        #run = int(sys.argv[2])
        group = int(sys.argv[2])
    else:
        ns = 1
        group = 1
        run = 1
    """
    one group processes 500 runs
    the first job -> group = 1
    group -> 2 it starts from run 501 up to 1000
    """

    

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
    numev = 0

    foutsize_name = "/home/roberta.pillera/MuonAnalysis/ProtonFiles/Group_"+str(ns)+"_"+str(group)+"_size.txt"
    foutsize = open(foutsize_name,"w") #output file

    for run in range(start,stop):
        
        sim_name = filename + str(run) + endstring
        n_events = 10000

        source = event_source(sim_name, max_events=n_events)
        calib = CameraCalibrator(r1_product="HESSIOR1Calibrator",eventsource=source)

        numev = 0
        for event in source:
            calib.calibrate(event)
            for telid in event.r0.tels_with_data:
                
                
                size = event.dl1.tel[telid].image[0].sum()
                foutsize.write(str(size))
                foutsize.write("\n")
                
                
            numev += 1

    print("%d events in group %d"%(numev,group))
    
    foutsize.close()

        
    
    
