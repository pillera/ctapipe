#!/bin/env python
import sys, os
import warnings
import numpy as np
from ctapipe.calib import CameraCalibrator
from ctapipe.io import event_source
from ctapipe.image.muon.muon_reco_functions import analyze_muon_event
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
        ns = 1
        group = 1
        
    """
    one group processes 500 runs
    the first job -> group = 1
    group -> 2 it starts from run 501 up to 1000
    """
    #north or south pointing
    #ns = 1
    
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
    stop = 2    
    tab = { 'Run_nr': [],
            'Ev_nr': [],
            'Tel_id': [],
            'Flag': [],
            'RingComp':[],
            'Energy': [],
            'Size': []}

    
    for run in range(start,stop+1):
        
        sim_name = filename + str(run) + endstring
        n_events = None

        source = event_source(sim_name, max_events=n_events)
        calib = CameraCalibrator(r1_product="HESSIOR1Calibrator",eventsource=source)

        numev = 0
        for event in source:
            
            numev += 1
            calib.calibrate(event)
            muon_evt = analyze_muon_event(event)
            if muon_evt['TelIds'] is not None:
                for tid in muon_evt['TelIds']:
                    idx = muon_evt['TelIds'].index(tid)
                    tab['Run_nr'].append(run)
                    tab['Ev_nr'].append(numev-1)
                    tab['Tel_id'].append(idx)
                    tab['Energy'].append(event.mc.energy.value)
       	       	    tab['Size'].append(event.dl1.tel[tid].image[0].sum())
                    if muon_evt['MuonIntensityParams'][idx] is not None:
                        tab['Flag'].append(True)
                        tab['RingComp'].append(muon_evt['MuonIntensityParams'][idx].ring_completeness)
                    else:
                        tab['Flag'].append(False)
                        tab['RingComp'].append(-1)
            else:
                for tid in event.r0.tels_with_data:
                    #idx = muon_evt['TelIds'].index(tid)
                    tab['Run_nr'].append(run)
                    tab['Ev_nr'].append(numev-1)
                    tab['Tel_id'].append(tid)
                    tab['Energy'].append(event.mc.energy.value)
                    tab['Size'].append(event.dl1.tel[tid].image[0].sum())
                
    table = Table(tab)
#    table.write("/home/roberta.pillera/MuonAnalysis/SummaryTables/Fit_table_all_"+str(ns)+"_"+str(group)+".fits",format='fits')
    table.write("/home/roberta.pillera/MuonAnalysis/SummaryTables/Test.fits",format="fits")   

    
    

        
    
    
