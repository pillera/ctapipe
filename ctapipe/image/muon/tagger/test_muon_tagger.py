from muon_tagger import MuonTagger

"""
Test of MuonTagger class
"""
if __name__ == '__main__':
    
    muon_data = "/Users/robertapillera/CTA/Data/muon_0deg_0deg_run4___cta-prod3-sst-gct-s-1xnsb_desert-2150m-Paranal-1xNSB-default-trigger-sst-gct.simtel.gz"
    shower_data = "/Users/robertapillera/CTA/Data/proton_0deg_0deg_run11___cta-prod3-sst-gct-s_desert-2150m-Paranal-1xNSB-default-trigger-sst-gct.simtel.gz"
    
    N = 500
    mutagger = MuonTagger(
        alg='mlp',
        muon_data=muon_data,
        shower_data=shower_data,
        Nmax_muon=N,
        Nmax_shower=N,
        )

    #mutagger.test_efficiency()
    #mutagger.test_roc(option=False)
    print(mutagger.test_speed(90))

