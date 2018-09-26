from muon_tagger import MuonTagger 
import matplotlib.pyplot as plt

if __name__ == '__main__':

    muon_data1 = "Data/muon_0deg_0deg_run4___cta-prod3-sst-gct-s-1xnsb_desert-2150m-Paranal-1xNSB-default-trigger-sst-gct.simtel.gz"
    muon_data2 = "Data/muon_0deg_0deg_run4___cta-prod3-sst-gct-s-2xnsb_desert-2150m-Paranal-2xNSB-default-trigger-sst-gct.simtel.gz"
    muon_data3 = "/Users/robertapillera/CTA/Data/muon_0deg_0deg_run4___cta-prod3-sst-gct-s-3xnsb_desert-2150m-Paranal-3xNSB-default-trigger-sst-gct.simtel.gz"

    shower_data11 = "Data/proton_0deg_0deg_run11___cta-prod3-sst-gct-s_desert-2150m-Paranal-1xNSB-default-trigger-sst-gct.simtel.gz"
    shower_data22 = "Data/proton_0deg_0deg_run11___cta-prod3-sst-gct-s-2xnsb_desert-2150m-Paranal-2xNSB-default-trigger-sst-gct.simtel.gz"
    shower_data33 = "Data/proton_0deg_0deg_run11___cta-prod3-sst-gct-s-3xnsb_desert-2150m-Paranal-3xNSB-default-trigger-sst-gct.simtel.gz"

    N = 500
    mutagger = MuonTagger(
        alg='taubin',
        muon_data=muon_data1,
        shower_data=shower_data11,
        Nmax_muon=N,
        Nmax_shower=N,
        )

    #Taubin
    eff1, _, _, _, x1 = mutagger.compute_values(option=False)
    mutagger.muon_data = muon_data2
    mutagger.shower_data = shower_data22
    eff2, _, _, _, x2 = mutagger.compute_values(option=False)
    mutagger.muon_data = muon_data3
    mutagger.shower_data = shower_data33
    eff3, _, _, _, x3 = mutagger.compute_values(option=False)
    fig1 = plt.figure()
    plt.plot(x1, eff1, color='crimson',lw=2, label='low NSB')
    plt.plot(x2, eff2, color='green',lw=2, label='medium NSB')
    plt.plot(x3, eff3, color='blue',lw=2, label='high NSB')
    plt.legend()
    plt.xlabel("|r - 0.05 m|")
    plt.ylabel("efficiency %")
    plt.title('Efficiency %s classifier for different NSB'%mutagger.alg)
    
    #MLP
    mutagger.alg = 'mlp'
    mutagger.muon_data = muon_data1
    mutagger.shower_data = shower_data11
    eff1, _, _, _, x1 = mutagger.compute_values(option=False)
    mutagger.muon_data = muon_data2
    mutagger.shower_data = shower_data22
    eff2, _, _, _, x2 = mutagger.compute_values(option=False)
    mutagger.muon_data = muon_data3
    mutagger.shower_data = shower_data33
    eff3, _, _, _, x3 = mutagger.compute_values(option=False)
    fig2 = plt.figure()
    plt.plot(x1, eff1, color='crimson',lw=2, label='low NSB')
    plt.plot(x2, eff2, color='green',lw=2, label='medium NSB')
    plt.plot(x3, eff3, color='blue',lw=2, label='high NSB')
    plt.legend()
    plt.xlabel("prediction probability")
    plt.ylabel("efficiency %")
    plt.title('Efficiency %s classifier for different NSB'%mutagger.alg)
    
    #Majority
    mutagger.alg = 'majority'
    mutagger.muon_data = muon_data1
    mutagger.shower_data = shower_data11
    eff1, _, _, _, x1 = mutagger.compute_values(option=False)
    mutagger.muon_data = muon_data2
    mutagger.shower_data = shower_data22
    eff2, _, _, _, x2 = mutagger.compute_values(option=False)
    mutagger.muon_data = muon_data3
    mutagger.shower_data = shower_data33
    eff3, _, _, _, x3 = mutagger.compute_values(option=False)
    fig3 = plt.figure()
    plt.plot(x1, eff1, color='crimson',lw=2, label='low NSB')
    plt.plot(x2, eff2, color='green',lw=2, label='medium NSB')
    plt.plot(x3, eff3, color='blue',lw=2, label='high NSB')
    plt.legend()
    plt.xlim(480,512)
    plt.xlabel("number of non-triggered superpixelsy")
    plt.ylabel("efficiency %")
    plt.title('Efficiency %s classifier for different NSB'%mutagger.alg)
    
