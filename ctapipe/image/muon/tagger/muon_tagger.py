import numpy as np
import matplotlib.pyplot as plt
from ctapipe.io import event_source
from ctapipe.instrument import CameraGeometry
from ctapipe.visualization import CameraDisplay
import time
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import auc
from sklearn.externals import joblib
from scipy.optimize import minimize
from astropy.units import Quantity

"""
Index:
    1. MuonTagger class line 18
    2. MLP trainer function line 429
"""
class MuonTagger:
    """
    Muon tagger efficiency, purity and frequency testing
    class.
    Classification of muons with:
        1. taubin circle fit 'taubin'
        2. multilayer perceptron machine learning algtorithm 'mlp'
            (a trained mlp classifier has to be given)
        3. number of triggered superpixel 'majority'

    Parameters
    ----------
    alg : str
        'tabubin', 'mlp' or 'majority'
    muon_data : str
        muon data filepath
    shower_data : str
        shower data filepath
    Nmax_muon : int
        maximum number of muon samples
    Nmax_shower : int
        maximum number of shower samples
    spfile : str
        filepath of the CHEC superpixel dictionary in .dat
        i.e. the corresponding superpixel for each pixel
        necessary for trigger pattern readout
    geomfile : str
        filepath to the CHEC superpixel geometry table 
        to be given to CameraGeometry in .fits format
        needed for taubin 
    clfile : str
        filename of the trained classifier in .pkl format
        default "Trained_Classif.pkl" 
    """

    def __init__(
                self,
                alg,
                muon_data,
                shower_data,
                Nmax_muon=500,
                Nmax_shower=500,
                spfile="CHEC_pixel_dictionary.dat",
                geomfile="CHEC_superpix_tab.fits",
                clfile="Trained_Classif.pkl"
                ):
        self.alg = alg
        self.muon_data = muon_data
        self.shower_data = shower_data
        self.Nmax_muon = Nmax_muon
        self.Nmax_shower = Nmax_shower
        self.spfile = spfile
        self.geomfile = geomfile
        self.clfile = clfile

    def _get_data(self):
        """
        Get trigger pattern from CORSIKA 
        simtel array simulation files
        Returns
        -------
        X : matrix with triggerpattern info 
        y : list with labels
        """
        chec_dic = np.loadtxt(self.spfile,dtype=int)
        X = [] #Test sample images
        y = [] #Test sample labels
           
        #Muon sample preparation
        #Note: muon label = 0
        n_m = self.Nmax_muon
        n_s = self.Nmax_shower
        with event_source(self.muon_data, max_events=n_m) as source:  
            for event in source:
                if len(event.mc.tel)>1:
                    continue                
                for telid in event.r0.tels_with_data:
                    geom = event.inst.subarray.tel[telid].camera
                    if geom.cam_id != 'CHEC':
                        print("Warning: different non CHEC camera event!")
                        continue
                    trig_info = np.zeros(512)
                    trig_info[chec_dic[event.r0.tel[telid].trig_pix_id]] = 1.0
                    X.append(trig_info)
                    y.append(0)
                    

        #Shower sample preparation
        #Note: shower label = 1
        
        with event_source(self.shower_data, max_events=n_s) as source:
            for event in source:
                if len(event.mc.tel)>1:
                    continue                
                for telid in event.r0.tels_with_data:
                    geom = event.inst.subarray.tel[telid].camera
                    if geom.cam_id != 'CHEC':
                        print("Warning: different non CHEC camera event!")
                        continue
                    trig_info = np.zeros(512)
                    trig_info[chec_dic[event.r0.tel[telid].trig_pix_id]] = 1.0
                    X.append(trig_info)
                    y.append(1)

        return X, y

    def compute_values(self, option=True):
        """
        Parameters
        ----------
        option: bool
            plot of classification parameters distribution 
            for given sample if option=True
        Returns
        -------
        eff: float array-like
            efficiency in %
        pur: float array-like
            purity in %
        true_pos: float array-like
            true positive rate in [0,1]
        false_pos: float array-like
            false positive rate in [0,1]
        cut_val: float array-like
            cut threshold values
        """
        X, y = self._get_data()
        xi_m = []
        xi_s = []

        if self.alg == 'taubin':
            geom = CameraGeometry.from_table(self.geomfile,format="fits")
            pos = np.empty(geom.pix_x.shape + (2,))
            pos[..., 0] = geom.pix_x.value
            pos[..., 1] = geom.pix_y.value
            for i in range(len(y)):
                rad, cx, cy = self.taubin_fit(pos[...,0],pos[...,1],X[i])
                if y[i] == 0: #it's a muon
                    xi_m.append(np.abs(rad-0.05))
                else: #it's a shower
                    xi_s.append(np.abs(rad-0.05))
            xlabel = '|r - 0.05 m|'
            bins = 50
            hist_range = (0.,0.12)
        elif self.alg == 'mlp':
            clf = joblib.load(self.clfile)           
            y_pred = clf.predict_proba(X)
            for i in range(len(y)):
                if y[i] == 0: #it's a muon
                    xi_m.append(y_pred[i][1])
                else: #it's a shower
                    xi_s.append(y_pred[i][1])
            xlabel = 'prediction probability'
            bins = 50
            hist_range = (0.,1.)
        elif self.alg == 'majority':
            for i in range(len(y)):
                if y[i] == 0: #it's a muon
                    xi_m.append(512-self.majority_func(X[i]))
                else: #it's a shower
                    xi_s.append(512-self.majority_func(X[i]))
            xlabel = 'number of non-triggered superpixels'
            bins = 512
            hist_range = (0,512)
        else: 
            print("ERROR: Invalid algorithm name")
            return

        xi_m = np.array(xi_m)
        xi_s = np.array(xi_s)
        if option==True:
            fig1 = plt.figure()
            hist_m = plt.hist(xi_m,bins=bins,range=hist_range,label='muons',color='red',alpha=0.6)
            hist_s = plt.hist(xi_s,bins=bins,range=hist_range,label='showers',color='blue',alpha=0.6)
            if self.alg == 'majority':
                plt.xlim(480,512)        
            plt.legend()
            plt.title("%s classifier"%self.alg)
            plt.xlabel(xlabel)
            plt.ylabel('entries')
            cut_val = hist_m[1][:-1]
        else :
            hist_m = np.histogram(xi_m,bins=bins,range=hist_range)
            hist_s = np.histogram(xi_s,bins=bins,range=hist_range)        
            cut_val = hist_m[1][:-1]
        integral_m = hist_m[0].sum()
        integral_s = hist_s[0].sum()
        cs_m = np.cumsum(hist_m[0])
        cs_s = np.cumsum(hist_s[0])
        eff = cs_m/(float(integral_m))*100.
        pur = cs_m/(cs_m.astype(np.float)+cs_s.astype(np.float))*100.        
        true_pos = cs_m/(float(integral_m))
        false_pos = cs_s/(float(integral_s))
        

        return eff, pur, true_pos, false_pos, cut_val

    def test_efficiency(self, option=True):
        """
        Test of efficiency and purity performances of 
        selection algorithm
        Parameters
        ----------
        option: bool

        Return
        ------
        plot of classification parameters distribution 
            for given sample if option=True
        plot of efficiency and purity vs. classification
            parameter
        """
        eff, pur, _, _, cut_val= self.compute_values(option)
        fig = plt.figure()
        histc_m = plt.plot(cut_val,eff,lw=2, label='efficiency', color='green')
        histc_s = plt.plot(cut_val,pur,lw=2,label='purity',color='darkorange')
        if self.alg == 'taubin':
            xlabel = '|r - 0.05 m|'
        elif self.alg == 'mlp':
            xlabel = 'prediction probability'
        else :
            plt.xlim(480,512)   
            xlabel = 'number of non-triggered superpixels'
        plt.xlabel(xlabel)
        plt.ylabel('%')
        plt.legend()
        plt.title("%s classifier"%self.alg)   
    
        
    def test_roc(self, option=True):
        """
        Test of receiver operating characteristic of the
        selection algorithm
        Parameters
        ----------
        option: bool

        Return
        ------
        plot of classification parameters distribution 
            for given sample if option=True
        plot of roc curve
        """

        eff, pur, true_pos, false_pos, _ = self.compute_values(option)  
        #ROC curve plot
        fig = plt.figure(figsize=(7,6))
        ax = fig.add_subplot(111)
        area = auc(false_pos, true_pos)
        plt.plot(false_pos, true_pos, color='crimson',lw=2)
        plt.xlim(0,1)
        plt.ylim(0,1)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        ax2 = ax.twinx()
        ax2.set_ylabel("Efficiency %")   
        ax2.set_ylim(0, 100)  
        ax3 = ax.twiny()   
        ax3.set_xlim(0,100)
        ax3.set_xlim(ax3.get_xlim()[::-1])
        ax3.set_xlabel("Purity %")
        plt.title("%s classifier ROC curve with area = %0.2f"%(self.alg,area),y=1.09)
   
    def test_speed(self, eff_star):
        """
        Gives an estimate of the speed of the classificator. This is not an 
        efficient implementation.
        Parameters
        ----------
        eff_star : double
            required speed test efficiency (in %) 

        Returns
        -------
        y_class : array-like
            predicted labels
        freq : float
            evaluated frequency of the method at the given efficiency
        xi_cut: float
            cut value at given efficiency
        pur_cut: float
            purity at given efficiency
        """
        eff, pur, true_pos, false_pos, cut_val = self.compute_values(option=False)  
        X, y = self._get_data()
        xi_cut = cut_val[np.where(eff >= eff_star)[0][0]]
        pur_cut = pur[np.where(eff >= eff_star)[0][0]]
        y_class = []
        n_m = 0
        n_s = 0

        if self.alg == 'taubin':
            geom = CameraGeometry.from_table(self.geomfile,format="fits")
            pos = np.empty(geom.pix_x.shape + (2,))
            pos[..., 0] = geom.pix_x.value
            pos[..., 1] = geom.pix_y.value
            start_time = time.clock()
            for i in range(len(y)):                
                rad, cx, cy = self.taubin_fit(pos[...,0],pos[...,1],X[i])
                if np.abs(rad-0.05) <= xi_cut: #it's a muon
                    n_m += 1
                    y_class.append(0)
                else: #it's a shower
                    n_s += 1
                    y_class.append(1)
            end_time = time.clock()
        elif self.alg == 'mlp':
            clf = joblib.load("Trained_Classif.pkl")  
            start_time = time.clock()          
            y_pred = clf.predict_proba(X)
            for i in range(len(y)):
                if y_pred[i][1] <= xi_cut: #it's a muon
                    y_class.append(0)
                else: #it's a shower
                    y_class.append(1)
            end_time = time.clock()
        elif self.alg == 'majority':
            start_time = time.clock()
            for i in range(len(y)):
                xi = 512-self.majority_func(X[i])
                if xi <= xi_cut: #it's a muon
                    y_class.append(0)
                else: #it's a shower
                    y_class.append(1)
            end_time = time.clock()
        else: 
            print("ERROR: Invalid algorithm name")
            return

        exec_time = end_time - start_time
        freq = float(len(y_class))/exec_time
        return y_class, freq, xi_cut, pur_cut

    def taubin_func(self,params, x, y, image):
        """
        Taubin fit function to be minimized
        Parameters
        ----------
        params: 3-tuple
            fit parameters (radius, center_x, center_y)
        x: array-like
            x coordinates
        y: array-like
            y coordinates
        image: array-like
            trigger pattern image
        """
        r, cx, cy = params
        return (((x-cx)**2+(y-cy)**2-r**2)**2)[image].sum()/(((x-cx)**2+(y-cy)**2)**2)[image].sum()

    def taubin_fit(self,x, y, image):
        """
        Do a Taubin fit using 
        Uses the sample mean and std for the initial guess
        Parameters
        ----------
        x: array-like or astropy quantity
            x coordinates of the points
        y: array-like or astropy quantity
            y coordinates of the points
        image: array-like
            weights of the points (0,1)

        This will usually be x and y coordinates and if pixel is on or off

        Returns
        -------
        radius: astropy-quantity
            radius of the ring
        center_x: astropy-quantity
            x coordinate of the ring center
        center_y: astropy-quantity
            y coordinate of the ring center
        """   
         
        x = Quantity(x).decompose()
        y = Quantity(y).decompose()
        assert x.unit == y.unit
        unit = x.unit
        x = x.value
        y = y.value

        #Initial guess: data set centroid and std deviation from mean
        pos = np.empty(x.shape + (2,))
        pos[...,0] = x
        pos[...,1] = y 
        centroid = np.average(pos, axis=0, weights=image)

        rad = np.average(np.sqrt(image*(x-centroid[0])**2+image*(y-centroid[1])**2))
        start = (rad,centroid[0],centroid[1])
        
        result = minimize(
            self.taubin_func,
            x0=start,
            args=(x,y,np.nonzero(image)[0]),
            method='L-BFGS-B',
            bounds=[(0.,0.15),(None,None),(None,None)],
            #callback=draw_circle
            )
        if not result.success:
            result.x = np.full_like(result.x, np.nan)

        return result.x * unit

    def majority_func(self,image):
        """
        Returns the number of triggered (super)pixels
        """
        return image.sum()


def Trainer(filename,
            muon_data,
            shower_data,
            Nmax_muon=500,
            Nmax_shower=500,
            spfile="CHEC_pixel_dictionary.dat"):
    """
    Train the MLP
    Save the trained classifier as .pkl file

    Parameters
    ----------
    filename: str
        where to save the trained classifier
    muon_data : str
        muon training data filepath
    shower_data : str
        shower training data filepath
    Nmax_muon : int
        maximum number of muon samples
    Nmax_shower : int
        maximum number of shower samples 
    spfile : str
        filepath of the CHEC superpixel dictionary in .dat
        i.e. the corresponding superpixel for each pixel
        necessary for trigger pattern readout   
    """

    
    #Get data    
    chec_dic = np.loadtxt(spfile,dtype=int)
    X = [] #Test sample images
    y = [] #Test sample labels
       
    #Muon sample preparation
    #Note: muon label = 0    
    with event_source(muon_data, max_events=Nmax_muon) as source:  
        for event in source:
            if len(event.mc.tel)>1:
                continue                
            for telid in event.r0.tels_with_data:
                geom = event.inst.subarray.tel[telid].camera
                if geom.cam_id != 'CHEC':
                    print("Warning: different non CHEC camera event!")
                    continue
                trig_info = np.zeros(512)
                trig_info[chec_dic[event.r0.tel[telid].trig_pix_id]] = 1.0
                X.append(trig_info)
                y.append(0)
                
    #Shower sample preparation
    #Note: shower label = 1    
    with event_source(shower_data, max_events=Nmax_shower) as source:
        for event in source:
            if len(event.mc.tel)>1:
                continue                
            for telid in event.r0.tels_with_data:
                geom = event.inst.subarray.tel[telid].camera
                if geom.cam_id != 'CHEC':
                    print("Warning: different non CHEC camera event!")
                    continue
                trig_info = np.zeros(512)
                trig_info[chec_dic[event.r0.tel[telid].trig_pix_id]] = 1.0
                X.append(trig_info)
                y.append(1)

    #Train the classifier
    clf = MLPClassifier()
    clf.fit(X,y)
    joblib.dump(clf,filename)

