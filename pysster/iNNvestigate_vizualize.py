import pandas as pd
import numpy as np
from janggu_layers import Reverse,Complement

from pysster.Data import Data
import pysster.utils as io

import innvestigate
#import innvestigate.utils as iutils
#import innvestigate.utils.tests.networks.imagenet
#import innvestigate.utils.visualizations as ivis

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



import utils_imagenet as imgnetutils

from keras.models import load_model, Model
from keras import backend as K

import vizsequence
from vizsequence import viz_sequence

import pickle
import numpy as np


    # Create analyzers.
#    patterns = net["patterns"]
methods = [
    # NAME                      OPT.PARAMS                POSTPROC FXN                TITLE

    # Show input.
#    ("input",                   {},                       imgnetutils.image,   "Input"),

    # Function
#    ("gradient",                {},                       imgnetutils.graymap, "Gradient"),
    ("integrated_gradients",    {},                       imgnetutils.graymap, "Integrated Gradients"),

    # Signal
#    ("deconvnet",               {},                       imgnetutils.bk_proj, "Deconvnet"),
#    ("guided_backprop",         {},                       imgnetutils.bk_proj, "Guided Backprop"),
    #("pattern.net",             {"patterns": patterns},   imgnetutils.bk_proj, "PatterNet"),

    # Interaction
    #("pattern.attribution",     {"patterns": patterns},   imgnetutils.heatmap, "Pattern Attribution"),


    ("lrp.epsilon",             {"epsilon": 1},           imgnetutils.heatmap, "LRP Epsilon"),


    #("lrp.sequential_preset_a_flat", {"epsilon": 1},      imgnetutils.heatmap, "LRP-PresetAFlat"),
    #("lrp.sequential_preset_b_flat", {"epsilon": 1},      imgnetutils.heatmap, "LRP-PresetBFlat"),
]
    

numToPlot = 1
numToPlot = 5
names=["artificial"]
numSplits=100
#names=["lung"]
for name in names:
    #load data
    SCORE=np.load(open(name+"_score.npy",'rb'))
    data=io.load_data("../"+name+"/data.pkl")
    SEQ,IF=data._get_data("test")           
    plt.rcParams["figure.figsize"] = (200, 2*numToPlot*len(methods))
 
    analysis=[]
    for m in methods:
        analysis.append(np.load(open(name+"_"+m[0]+"_iNN.npy",'rb')))


#    fig, ax = plt.subplots(numToPlot*len(analyzers),1)
    for cl in range(len(SCORE[0])):
        split = pd.qcut(SCORE[:,cl], numSplits, labels=False)
        
        fig, ax = plt.subplots(numToPlot*len(methods),1)
        for j in range(0, len(methods)):
            analysisTop=analysis[j][split == (numSplits-1)]
            analysisBot=analysis[j][split == 0]
    
            for i in range(0, numToPlot):
                # plot best quartile
                ind=i+j*numToPlot
                viz_sequence.plot_weights_given_ax(ax[ind],analysisTop[i], subticks_frequency=10,highlight={},height_padding_factor=0.2,length_padding=1.0)
                ax[ind].set_title("example class "+str(cl),fontsize='small')
    
            ax[j*numToPlot].set_title(methods[j][3]+" example class "+str(cl),fontsize='small')
            fig.tight_layout()
            fig.subplots_adjust(wspace=0, hspace=0.01)
            
        plt.savefig(name+"_iNN_viz_class"+str(cl)+".png")
        plt.close()
       
        print("done class "+str(cl))
