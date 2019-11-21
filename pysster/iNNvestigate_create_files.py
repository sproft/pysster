from os import remove
from tempfile import gettempdir
import gzip
import pickle
import pandas as pd
import numpy as np
from janggu_layers import Reverse,Complement

from pysster.Data import Data
import pysster.utils as io
import keras 

import innvestigate

import innvestigate.utils as iutils
#import innvestigate.utils.tests.networks.imagenet
#import innvestigate.utils.visualizations as ivis

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



import utils_imagenet as imgnetutils

from keras.models import load_model, Model
from keras import backend as K
import keras

import vizsequence
from vizsequence import viz_sequence


def change_activation(model):
    print("changing activation...")
    custom_objects={'Reverse':Reverse,'Complement':Complement}
    path = "{}/temp_model_file".format(gettempdir())
    model.save(path, overwrite = True)
    model = load_model(path, custom_objects=custom_objects)
    model.layers[-1].activation = keras.activations.linear
    model.save(path, overwrite = True)
    K.clear_session()
    model = load_model(path, custom_objects=custom_objects)
    model.layers[0].name="in_dna"
    remove(path)
    return model

def create_lists(analyzers,name,SEQ,methods,SCORE):
    np.save(open(name+"_score.npy","wb"), SCORE)
    analysis=np.zeros(shape=SEQ.shape)
    print(analysis.shape)
    for j in range(0, len(analyzers)):
        for i,seq in enumerate(SEQ):
            analysis[i]=(analyzers[j].analyze(np.expand_dims(seq,axis=0)))
        print(analysis.shape)
        np.save(open(name+"_"+methods[j][0]+"_iNN.npy","wb"),analysis)
            

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
    
names=["artificial"]
#names=["lung"]
for name in names:

    #load data
    data=io.load_data(name+"/data.pkl")
    
    SEQ,IF=data._get_data("test")
    
    #load model
    custom_objects={'Reverse':Reverse, 'Complement':Complement}
    model=load_model(name+"/model.pkl.h5", custom_objects = custom_objects)
    SCORE=model.predict(SEQ, batch_size=1000, verbose=1)

    pattern_type = "relu"
    channels_first = K.image_data_format == "channels_first"
     
   # model_wo_softmax = iutils.keras.graph.model_wo_softmax(model)
    model=change_activation(model)

    # Create analyzers.
    print("analyse")
    analyzers = []
    for method in methods:
        try:
            analyzer = innvestigate.create_analyzer(method[0],
                                                    model,
                                                    **method[1])
        except innvestigate.NotAnalyzeableModelException:
            print("ERROR")
            analyzer = None
        analyzers.append(analyzer)
    create_lists(analyzers,name,SEQ,methods,SCORE)
    
