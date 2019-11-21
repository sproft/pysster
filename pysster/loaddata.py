import numpy as np
from sklearn import preprocessing
import joblib
import os

datadir=os.environ["DREAM_DATA"]+"/"

def loadDna(batchid=None,dataset="train"):
    assert batchid==None or (batchid>=0 and batchid<20), "loadDna: wrong batchid"
    if batchid==None:
        dna=joblib.load(datadir + dataset+"_input/dna/dna.pkl")
        dna=np.unpackbits(dna,axis=3).astype("int8")
    else:
        dna=joblib.load(datadir + dataset+"_input/dna/dna."+str(batchid)+".pkl").astype("int8")
    return dna

def loadDhs(batchid=None,dataset="train"):
    assert batchid==None or (batchid>=0 and batchid<20), "loadDhs: wrong batchid"
    if batchid==None:
        dhs=joblib.load(datadir + dataset+"_input/dhs/dhs.pkl")
    else:
        dhs=joblib.load(datadir + dataset+"_input/dhs/dhs."+str(batchid)+".pkl")
    return dhs

def loadDhsSum(batchid=None,dataset="train"):
    assert batchid==None or (batchid>=0 and batchid<20), "loadDhs: wrong batchid"
    if batchid==None:
        dhs=joblib.load(datadir + dataset+"_input/dhs-sum/dhs.pkl")
    else:
        dhs=joblib.load(datadir + dataset+"_input/dhs-sum/dhs."+str(batchid)+".pkl")
    pseudocount = 0.1
    return np.log2(dhs + pseudocount)

def loadRna():
    rna=joblib.load(datadir + "train_input/rna/rna.pkl")
    return rna

def loadRnaPerPosition(dataset):
    rna=joblib.load(datadir + "train_input/rna/neighbouring_rnaseq.pkl")
    return rna

def loadTfRna():
    rna=joblib.load(datadir + "train_input/rna/tfrna.pkl")
    return rna

def loadLoc2Gene(batchid=None,dist=1000,dataset="train"):
    assert batchid==None or (batchid>=0 and batchid<20), "loadDhs: wrong batchid"
    if batchid==None:
        l2g=joblib.load(datadir + dataset+"_input/loc2gene/loc2gene."+str(dist)+".pkl")
    else:
        l2g=joblib.load(datadir + dataset+"_input/loc2gene/loc2gene."+str(dist)+"."+str(batchid)+".pkl")
    return l2g

def loadGene2Chip(dataset="train"):
    return joblib.load(datadir + dataset+"_input/gene2chip/gene2chip.pkl")

def loadChip(batchid=None,threshold="conservative",dataset="train"):
    assert batchid==None or (batchid>=0 and batchid<20), "loadDhs: wrong batchid"
    if batchid==None:
        chip=joblib.load(datadir + dataset+"_input/chip/chip."+threshold+".pkl")
    else:
        chip=joblib.load(datadir + dataset+"_input/chip/chip."+threshold+"."+str(batchid)+".pkl")
    return chip

def loadChipWithin(batchid=None,threshold="conservative",dataset="train"):
    assert batchid==None or (batchid>=0 and batchid<20), "loadDhs: wrong batchid"
    if batchid==None:
        chip=joblib.load(datadir + dataset+"_input/chip/chip.within."+threshold+".pkl")
    else:
        chip=joblib.load(datadir + dataset+"_input/chip/chip.within."+threshold+"."+str(batchid)+".pkl")
    return chip

def loadAggrChip(batchid=None,threshold="conservative",dataset="train"):
    assert batchid==None or (batchid>=0 and batchid<20), "loadDhs: wrong batchid"
    if batchid==None:
        chip=joblib.load(datadir + dataset+"_input/chip/chip.aggregate."+threshold+".pkl")
    else:
        chip=joblib.load(datadir + dataset+"_input/chip/chip.aggregate."+threshold+"."+str(batchid)+".pkl")
    return chip

