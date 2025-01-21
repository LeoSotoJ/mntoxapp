import os
import pandas as pd
import matchms

def count_spectrums(spectrums):
   
    unique_inchikeys=set([s.get('inchikey') for s in spectrums])
    print('Total number of spectra:',len(spectrums))
    print('Total number of unique chemicals:',len(unique_inchikeys))
    print('Average number of spectrums per unique chemical:',len(spectrums)/len(unique_inchikeys))

#Shows the index, if any, in the spectrums list for an specified inchikey
def index_inchikey(spectrums, inchikey):
    
    indices = [i for i, s in enumerate(spectrums) if s.get('inchikey') == inchikey]
    for index in indices:
        print("index:", index)
    return index

def index_id(spectrums, id):
    
    indices = [i for i, s in enumerate(spectrums) if s.get('id') == id]
    for index in indices:
        print("index:", index)
    return index