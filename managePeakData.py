# -*- coding: utf-8 -*-
'''
Created on 5th September 2016
@author: Daniel Durrenberger
daniel.durrenberger@amaris.com

Python 2.7.12
Pandas 0.18.1
Numpy 1.11.1
Scipy 0.18.0
'''

import numpy as np
import pandas as pd

def importPeak(coil=28, filename='peaks.h5', path='./output/'):
    store = pd.HDFStore(path+filename)
    df = store['coil_'+str(coil)]
    store.close()
    return df
