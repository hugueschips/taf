'''
Created on 13th may 2016
@author: Daniel Durrenberger
daniel.durrenberger@amaris.com

Python 2.7.12
Pandas 0.18.1
Numpy 1.11.1
Scipy 0.18.0
'''

import os
import numpy as np
import pandas as pd
import scipy
import os.path
import preprocessing as pp

# Default storage location
insp_excel = (
            '/Users/Hans/Documents/ArcelorMittal/'
            'Data/early_2015/data_inspection_excel/'
            'inspection_early2015_w_sound.xlsx'
            )
sound_directory = (
            '/Users/Hans/Documents/ArcelorMittal/'
            'Data/early_2015/data_sound_txt/'
            )

def import_excelfile(inspection_file=insp_excel, sheet_name='Sheet1'):
    '''
    inspection_file = '/Users/Hans/Documents/ArcelorMittal/
                        Data/early_2015/data_inspection_excel/
                        inspection_early2015_w_sound.xlsx'
    sheet_name = 'Sheet1'
    '''
    df = pd.read_excel(inspection_file, sheetname=sheet_name)
    return df

def export_excelfile(dfs):
    writer = pd.ExcelWriter('new_file.xlsx', engine='xlsxwriter')
    dfs.to_excel(writer, sheet_name='Sheet1')
    writer.save()
    return()

def clean_inspection_file(df, sound_directory):
    dfi = df.drop_duplicates('coil_number')
    for coilNumber in dfi.index:
        coilName = dfi.coilNumber[coilNumber]
        file_name = sound_directory+str(coilNumber)+'_all.txt'
        if os.path.isfile(file_name):
            dfi.set_value(coilNumber,'sound_file', True)
        else:
            dfi.set_value(coilNumber,'sound_file', False)
    dfs = dfi[dfi.sound_file==True]
    n_saved_lines = len(df)-len(dfs)
    return dfi

def import_txt_sound(
                    coil,
                    dfi=None,
                    sound_directory=sound_directory
                    ):
    # import sound files
    # sound files have to be exported from d7d to txt format with Dewesoft
    # and saved in sound_directory
    if dfi==None:
        dfi = import_excelfile()
    col_names = ['Time', 'Micro', 'Speed', 'Distance', 'Decoiler']
    i = coil
    #print('Importing coil number '+str(i))
    coilnumber = dfi.coil_number[i]
    filename = sound_directory+str(coilnumber)+'_all.txt'
    df = pd.read_csv(
                        filename,
                        sep='\t',
                        names=col_names,
                        skiprows=0,
                        header=0
                        )
    duration = df.Time.values[-1]-df.Time.values[0]
    fs = (len(df)-1)/duration
    dfi.set_value(i, 'duration', duration)
    dfi.set_value(i, 'fs', fs)
    dfi.set_value(i, 'n', len(df))
    return df, dfi

def store_dfi(dfi, hdf=None):
    if hdf==None:
        hdf = (
                '/Users/Hans/Documents/ArcelorMittal/'
                'Data/early_2015/hdf/sound.hdf5'
                )
    store = pd.HDFStore(hdf)
    store['inspection'] = dfi
    store.close()
    return hdf

def store_sound(
                coil,
                hdf=None,
                folder='sound',
                sound_directory=sound_directory
                ):
    if hdf==None:
        hdf = (
                '/Users/Hans/Documents/ArcelorMittal/'
                'Data/early_2015/hdf/sound.hdf5'
                )
    store = pd.HDFStore(hdf)
    dfi = store['inspection']
    number = dfi.coil_number[coil]
    directory = folder+'/coil_'+str(number)
    df, dfii = import_txt_sound(coil, dfi, sound_directory)
    store[directory] = df
    store['inspection'] = dfii
    store.close()
    return hdf

def import_txt_sound_old(
                    coil,
                    dfi=None,
                    sound_directory=sound_directory,
                    resample=10000
                    ):
    # import sound files
    # sound files have to be exported from d7d to txt format with Dewesoft
    # and saved in sound_directory
    '''
        sound_directory = '/Users/Hans/Documents/ArcelorMittal/
                            Data/early_2015/data_sound_txt/'
    '''
    if dfi==None:
        dfi = import_excelfile()
    col_names = ['Time', 'Micro', 'Speed', 'Distance', 'Decoiler']
    store = pd.HDFStore('early_2015_'+str(int(resample/1000))+'kHz.h5')
    i = coil
    print('Importing coil number '+str(i))
    coilnumber = dfi.coil_number[i]
    filename = sound_directory+str(coilnumber)+'_all.txt'
    dfs = pd.read_csv(
                        filename,
                        sep='\t',
                        names=col_names,
                        skiprows=0,
                        header=0
                        )
    duration = dfs.Time.values[-1]-dfs.Time.values[0]
    fsraw = (len(dfs)-1)/duration
    resample_factor = max(int(fs/resample),1)
    fs = fsraw/resample_factor
    inspection_df.set_value(i, 'duration', duration)
    inspection_df.set_value(i, 'fs', fs)
    dfr = dfs[::resample_factor]              # resample from 10kHz to 5kHz
    dfr.index = range(len(dfr))                                   # reindex
    store['sound/coil_'+str(coilnumber)] = dfr
    store['inspection'] = inspection_df
    store.close()
    return dfs, fsraw

def clean_df(df, l):
    '''
        interpolate speed, coiler and decoiler signal because of their
        undersamplerate compared to sound signal.
        Due to better samplerate, decoiler signal is ignored and replace by
        speed signal integration initialized by coil length value available in
        inspection file.
    '''
    time = df.Time.values
    a, b, f, fs = pp.sample_and_interpolate(time, df.Distance.values)
    coiler = f(time)
    a, b, f, fs = pp.sample_and_interpolate(time, df.Speed.values)
    speed = f(time)                                      # speed stays in m/min
    samplerate = (int(len(df))-1)/(time[-1]-time[0])
    dt = 1./samplerate
    decoiler = l-np.add.accumulate(dt*speed/60)       # /60 to put speed in m/s
    dfout = pd.DataFrame({
                        'time':time,
                        'micro':df.Micro.values,
                        'rms':pp.fast_rms(
                                        df.Micro.values,
                                        window=5.,
                                        samplerate=samplerate
                                        ),
                        'speed':speed,
                        'decoiler':decoiler,
                        'coiler':coiler
                        })
    dfout = dfout[['time', 'micro', 'rms', 'speed', 'decoiler', 'coiler']]
    return dfout

def max_speed_seq(speed, samplerate=10000, precision=5):
    '''
        returns start index and end index where max speed is reached
        precision is in seconds and samplerate in Hz
    '''
    pas = int(samplerate//precision)
    dt = 1./samplerate
    max_speed = int(speed.max())
    max_speed_min = max_speed * 0.95
    max_speed_start = 0
    i = 0
    # looking for start
    while (max_speed_start==0) & (i<len(speed)):
        if speed[i]<max_speed_min:
            i += pas
        else:
            max_speed_start = i
    # looking for end
    max_speed_end = len(speed)
    i = max_speed_start
    while (max_speed_end==len(speed)) & (i<len(speed)):
        if speed[i]>max_speed_min:
            i += pas
        else:
            max_speed_end = i
    # crop 4s at start and end to avoid non-constant speed
    max_speed_start += 4*samplerate
    max_speed_end -= 4*samplerate
    return max_speed_start, max_speed_start+max_speed_end

def status(i):
    dfi = import_dfi()
    if (dfi.sticking[i]==1):
        status = 'Sticking'
    else:
        status = 'Non sticking'
    return status

def files(
        hdfdescription='complete',
        data='inspection'
        ):
    directory = '/Users/Hans/Documents/ArcelorMittal/Data/early_2015/hdf/'
    available_hdf = [
                    'complete',
                    'ang',
                    'emd'
                    ]
    files = [
            'early_2015_10kHz_interpol.h5',
            'early_2015_ang.h5',
            'emd_30.h5'
            ]
    available_data = [
                    'inspection',
                    'emd_report',
                    'sound',
                    'ang',
                    'emd'
                    ]
    folders = [
                'inspection',
                'emd_report',
                'sound',
                'normalized_sound',
                'emd'
                ]
    # hdf
    for i in range(len(available_hdf)):
        if hdfdescription==available_hdf[i]:
            filename = files[i]
            break
    hdf = directory+filename
    # folder
    for i in range(len(available_data)):
        if data==available_data[i]:
            folder = folders[i]
    return hdf, folder

def import_dfi(hdfdescription='complete', dfi='inspection'):
    hdf, folder = files(hdfdescription=hdfdescription, data=dfi)
    store = pd.HDFStore(hdf)
    dfi = store[folder]
    store.close()
    return dfi

def import_data(coil=28, hdfdescription='complete', data='sound'):
    dfi = import_dfi()
    coil_number = dfi.coil_number[coil]
    hdf, folder = files(hdfdescription=hdfdescription, data=data)
    store = pd.HDFStore(hdf)
    df = store[folder+'/coil_'+str(coil_number)]
    store.close()
    return df

def dfToArrays(df, i0=0, iN=-1):
    i0, iN = int(i0), int(iN)
    time = df.time.values[i0:iN]
    signal = df.micro.values[i0:iN]
    speed = df.speed.values[i0:iN]
    decoiler = df.decoiler.values[i0:iN]
    coiler = df.coiler.values[i0:iN]
    coiler = coiler-coiler[0]
    decoiler = decoiler-decoiler[-1]
    return time, signal, speed, decoiler, coiler

def xInfo(x):
    a, b = x[0], x[-1]
    n = len(x)
    dx = (b-a)/(n-1)
    fs = int(1//dx)+1
    return a, b, n, dx, fs

def write_wav(coil):
    import scipy.io.wavfile
    if coil<10:
        name = 'coil0'
    else:
        name = 'coil'
    df = md.import_data(coil)
    a, b, n, dt, fs = xInfo(df.time.values)
    scipy.io.wavfile.write(name+str(i)+'.wav', fs, df.micro.values)

def store_peaks(
                xpeak_imf,
                ypeak_imf,
                coil,
                filename='peaks.h5',
                path='./output/'
                ):
    if not os.path.exists(path):
        print(path+' has been created')
        os.makedirs(path)
    d = {}
    i = 0
    for xpeak, ypeak in zip(xpeak_imf, ypeak_imf):
        namex = 'Ximf'+str(i)
        namey = 'Yimf'+str(i)
        i += 1
        d[namex] = xpeak
        d[namey] = ypeak
    df = pd.DataFrame(d)
    store = pd.HDFStore(path+filename)
    store['coil_'+str(coil)] = df
    store.close()
    return df, path+filename
