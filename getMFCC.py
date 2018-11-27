from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
import scipy.io.wavfile as wav
import numpy as np
import os
from cmvn import cmvnOperation
from concatenate import concatenate
# directory where we your .wav files are
directoryName = "/home/wy/all_data/data_c863_backup"  # put your own directory here
# directory to put our results in, you can change the name if you like
resultsDirectory = directoryName + "/MFCCresults"

# make a new folder in this directory to save our results in
if not os.path.exists(resultsDirectory):
    os.makedirs(resultsDirectory)
mfccs =[]
# get MFCCs for every .wav file in our specified directory
def getMFCC():
    for filename in os.listdir(directoryName):
        if filename.endswith('.wav'): # only get MFCCs from .wavs
            # read in our file
            (rate,sig) = wav.read(directoryName + "/" +filename)
    
            # get mfcc
            mfcc_feat = mfcc(sig,rate)
            d_mfcc_feat = delta(mfcc_feat, 2)
            dd_mfcc_feat = delta(d_mfcc_feat, 2)
            # get filterbank energies
            # fbank_feat = logfbank(sig,rate)
    
            # get mfcc+dmfcc+ddmfcc = 39d
            mfcc_39d = np.hstack([mfcc_feat, np.hstack([d_mfcc_feat, dd_mfcc_feat])])
            mfcc_39d = cmvnOperation(mfcc_39d)
            # mfcc_39d = concatenate(mfcc_39d)
            # print(mfcc_feat.shape)
            # print(d_mfcc_feat.shape)
            # print(dd_mfcc_feat.shape)
            print(mfcc_39d.shape)
            mfccs.append(mfcc_39d)
            # create a file to save our results in
            outputFile = resultsDirectory + "/" + os.path.splitext(filename)[0] + ".csv"
            file = open(outputFile, 'w+') # make file/over write existing file
            np.savetxt(file, mfcc_39d, delimiter=",") #save MFCCs as .csv
            file.close() # close file
    print("------Extracted done------")
    print("mfccs' length is: ", len(mfccs))
    return mfccs

if __name__ == "__main__":
    a = getMFCC()
    print(a)

