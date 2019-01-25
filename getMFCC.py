from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
import sys
from cmvn import cmvnOperation
# from python_speech_features.create_labels import create_labels
# from python_speech_features.concatenate import concatenate
# from python_speech_features.cmvn import cmvnOperation
import scipy.io.wavfile as wav
import numpy as np
import os

# directory where we your .wav files are
directoryName = "/home/wangyan/all_data/data_c863_backup"  # put your own directory here
# directory to put our results in, you can change the name if you like
resultsDirectory = directoryName + "/MFCCresults"

# make a new folder in this directory to save our results in
if not os.path.exists(resultsDirectory):
    os.makedirs(resultsDirectory)

# get MFCCs for every .wav file in our specified directory
def getMFCC():
    mfccs = []
    mfccs_list = []
    for filename in os.listdir(directoryName):
        print(filename)
        if filename.endswith('.wav'): # only get MFCCs from .wavs
            # read in our file
            (rate,sig) = wav.read(directoryName + "/" +filename)

            # get mfcc
            mfcc_feat = mfcc(sig,rate)
            d_mfcc_feat = delta(mfcc_feat, 2)
            dd_mfcc_feat = delta(d_mfcc_feat, 2)
            # get filterbank energies
            fbank_feat = logfbank(sig,rate)

            # get mfcc+dmfcc+ddmfcc = 39d
            mfcc_39d = np.hstack([mfcc_feat, np.hstack([d_mfcc_feat, dd_mfcc_feat])])
            mfcc_39d = cmvnOperation(mfcc_39d)
            # mfcc_39d = concatenate(mfcc_39d)
            # print(mfcc_feat.shape)
            # print(d_mfcc_feat.shape)
            # print(dd_mfcc_feat.shape)
            print(mfcc_39d.shape)
            for i in mfcc_39d:
                mfccs.append(i)
            mfccs_list.append(mfcc_39d)
            # create a file to save our results in
            outputFile = resultsDirectory + "/" + os.path.splitext(filename)[0] + ".csv"
            file = open(outputFile, 'w+') # make file/over write existing file
            np.savetxt(file, mfcc_39d, delimiter=",") #save MFCCs as .csv
            file.close() # close file
    print("------Extracted done------")
    outputFile_2 = resultsDirectory + "/mfccs.csv"
    file = open(outputFile_2, 'w+') # make file/over write existing file
    np.savetxt(file, mfccs, delimiter=",")
    print("----------mfccs have been saved----------")

    '''
    return中间等待5分钟左右，生成文件1.7G，需要等待打印出的内容 
    '''
    print("---------Please wait,specific value of mfcc is printing---------")
    return mfccs, mfccs_list

if __name__ == "__main__":
    mfccs, mfccs_list = getMFCC()
    print("mfccs is :", mfccs, "/n" + "mfccs_list is :", mfccs_list)
    print("mfccs' length is: ", len(mfccs))
    print("mfccs_list's length is: ", len(mfccs_list))
    # _, b = create_labels()
    # print("b.shape is :", b.shape)
    # print("b is :", b)
    # 最后输出结果：
    # mfccs' length is:  1428082    
    # mfccs_list's length is:  3320

