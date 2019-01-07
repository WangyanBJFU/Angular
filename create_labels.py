import os

dirName = '/home/wy/all_data/data_c863_backup/MFCCresults'
resultsDirectory = dirName + "/labels"
def create_labels():
    if not os.path.exists(resultsDirectory):
        os.makedirs(resultsDirectory)
    mfccs = []
    labels = []

    for filename in os.listdir(dirName):
        if filename.endswith('.csv'):
            mfccs.append(filename)
            labels.append(filename.split("/")[-1][:3])
            # outputFile = resultsDirectort + "/"
            # file = open(outputFile)
    return mfccs, labels
'''
with open('/home/wy/all_data/data_c863_backup/MFCCresults/c863.data', 'w') as file:
    humans = os.listdir(dirName)
    # print(humans)
    index = 0
    # for human in humans[:76]:    # humans不约束范围的话总数是166
    for human in humans[:166]:
        path = os.path.join(root, human)
        waves = os.listdir(path)
        # print(waves)
        index += 1
        # count 166 speakers, index represents their representation, called 1-166.

        # 166 folders
        # total of 3320 items

        print(index)
        for wave in waves:
            path_wav = os.path.join(path, wave)
            file.writelines(path_wav + " %d\n" % index)
    print("data url has been produced")

'''
