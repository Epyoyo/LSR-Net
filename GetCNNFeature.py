import numpy as np
import os

def walkFile(file):
    fileList = []
    for root, dirs, files in os.walk(file):
        for f in files:
            fileList.append(os.path.join(root, f))
        return fileList

fileList1=walkFile("Normal")
fileList2=walkFile("Crackle")
fileList3=walkFile("Wheeze")
fileList4=walkFile("Crackle_Wheeze")

import librosa
def get_feature(filelist):
    features=[]
    for i in filelist:
        wav, sample_rate = librosa.load(i, sr=19063)
        mel_spec = librosa.feature.melspectrogram(wav, sample_rate, n_fft=2048, hop_length=512, n_mels=224)
        logmel_spec = librosa.power_to_db(mel_spec)
        logmel_spec=logmel_spec.T
        logmel_spec=np.reshape(logmel_spec,(224,224,1))
        logmel_spec=np.concatenate((logmel_spec,logmel_spec,logmel_spec),axis=-1)
        features.append(logmel_spec)
    return features

from threading import Thread
class MyThread2(Thread):
    def __init__(self,fileListName):
        Thread.__init__(self)
        self.fileList=fileListName

    def run(self):
        self.result = get_feature(self.fileList)

    def get_result(self):
        return self.result

Mythd1 = MyThread2(fileList1)
Mythd2 = MyThread2(fileList2)
Mythd3 = MyThread2(fileList3)
Mythd4 = MyThread2(fileList4)

Mythd1.start()
Mythd2.start()
Mythd3.start()
Mythd4.start()

Mythd1.join()
Mythd2.join()
Mythd3.join()
Mythd4.join()

features1=Mythd1.get_result()
features2=Mythd2.get_result()
features3=Mythd3.get_result()
features4=Mythd4.get_result()

np.save('Feature_CNN_Normal.npy',features1)
np.save('Feature_CNN_Crackle.npy',features2)
np.save('Feature_CNN_Wheeze.npy',features3)
np.save('Feature_CNN_Crackle&Wheeze.npy',features4)
