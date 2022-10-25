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

from vggish_input import wavfile_to_examples
def get_feature(filelist):
    features=[]
    for i in filelist:
        ret1=wavfile_to_examples(i)
        ret = np.array(ret1)
        ret=np.reshape(ret,(6,96,64,1))
        features.append(ret)
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

np.save('Feature_VGGish_Normal.npy',features1)
np.save('Feature_VGGish_Crackle.npy',features2)
np.save('Feature_VGGish_Wheeze.npy',features3)
np.save('Feature_VGGish_Crackle&Wheeze.npy',features4)
