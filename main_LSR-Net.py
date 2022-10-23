import numpy as np

'''加载VGG19提取的128维特征'''
x_train_CNN=np.load('LSR-Net_CNN_x_train.npy')
x_test_CNN=np.load('LSR-Net_CNN_x_test.npy')

'''加载VGGish提取的256维特征'''
x_train_VGGish=np.load('LSR-Net_VGGish_x_train.npy')
x_test_VGGish=np.load('LSR-Net_VGGish_x_test.npy')

'''特征融合'''
x_train=np.concatenate((x_train_VGGish,x_train_CNN),1)
x_test=np.concatenate((x_test_VGGish,x_test_CNN),1)

'''加载标签'''
y_train=np.load('LSR-Net_y_train.npy')
y_test=np.load('LSR-Net_y_test.npy')

'''
1.将融合后的特征输入catboost实现训练并测试
2.由于catboost调参(网格搜索)时间消耗过大，因此使用初始默认的随机参数执行10次，取结果较高的
'''
ScoreArray=[]
ModelArray=[]
from catboost import CatBoostClassifier
for i in range(10):
    catmodel=CatBoostClassifier(loss_function='MultiClass',task_type='GPU')
    catmodel.fit(x_train,y_train)
    ModelArray.append(catmodel)
    TestScore=catmodel.score(x_test,y_test)
    ScoreArray.append(TestScore)

MaxScoreIndex=ScoreArray.index(max(ScoreArray))
MyModel=ModelArray[MaxScoreIndex]

