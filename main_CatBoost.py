import numpy as np

'''加载VGG19提取的128维特征'''
x_train_CNN=np.load('LSR-Net_CNN_x_train.npy')
x_test_CNN=np.load('LSR-Net_CNN_x_test.npy')

'''加载VGGish提取的256维特征'''
x_train_VGGish=np.load('LSR-Net_VGGish_x_train.npy')
x_test_VGGish=np.load('LSR-Net_VGGish_x_test.npy')

'''将两部分特征融合为384维特征'''
x_train=np.concatenate((x_train_VGGish,x_train_CNN),1)
x_test=np.concatenate((x_test_VGGish,x_test_CNN),1)

'''加载标签'''
y_train=np.load('LSR-Net_y_train.npy')
y_test=np.load('LSR-Net_y_test.npy')

'''
1.将融合后的特征输入catboost实现训练并测试
2.由于catboost调参(网格搜索)时间消耗过大，这里没有采用网格搜索的方式。直接使用初始的随机参数执行10次，取结果较高的一次。
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

# 绘制混淆矩阵
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
labels_Name = ['正常肺音', '爆裂音', '哮鸣音', '混合肺音']
MyModel=ModelArray[MaxScoreIndex]
y_pred=MyModel.predict((x_test2))
print(y_pred)
y_true=y_test2
print(y_true)
tick_marks = np.array(range(len(labels_Name))) + 0.5
def plot_confusion_matrix(cm, title='Confusion Matrix', cmap = plt.cm.Greys):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号 #有中文出现的情况，需要u'内容'
    font2 = {'size': 18}
    plt.title(title,font2)
    plt.colorbar()
    xlocations = np.array(range(len(labels_Name)))
    plt.tick_params(labelsize=15)
    plt.rc('font', family='Times New Roman')
    plt.xticks(xlocations, labels_Name)
    plt.yticks(xlocations, labels_Name)
    plt.ylabel('真实标签',font2)
    plt.xlabel('预测标签',font2)
    # plt.ylabel('True label',font2)
    # plt.xlabel('Predicted label',font2)

cm = confusion_matrix(y_true, y_pred)
np.set_printoptions(precision=2)
#cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
cm_normalized = cm
#print cm_normalized
print(cm_normalized)
plt.figure(figsize=(6, 6), dpi=150)
# plt.figure()
ind_array = np.arange(len(labels_Name))
x, y = np.meshgrid(ind_array, ind_array)
for x_val, y_val in zip(x.flatten(), y.flatten()):
    c = cm_normalized[y_val][x_val]
    if c > 0.01:
        if x_val==0 and y_val==0:
            plt.text(x_val, y_val, "%d" % (c,), color='white', fontsize=30, va='center', ha='center')
        else:
            plt.text(x_val, y_val, "%d" % (c,), color='red', fontsize=30, va='center', ha='center')
# offset the tick
plt.gca().set_xticks(tick_marks, minor=True)
plt.gca().set_yticks(tick_marks, minor=True)
plt.gca().xaxis.set_ticks_position('none')
plt.gca().yaxis.set_ticks_position('none')
plt.grid(True, which='minor', linestyle='-')
plt.gcf().subplots_adjust(bottom=0.15)
plot_confusion_matrix(cm_normalized, title='混淆矩阵')
plt.show()

