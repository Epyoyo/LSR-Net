import numpy as np
import tensorflow as tf

gpus=tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu,True)

'''
1.将肺音波形数据转换为(N/0.96,96,64,1)的对数梅尔频率谱 (这里按照Google提供的转换方式及参数)，然后保存在.npy文件中
2.加载.npy文件 即 加载(N/0.96,96,64,1)的对数梅尔频率特征
'''
normal_feature=np.load('Feature_VGGish_Normal.npy')
crackle_feature=np.load('Feature_VGGish_Crackle.npy')
wheeze_feature=np.load('Feature_VGGish_Wheeze.npy')
mix_feature=np.load('Feature_VGGish_Crackle&Wheeze.npy')
features=np.concatenate([normal_feature,crackle_feature,wheeze_feature,mix_feature],axis=0)

'''制作类别标签'''
normal_label=[0]*len(normal_feature)
crackle_label=[1]*len(crackle_feature)
wheeze_label=[2]*len(wheeze_feature)
mix_label=[3]*len(mix_feature)
normal_label=np.array(normal_label)
crackle_label=np.array(crackle_label)
wheeze_label=np.array(wheeze_label)
mix_label=np.array(mix_label)
labels=np.concatenate([normal_label,crackle_label,wheeze_label,mix_label],axis=0)

'''按照8：2划分训练集与测试集，这里的随机数种子务必与main_VGG19中的相同'''
np.random.seed(7)
np.random.shuffle(features)
np.random.seed(7)
np.random.shuffle(labels)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(features,labels,test_size=0.2,random_state=11,stratify=labels)
from tensorflow.keras.utils import to_categorical
y_train1= to_categorical(y_train)
y_test1=to_categorical(y_test)
x_train1=np.reshape(x_train,(len(x_train),6,96,64,1))
x_test1=np.reshape(x_test,(len(x_test),6,96,64,1))

from tensorflow.keras.layers import TimeDistributed, Bidirectional, GRU,Flatten, Dense,Reshape
from tensorflow.keras import optimizers,Model
from tensorflow.keras.models import Sequential
import vggish_params

'''Build_New_VGGish函数的功能：在预训练模型VGGish中融合ECA-Net'''
from BuildVGGish import Build_New_VGGish
'''Feature_Aug函数的功能:利用多尺度空洞卷积对VGGish网络提取的特征向量进行运算，实现特征增强'''
from FeatureAug import Feature_Aug

'''构建VGGish优化模型'''
vggish_model2=Build_New_VGGish()
model = Sequential()
'''利用TimeDistributed函数实现让VGGish按照时序学习对数梅尔频率谱'''
model.add(TimeDistributed(vggish_model2, input_shape=(6, vggish_params.NUM_FRAMES, vggish_params.NUM_BANDS, 1)))
model.add(Reshape((6,128,1)))
model.add(Feature_Aug())
model.add(Bidirectional(GRU(128,return_sequences=False),name='out_layer2'))
model.add(Flatten())
model.add(Dense(4, activation='softmax'))
model.summary()

'''训练VGGish优化模型'''
opt = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.00, amsgrad=False)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['acc'])
history = model.fit(x_train1, y_train1, epochs=10, batch_size=16, validation_split=0.1)

'''将VGGish优化模型当作特征提取器,提取高层的256维特征向量'''
permute_layer_model2 = Model(model.input,model.get_layer('out_layer2').output)
permute_layer_output_xtrain = permute_layer_model2.predict((x_train1))
permute_layer_output_xtrain=np.reshape(permute_layer_output_xtrain,(len(permute_layer_output_xtrain),256))
permute_layer_output_xtest=permute_layer_model2.predict((x_test1))
permute_layer_output_xtest=np.reshape(permute_layer_output_xtest,(len(permute_layer_output_xtest),256))

np.save("LSR-Net_VGGish_x_train.npy",permute_layer_output_xtrain)
np.save("LSR-Net_VGGish_x_test.npy",permute_layer_output_xtest)
np.save("LSR-Net_y_train.npy",y_train)
np.save("LSR-Net_y_test.npy",y_test)









