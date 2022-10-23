import numpy as np
import tensorflow as tf

gpus=tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu,True)

'''
1.将肺音波形数据转换为(224,224,3)的对数梅尔频率谱，然后保存在.npy文件中
2.加载.npy文件 即 加载(224,224,3)的对数梅尔频率特征
'''
normal_feature=np.load('Feature_CNN_Normal.npy')
crackle_feature=np.load('Feature_CNN_Crackle.npy')
wheeze_feature=np.load('Feature_CNN_Wheeze.npy')
mix_feature=np.load('Feature_CNN_Crackle&Wheeze.npy')
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

'''按照8：2划分训练集与测试集，这里的随机数种子务必与main_VGGish(opt)中的相同'''
np.random.seed(7)
np.random.shuffle(features)
np.random.seed(7)
np.random.shuffle(labels)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(features,labels,test_size=0.2,random_state=11,stratify=labels)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import optimizers,Model
y_train1= to_categorical(y_train)
y_test1=to_categorical(y_test)
x_train1=np.reshape(x_train,(len(x_train),224,224,3))
x_test1=np.reshape(x_test,(len(x_test),224,224,3))

'''训练ImageNet上的预训练模型'''
from ECANet import ECAAttention
model=tf.keras.applications.VGG19(include_top=False,weights='imagenet')
inputs = tf.keras.layers.Input(shape=(224, 224, 3))
x = model(inputs)  # 此处x为MobileNetV2模型去处顶层时输出的特征相应图。
x = ECAAttention(512)(x)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x=tf.keras.layers.Dense(128)(x)
x=tf.keras.layers.Flatten(name='out_layer2')(x)
outputs = tf.keras.layers.Dense(4, activation='softmax',
                      use_bias=True, name='Logits')(x)
model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
model.summary()
opt = optimizers.Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.00, amsgrad=False)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['acc'])
history = model.fit(x_train1, y_train1, epochs=15, batch_size=16, validation_split=0.1)

'''将训练后的CNN当作特征提取器 提取高层的128维特征向量'''
permute_layer_model2 = Model(model.input,model.get_layer('out_layer2').output)
permute_layer_output_xtrain = permute_layer_model2.predict((x_train1))
permute_layer_output_xtrain=np.reshape(permute_layer_output_xtrain,(len(permute_layer_output_xtrain),128))
permute_layer_output_xtest=permute_layer_model2.predict((x_test1))
permute_layer_output_xtest=np.reshape(permute_layer_output_xtest,(len(permute_layer_output_xtest),128))

np.save("LSR-Net_CNN_x_train.npy",permute_layer_output_xtrain)
np.save("LSR-Net_CNN_x_test.npy",permute_layer_output_xtest)
