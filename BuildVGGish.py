import tensorflow as tf
gpus=tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu,True)
import vggish_params as params
from tensorflow.keras.layers import  Flatten, Dense,Conv2D,MaxPooling2D,Input
from tensorflow.keras import Model
WEIGHTS_PATH='vggish_audioset_weights_without_fc2.h5'
WEIGHTS_PATH_TOP='vggish_audioset_weights.h5'
def Build_VGGish(input_shape=None,out_dim=None):
    if out_dim is None:
       out_dim = params.EMBEDDING_SIZE
    if input_shape is None:
        input_shape =(params.NUM_FRAMES, params.NUM_BANDS, 1)
    aud_input = Input(shape=input_shape, name='input_1')
    # Block 1
    x = Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv1')(aud_input)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool1')(x)
    # Block 2
    x = Conv2D(128, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool2')(x)
    # Block 3
    x = Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv3/conv3_1')(x)
    x = Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv3/conv3_2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool3')(x)
    # Block 4
    x = Conv2D(512, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv4/conv4_1')(x)
    x = Conv2D(512, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv4/conv4_2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool4')(x)
    x = Flatten(name='flatten_')(x)
    x = Dense(4096, activation='relu', name='vggish_fc1/fc1_1')(x)
    x = Dense(4096, activation='relu', name='vggish_fc1/fc1_2')(x)
    x = Dense(out_dim, activation='relu', name='vggish_fc2')(x)
    inputs = aud_input
    # Create model.
    model = Model(inputs, x, name='VGGish')
    # load weights
    model.load_weights(WEIGHTS_PATH_TOP)
    return model


# Build New VGGish,
vggish_model1=Build_VGGish()
from ECANet import ECAAttention
def Build_New_VGGish(input_shape = None,):
    if input_shape is None:
        input_shape = (96, 64, 1)
    aud_input = tf.keras.layers.Input(shape=input_shape, name='input_1')
    x=vggish_model1.get_layer(name='conv1')( aud_input )#64
    x=vggish_model1.get_layer(name='pool1')(x)
    x=vggish_model1.get_layer(name='conv2')(x)
    x=vggish_model1.get_layer(name='pool2')(x)
    x=vggish_model1.get_layer(name='conv3/conv3_1')(x)

    x=vggish_model1.get_layer(name='conv3/conv3_2')(x)  # 256
    x=vggish_model1.get_layer(name='pool3')(x)
    x=vggish_model1.get_layer(name='conv4/conv4_1')(x)
    x=vggish_model1.get_layer(name='conv4/conv4_2')(x)  # 512
    x=vggish_model1.get_layer(name='pool4')(x)
    # 在这里加入ECA-Net注意力
    x = ECAAttention(512)(x)
    x=vggish_model1.get_layer(name='flatten_')(x)
    x=vggish_model1.get_layer(name='vggish_fc1/fc1_1')(x)
    x=vggish_model1.get_layer(name='vggish_fc2')(x)
    model = Model(aud_input, x, name='VGGish')
    return model
