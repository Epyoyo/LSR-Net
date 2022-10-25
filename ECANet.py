from tensorflow.keras import layers
import math
from tensorflow.keras.layers import Reshape,multiply

class ECAAttention(layers.Layer):
    def __init__(self, in_planes,ratio=16):
        super(ECAAttention, self).__init__()
        channel=in_planes
        b=1
        gamma=2
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        self.G1=layers.GlobalAveragePooling2D()
        self.R1=Reshape((-1,1))
        self.C1=layers.Conv1D(1,kernel_size=kernel_size,padding="same", use_bias=False,)
        self.A1=layers.Activation('sigmoid')
        self.R2=Reshape((1,1,-1))
    def call(self, inputs):
        x=self.G1(inputs)
        x=self.R1(x)
        x=self.C1(x)
        x=self.A1(x)
        x=self.R2(x)
        out=multiply([inputs,x])
        return out
    def compute_output_shape(self, input_shape):
        return input_shape
