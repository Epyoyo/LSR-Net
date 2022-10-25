import tensorflow as tf
from tensorflow.keras import layers

class Feature_Aug(layers.Layer):
    def __init__(self, kernel_size=13):
        super(Feature_Aug, self).__init__()
        self.conv1 = tf.keras.layers.Conv1D(1, kernel_size=3, padding='same',dilation_rate=1)
        self.conv2 = tf.keras.layers.Conv1D(1, kernel_size=5, padding='same',dilation_rate=1)
        self.conv3 = tf.keras.layers.Conv1D(1, kernel_size=7,padding='same',dilation_rate=1)

        self.conv4 = tf.keras.layers.Conv1D(1, kernel_size=3,  padding='same',dilation_rate=2)
        self.conv5 = tf.keras.layers.Conv1D(1, kernel_size=5,  padding='same',dilation_rate=2)
        self.conv6 = tf.keras.layers.Conv1D(1, kernel_size=7,padding='same',dilation_rate=2)

        self.conv7 = tf.keras.layers.Conv1D(1, kernel_size=4,padding='same',dilation_rate=3)
        self.conv8 = tf.keras.layers.Conv1D(1, kernel_size=5,  padding='same',dilation_rate=3)
        self.conv9 = tf.keras.layers.Conv1D(1, kernel_size=7,  padding='same', dilation_rate=3)

    def call(self, inputs):

        out = tf.stack([inputs, inputs], axis=-1)
        out = tf.keras.layers.Reshape((6, 128, 2))(out)
        #
        out0 = out[:, 0, :, :]
        out0_1 = self.conv1(out0)
        out0_2 = self.conv2(out0)
        out0_3 = self.conv3(out0)
        out0_4 = self.conv4(out0)
        out0_5 = self.conv5(out0)
        out0_6 = self.conv6(out0)
        out0_7 = self.conv7(out0)
        out0_8 = self.conv8(out0)
        out0_9 = self.conv9(out0)
        out0_0 = out0_1 + out0_2 + out0_3 + out0_4 + out0_5 + out0_6 + out0_7 + out0_8 + out0_9

        temp_out = out0_0
        last_out = None
        for i in range(1, 6):
            out_i = out[:, i, :, :]
            outi_1 = self.conv1(out_i)
            outi_2 = self.conv2(out_i)
            outi_3 = self.conv3(out_i)
            outi_4 = self.conv4(out_i)
            outi_5 = self.conv5(out_i)
            outi_6 = self.conv6(out_i)
            outi_7 = self.conv7(out_i)
            outi_8 = self.conv8(out_i)
            outi_9 = self.conv9(out_i)
            outi_0 = outi_1 + outi_2 + outi_3 + outi_4 + outi_5 + outi_6 + outi_7 + outi_8 + outi_9
            # outi_0的维度是(batch,128,1)
            # 利用Concatenate拼接后:(batch,128,6)
            last_out = tf.keras.layers.Concatenate(axis=2)([temp_out, outi_0])
            temp_out = last_out

        # 为了符合后续的 循环神经网络的输入格式，利用transpose将其转换为(batch,6,128)
        last_out = tf.transpose(last_out, perm=[0, 2, 1])
        return last_out

    def compute_output_shape(self, input_shape):
        return input_shape

