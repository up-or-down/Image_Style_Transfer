# coding=utf-8
import numpy as np
import struct
import os
import time
import pickle 

def show_matrix(mat, name):
    #print(name + str(mat.shape) + ' mean %f, std %f' % (mat.mean(), mat.std()))
    pass

def show_time(time, name):
    #print(name + str(time))
    pass

class ConvolutionalLayer(object):
    def __init__(self, kernel_size, channel_in, channel_out, padding, stride):
        # 卷积层的初始化
        self.kernel_size = kernel_size
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.padding = padding
        self.stride = stride
        print('\tConvolutional layer with kernel size %d, input channel %d, output channel %d.' % (self.kernel_size, self.channel_in, self.channel_out))
    def init_param(self, std=0.01):  # 参数初始化
        self.weight = np.random.normal(loc=0.0, scale=std, size=(self.channel_in, self.kernel_size, self.kernel_size, self.channel_out))
        self.bias = np.zeros([self.channel_out])
    def forward(self, input):  # 前向传播的计算
        start_time = time.time()
        self.input = input # [N, C, H, W]
        # TODO: 边界扩充
        height = self.input.shape[2] + 2 * self.padding
        width = self.input.shape[3] + 2 * self.padding
        self.input_pad = np.zeros([self.input.shape[0], self.input.shape[1], height, width])
        self.input_pad[:, :, self.padding:self.input.shape[2]+self.padding, self.padding:self.input.shape[3]+self.padding] = self.input
        height_out = (height - self.kernel_size) // self.stride + 1
        width_out = (width - self.kernel_size) // self.stride + 1
        self.output = np.zeros([self.input.shape[0], self.channel_out, height_out, width_out])
        for idxn in range(self.input.shape[0]):
            for idxc in range(self.channel_out):
                for idxh in range(height_out):
                    for idxw in range(width_out):
                        # TODO: 计算卷积层的前向传播，特征图与卷积核的内积再加偏置
                        self.output[idxn, idxc, idxh, idxw] = np.sum(self.weight[:, :, :, idxc] * self.input_pad[idxn, :, idxh*self.stride:idxh*self.stride+self.kernel_size, idxw*self.stride:idxw*self.stride+self.kernel_size]) + self.bias[idxc]
        return self.output
    
    def backward(self, top_diff):
        start_time = time.time()
        self.d_weight = np.zeros(self.weight.shape)
        self.d_bias = np.zeros(self.bias.shape)
        bottom_diff = np.zeros(self.input_pad.shape)
        for idxn in range(top_diff.shape[0]):
            for idxc in range(top_diff.shape[1]):
                for idxh in range(top_diff.shape[2]):
                    for idxw in range(top_diff.shape[3]):
                        # TODO： 计算卷积层的反向传播， 权重、偏置的梯度和本层损失
                        self.d_weight[:, :, :, idxc] += \
                            top_diff[idxn, idxc, idxh, idxw] * self.input_pad[idxn, :, \
                            idxh*self.stride:idxh*self.stride+self.kernel_size, \
                            idxw*self.stride:idxw*self.stride+self.kernel_size]
                        self.d_bias[idxc] += top_diff[idxn, idxc, idxh, idxw]
                        bottom_diff[idxn, :, idxh*self.stride:idxh*self.stride+self.kernel_size, \
                            idxw*self.stride:idxw*self.stride+self.kernel_size] += \
                            top_diff[idxn, idxc, idxh, idxw] * self.weight[:, :, :, idxc]
        bottom_diff = bottom_diff[:, :, self.padding:self.padding+self.input.shape[2], self.padding:self.padding+self.input.shape[3]]
        self.backward_time = time.time() - start_time
        return bottom_diff
    
    
    def load_param(self, weight, bias):  # 参数加载
        assert self.weight.shape == weight.shape
        #print self.bias.shape
        #print bias.shape
        assert self.bias.shape == bias.shape
        self.weight = weight
        self.bias = bias

class MaxPoolingLayer(object):
    def __init__(self, kernel_size, stride):  # 最大池化层的初始化
        self.kernel_size = kernel_size
        self.stride = stride
        print('\tMax pooling layer with kernel size %d, stride %d.' % (self.kernel_size, self.stride))
    def forward(self, input):  # 前向传播的计算
        start_time = time.time()
        self.input = input # [N, C, H, W]
        print("input_shape:",self.input.shape)
        self.max_index = np.zeros(self.input.shape)
        height_out = (self.input.shape[2] - self.kernel_size) // self.stride + 1
        width_out = (self.input.shape[3] - self.kernel_size) // self.stride + 1
        self.output = np.zeros([self.input.shape[0], self.input.shape[1], height_out, width_out])
        for idxn in range(self.input.shape[0]):
            for idxc in range(self.input.shape[1]):
                for idxh in range(height_out):
                    for idxw in range(width_out):
			            # TODO： 计算最大池化层的前向传播， 取池化窗口内的最大值
                        self.output[idxn, idxc, idxh, idxw] = np.max(self.input[idxn, idxc, idxh*self.stride:idxh*self.stride+self.kernel_size, idxw*self.stride:idxw*self.stride+self.kernel_size])
                        now_index = np.argmax(self.input[idxn,idxc,idxh*self.stride:idxh*self.stride+self.kernel_size,idxw*self.stride:idxw*self.stride+self.kernel_size])
                        now_h = now_index//self.kernel_size 
                        now_w = now_index - now_h * self.kernel_size
                        #print("now_h:"+str(now_h)+",now_w:"+str(now_w))
                        #print("now_index"+str(now_index))
                        self.max_index[idxn,idxc,idxh*self.stride+now_h,idxw*self.stride+now_w] = 1 #用于反向传播时索引最大值的位置
        return self.output
    def backward(self, top_diff):
        #f = open('top_diff.txt','wb')
        #pickle.dump(top_diff,f)
        #f.close()
        bottom_diff = np.zeros(self.input.shape)
        for idxn in range(top_diff.shape[0]):
            for idxc in range(top_diff.shape[1]):
                for idxh in range(top_diff.shape[2]):
                    for idxw in range(top_diff.shape[3]):
                        # TODO: 最大池化层的反向传播， 计算池化窗口中最大值位置， 并传递损失
                        # 注意：池化作用于图像中不重合的区域
                        #f = open('max_index.txt','wb')
                        #pickle.dump(self.max_index,f)
                        #f.close()
                        max_index = np.argwhere(self.max_index[idxn, idxc, \
                            idxh*self.stride:idxh*self.stride+self.kernel_size, \
                            idxw*self.stride:idxw*self.stride+self.kernel_size])[0]
                        bottom_diff[idxn, idxc, idxh*self.stride+max_index[0], idxw*self.stride+max_index[1]] = \
                            top_diff[idxn, idxc, idxh, idxw]
        return bottom_diff

class FlattenLayer(object):
    def __init__(self, input_shape, output_shape):  # 扁平化层的初始化
        self.input_shape = input_shape
        self.output_shape = output_shape
        assert np.prod(self.input_shape) == np.prod(self.output_shape)
        print('\tFlatten layer with input shape %s, output shape %s.' % (str(self.input_shape), str(self.output_shape)))
    def forward(self, input):  # 前向传播的计算
        #print str(list(input.shape[1:]))
        #print str(list(self.input_shape))
        assert list(input.shape[1:]) == list(self.input_shape)
        # matconvnet feature map dim: [N, height, width, channel]
        # ours feature map dim: [N, channel, height, width]
        # TODO：转换 input 维度顺序
        """ [N, channel, height, width] -> [N, height, width, channel] """
        self.input = input.transpose(0, 2, 3, 1)
        self.output = self.input.reshape([self.input.shape[0]] + list(self.output_shape))
        show_matrix(self.output, 'flatten out ')
        return self.output
