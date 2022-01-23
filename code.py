#https://github.com/jgabriellima/backpropagation
#参考了此类代码，因为好多人转发这个代码，在此基础上进行了修改
#以下内容复制即可运行
#需要的库和pyhon环境
#Python3.7(64-bit)
#keras(2.60)  numpy(1.19.2)
#__author__ =Daylighttx

from keras.datasets import mnist
import numpy as np
import math
import random


random.seed(0) #用seed（n）可以让random每次都为一个值，不同n为不同值

def one_hot(targets, num):
    result = np.zeros((num, 10))
    for i in range(num):
        result[i][targets[i]] = 1
    return result#one_hot编码，输入训练集和个数，能将前num位变为one_hot编码格式

def rand(a, b):

    return (b - a) * random.random() + a  #创建一个a，b之间的随机数


def makeMatrix(I, J, fill=0.0):
    m = []
    for i in range(I):
        m.append([fill] * J)
    return m  #定义一个矩阵，I为行长，J为列长，用0.0填充


def randomizeMatrix(matrix, a, b):
    """
    随机初始化矩阵
    :param matrix:
    :param a:
    :param b:
    """
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            matrix[i][j] = random.uniform(a, b)  #随机化矩阵，随机数范围在a，b之间


def sigmoid(x):
    #用math.exp(-x)可能会报错 only size-1 arrays can be converted to Python scalars
    #改用np.exp(-x)即可
    return  1.0 / (1.0 + np.exp(-x))  #sigmoid函数，用于激活


def dsigmoid(y):
    """
    sigmoid 函数的导数
    :param y:
    :return:
    """
    return y * (1 - y)  #sigmoid函数的导数，在计算误差反向传播时使用


class NN:   #定义NN类（Neural Network）
    def __init__(self, ni, nh, no):
        # number of input, hidden, and output nodes
        """  
        构造神经网络
        :param ni:输入单元数量 
        :param nh:隐藏单元数量
        :param no:输出单元数量
        """
        self.ni = ni + 1  # +1 是为了偏置节点
        self.nh = nh
        self.no = no

        # 激活值（输出值）active
        self.ai = [1.0] * self.ni
        self.ah = [1.0] * self.nh
        self.ao = [1.0] * self.no

        # 权重矩阵 weight
        self.wi = makeMatrix(self.ni, self.nh)  # 输入层到隐藏层 -num_input;\num_hidden
        self.wo = makeMatrix(self.nh, self.no)  # 隐藏层到输出层 -num_hidden;\num_output
        # 将权重矩阵随机化
        randomizeMatrix(self.wi, -0.2, 0.2)  #input权重矩阵在-0.2和0.2间随机化
        randomizeMatrix(self.wo, -2.0, 2.0)  #随机output权重矩阵
        # 权重矩阵的上次梯度
        self.ci = makeMatrix(self.ni, self.nh) #保存更新前的权重
        self.co = makeMatrix(self.nh, self.no)


    def update(self, inputs):
        if len(inputs) != self.ni - 1:		# 输入的样本特征量数等于神经网络输入层数-1，因为有一个是偏置节点
            raise ValueError('与输入层节点数不符！')		# 使用raise手工抛出异常，若引发该异常，中断程序
 
        # 激活输入层
        for i in range(self.ni - 1):		# 输入层中的偏置节点 = 1，不用激活
            self.ai[i] = inputs[i]		# 将输入样本的特征量赋值给神经网络输入层的其他节点
 
        # 激活隐藏层
        for j in range(self.nh):	# self.nh表示隐藏层的节点数，包括隐藏层的第一个节点，也就是我们人为加的偏置节点，偏置节点恒为1，是不需要激活的；应该是self.nh -1,但原代码也并不影响结果
            sum = 0.0		# 激活项a = g(z)  z = Θ^T x ;sum相当于z，每次循环归零
            for i in range(self.ni):	#通过循环z = Θ^T x ，因为Θ、x均为向量
                sum = sum + self.ai[i] * self.wi[i][j]		#〖 Z〗^((2))=Θ^((1)) a^((1))
            self.ah[j] = sigmoid(sum)		# a^((2))=g(z^((2)))，这里使用sigmoid()函数作为激活函数
 
        # 激活输出层
        for k in range(self.no):
            sum = 0.0
            for j in range(self.nh):
                sum = sum + self.ah[j] * self.wo[j][k]		#〖 Z〗^((3))=Θ^((2)) a^((2))
            self.ao[k] = sigmoid(sum)		# a^((3))=g(z^((3)))
 
        return self.ao[:]			# 返回输出值，即为某样本的预测值

    def backPropagate(self, targets, lr):		# targets为某样本实际种类分类，lr为梯度下降算法的学习率
        
        # 计算输出层的误差
        output_deltas = [0.0] * self.no		#记录方向传播的误差；输出层误差容易求，把样本的实际值减去我们当前神经网络预测的值，δ^((3))=〖y-a〗^((3) );但是输出层的误差是由前面层一层一层累加的结果，我们将误差方向传播的过程叫方向传播算法。由算法知：δ^((2))=〖(Θ^((2)))〗^T δ^((3)).*g^' (z^((2)))
        for k in range(self.no):
            error = targets[k] - self.ao[k]	#δ^((3))=〖y-a〗^((3) ),得到输出层的误差
            output_deltas[k] = error * dsigmoid(self.ao[k])          # dsigmoid()函数的功能是求公式中 g^' (z^((2))) 项，而output_deltas记录的是δ^((3)).*g^' (z^((2)))的值
                        
            # 计算隐藏层的误差
        hidden_deltas = [0.0] * self.nh		#记录的是δ^((2)).*g^' (z^((1)))的值
 
        for j in range(self.nh):
            error = 0.0
            for k in range(self.no):
                error = error + output_deltas[k] * self.wo[j][k]		#求δ^((2))，隐藏层的误差
            hidden_deltas[j] = dsigmoid(self.ah[j]) * error
 
        # 更新输出层权重
        for j in range(self.nh):		# 反向传播算法，求出每个节点的误差后，反向更新权重；由算法知Δ(_ij ^((L)))=Δ(_ij ^((L)))+a(_j  ^((L)))δ(_i      ^((L+1)))    ,而∂/(∂Θ_ij^((L) ) ) J(Θ)=Δ_ij^((L))   (λ=0) λ为正则化系数。代入梯度下降算法中：Θ_ij^((L))=Θ_ij^((L))+α  ∂/(∂Θ_ij^((L) ) ) J(Θ)即可更新权重
            for k in range(self.no):
                change = output_deltas[k] * self.ah[j]		# 求 a(_j ^((L)))δ(_i^((L+1)))  项
                self.wo[j][k] = self.wo[j][k] + lr * change 	# 用于梯度下降算法
 
        # 更新输入层权重
        for i in range(self.ni):		# 与上同理
            for j in range(self.nh):
                change = hidden_deltas[j] * self.ai[i]
                self.wi[i][j] = self.wi[i][j] + lr * change
 
        # 计算误差
        error = 0.0		# 每调用一次先归零，不停地进行迭代
        error += 0.5 * (targets[k] - self.ao[k]) ** 2		# 神经网络的性能度量，其实就是均方误差少了除以整数，但不影响度量
        return error	# 返回此时训练集的误差

    def train(self, epochs, records, lr, train_images, train_labels): 
        #epochs为迭代的次数， records为训练的个数，一共60000个训练集，records的值代表了训练的的个数从0~60000
        #lr为学习率, train_images为输入的图像，train_labels为输入的标签
        for i in range(epochs):
            error = 0.0
            for p in range(records):
               self.update(train_images[p])   #前向传播
               error += self.backPropagate(train_labels[p],lr) #反向传播更新权值，得到错误率
            #求此次迭代学习中平均错误率
            print('error: %-.5f' % (error/records)) 

    def test(self, records, test_images, test_labels):
        #records为测试的个数，mnist测试集为10000个，
        #test_images为测试的图像，test_labels为测试的标签
        count = 0		# 记录测试样本的实际值与预测值相等的个数
        for p in range(records):
            target = test_labels[p]
            result = self.update(test_images[p])
            index = result.index(max(result))	#index选出10个之中概率最大的那个
            count += (target == index)		
        accuracy = float(count / records)		#准确率为预测结果和标签值匹配的个数除以总个数
        print('accuracy: %-.9f' % accuracy)	


if __name__ == '__main__':

    #加载mnist数据集,这为keras/tensorflow中自带的库，可以导入mnist数据集
    #此句将下面四个内容完全的分离开，利于后续处理
    (train_images, train_labels),(test_images, test_labels) = mnist.load_data()
    train_images = train_images.reshape(60000,-1)/255#60000张训练集
    test_images = test_images.reshape(10000,-1)/255#10000张测试集
    #用one_hot将labels转换为one-hot编码格式
    b_train_labels = one_hot(train_labels,60000)
    b_test_labels = one_hot(test_labels,10000)#mnist数据集的准备工作，需要安装keras/tensorflow
    #需在前面加上‘from keras.datasets import mnist’


    myNN = NN(784,50,10) #定义一个输入为784，隐藏层为50，输出层为10的神经网络
    #由于输入的数据图像为28*28=784，故输入层应为784， 最后结果为10个数0~9,故输出层为10，隐藏层可自行调整，越大网络就越复杂，运行时间也就更长
    
    #train(self, epochs, records, lr,train_images, train_labels)
    print('train(epochs=4, records=1000, lr=0.1)')
    myNN.train(4, 1000, 0.1, train_images, b_train_labels)
    #records最好在1000以上，太低的records预测不准，1000时预测约为0.8，100时约为0.6
    #lr(learning rate)为0.1时效果较0.2时好

    #test(self, records, test_images, test_labels)
    #此records可随意取，但不宜太小，以减少偶然误差
    print('test(records=100)')
    myNN.test(100, test_images, test_labels)





