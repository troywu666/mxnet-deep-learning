# 基于mxnet框架的ndarray和autograd实现深度学习

标签（空格分隔）： 不调包，从0生成

---
# 一、深度学习基础
## 1、ndarray的基本使用
* 注意内存的开销,可使用如nd.elemwise_add函数指定存入内存，避免临时内存
```python
nd.elemwise_add(X,Y,out=Z)
##指定X和Y的和存放在Z中，从而减少X+Y的临时内存
```
* ndarray的广播机制

---
## 2、自动梯度的计算
* 在对y求梯度时，若y不是标量，则将y的元素求和后再求梯度，与y.sum()后再求梯度为同一值
* 求梯度过程：
```python
x.grad_attach() ##为x申请内存
with autograd.record(): ##申请对梯度计算做记录
    f(x).backward() ##反向传导计算，即求导
x.grad ##输出导数值
```
* 使用record函数后会默认将预测模式转为训练模式，可以用过is_training()来查看

---
## 3、单层神经网络
### a、单层线性回归神经网络
```python
from mxnet import nd
from mxnet import autograd as ag

#手动创建数据集
num_exanples=1000
num_inputs=2
true_w=[2,-3.4]
true_b=4.2
features=nd.random.normal(scale=1,shape=(num_exanples,num_inputs))
labels=features[ : ,0]*true_w[0]+features[ : ,1]*true_w[1]+true_b
noise=nd.random.normal(scale=0.01,shape=(labels.shape))
labels+=noise

#训练数据集时，需要遍历数据。但一次性读取所有数据会对内存产生极大开销，所以需要定义batach_size,读取小批量数据
import random
def data_iter(batch_size,features,labels):
    num_examples=len(features)
    indices=list(range(num_examples))
    random.shuffle(indices)
    for i in range(0,num_examples,batch_size):
        j=nd.array(indices[i:min(i+batch_size,num_examples)])
        yield features.take(j),labels.take(j)##根据索引返回相应函数
        
#初始化参数w和b，并申请计算梯度的内存
w=nd.random.normal(scale=0.01,shape=(num_inputs,1))
b=nd.zeros(shape=(1,))
w.attach_grad()
b.attach_grad()

#定义损失函数
def square_loss(y,y_hat):
    return (y.reshape(y_hat.shape)-y_hat)**2/2
    
#定义随机梯度下降函数（尽管线性回归具解析解，但绝大多数模型不可能有解析解）
def sgd(params,batch_size,lr):
    for param in params:
        param[:]-=lr*param.grad/batch_size##[:]是将内容传送到新地址
        
#线性回归表达式
def linreg(x,w,b):
    return nd.dot(x,w)+b

#训练模型
loss=square_loss
net=linreg
epoch=5
lr=0.03
for i in range(epoch):
    for x,y in data_iter(batch_size,features,labels):
        with ag.record():
            loss(y,net(x,w,b)).backward()
        sgd([w,b],batch_size,lr)        
    l=loss(labels,net(features,w,b))
    print("epoch%d  :  loss=%f"%(i+1,l.mean().asnumpy()))
```

---
### b、单层softmax回归神经网络
```python
import d2lzh as d2l
from mxnet import nd
from mxnet import autograd as ag

#加载数据集
batch_size=256
train_iter,test_iter =d2l.load_data_fashion_mnist(batch_size)

#初始化模型参数
num_inputs=28*28
num_outputs=10

w=nd.random.normal(scale=0.01,shape=(num_inputs,num_outputs))
b=nd.zeros(shape=num_outputs)

w.attach_grad()
b.attach_grad()

#定义net函数
def Softmax(X):
    X_exp=X.exp()
    X_exp_sum=X_exp.sum(axis=1,keepdims=True)
    return X_exp/X_exp_sum
def net(X):
    return Softmax(nd.dot(X.reshape(-1,num_inputs),w)+b)   

#定义损失函数和正确率函数
def cross_entropy(y_hat,y):
    return -nd.pick(y_hat,y).log()
def evaluate_accuracy(data_iter,net):
    acc_sum,n=0.0,0
    for x,y in data_iter:
        y_hat=net(x)
        y=y.astype('float32')
        acc_sum+=((y_hat.argmax(axis=1))==y).sum().asscalar()
        n+=y.size
    return acc_sum/n

#定义随机梯度下降函数
def sgd(params,lr,batch_size):    
    for param in params:
        param[:]-=lr*param.grad/batch_size

#训练模型
n=0.0
lr=0.1
epochs=10
def train_ch3(net,train_iter,test_iter,loss,epochs,batch_size,w,b,lr,trainer):
    for epoch in range(epochs):
        train_loss,train_accuracy,n=0.0,0.0,0
        for x,y in train_iter: 
            with ag.record():
                l=loss(net(x),y).sum()
            l.backward()
            n+=y.size
            trainer([w,b],lr,batch_size)
            train_loss+=l.asscalar()
            y=y.astype("float32")
            train_accuracy+=(net(x).argmax(axis=1)==y).sum().asscalar()
        test_accuracy=evaluate_accuracy(test_iter,net)
        print("NO.%s :train_loss: %.4f, train_accuracy: %.4f, test_accuracy: %.4f"%(epoch+1,train_loss/n,train_accuracy/n,test_accuracy))
train_ch3(net,train_iter,test_iter,cross_entropy,epochs,batch_size,w,b,lr,sgd)

#用测试集验证并可视化
for x,y in test_iter:
    true_labels=d2l.get_fashion_mnist_labels(y.asnumpy())
    pred_labels=d2l.get_fashion_mnist_labels(net(x).argmax(axis=1).asnumpy())
    titles=[true+"\n"+pred for true,pred in zip(true_labels,pred_labels)]
    d2l.show_fashion_mnist(x[0:9],titles[0:9])
    break
```
---

## 4、多层感知机
多层感知机在单层神经网络的基础上引入了一到多个隐藏层，隐藏层位于输入层和输出层中间。
若在隐藏层使用映射函数，将使得多层感知机变为单层神经网络，因此需要在隐藏层引入非线性变换。
```python
#加载数据
from mxnet import autograd as ag
from mxnet import ndarray as nd
import d2lzh as d2l
batch_size=256
train_iter,test_iter=d2l.load_data_fashion_mnist(batch_size)

#初始化参数，并为求导申请内存
num_inputs,num_outputs,num_hiddens=28*28,10,256

w1=nd.random.normal(scale=0.01,shape=(num_inputs,num_hiddens))
b1=nd.zeros(num_hiddens)
w2=nd.random.normal(scale=0.01,shape=(num_hiddens,num_outputs))
b2=nd.zeros(num_outputs)

for param in [w1,b1,w2,b2]:
    param.attach_grad()

#定义隐藏层的非线性变换函数，此处使用relu函数
def relu(X):
    return nd.maximum(X,0)

#定义神经网络函数
def Softmax(X):
    X_sum_exp=X.exp().sum(axis=1,keepdims=True)
    ##重要容易出错信息,keepdims用于保持维度特性
    return X.exp()/X_sum_exp
def net(X):
    X=X.reshape(-1,num_inputs)
    H1=relu(nd.dot(X,w1)+b1)
    return Softmax(nd.dot(H1,w2)+b2)

#定义损失函数和验证函数
def cross_entropy(y_hat,y):
    return -nd.pick(y_hat,y).log()
def evaluate_accuracy(data_iter,net):
    corr,n=0.0,0
    for X,y in data_iter:
        y=y.astype("float32")
        y_hat=net(X)
        corr+=(y_hat.argmax(axis=1)==y).sum().asscalar()
        n+=y.size
    return corr/n

#定义随机梯度下降函数
def sgd(params,lr,batch_size):
    for param in params:
        param[:]-=lr*param.grad/batch_size

#定义训练函数和开始训练
def train_ch3(train_iter,test_iter,batch_size,lr,net,params,epochs,loss,trainer):
    for epoch in range(epochs):
        for X,y in train_iter:
            y=y.astype("float32")
            n,train_cross_enropy,train_acc=0,0.0,0.0
            with ag.record():
                l=loss(net(X),y).sum()
            l.backward()
            trainer(params,lr,batch_size)
            n+=y.size
            train_cross_enropy+=l.asscalar()
            train_acc+=(net(X).argmax(axis=1)==y).sum().asscalar()
        test_acc=evaluate_accuracy(test_iter,net)
        print("NO.%s ,train_loss is %.3f, train_acc is %.4f, test_acc is %.4f"%(epoch+1,train_cross_enropy/n,train_acc/n,test_acc))
batch_size,lr,epochs=256,0.5,10
train_ch3(train_iter,test_iter,batch_size,lr,net,[w1,b1,w2,b2],epochs,cross_entropy,sgd)
```
---
## 5、正则化
因神经网络你和能力极强，拟合数据时很容易出现过拟合现象，因此需要对模型做正则化，对抗过拟合
### a、权重衰减
即L2正则化，在损失函数上加上L2惩罚项
```python
from mxnet import autograd as ag
from mxnet import nd

def sgd(params,lr,batch_size):
    for param in params:
        param[:]-=lr*param.grad/batch_size

def l2_penalty(w):
    return (w**2).sum()/2

def entropy_loss(y_hat,y):
    return -nd.pick(y_hat,y).log().sum()

def evaluate_accuracy(data_iter,net):
    acc,n=0.0,0
    for X,y in data_iter:
        y=y.astype('float32')
        acc+=(net(X).argmax(axis=1)==y).sum().asscalar()
        n+=y.size
    return acc/n

def trainer_ch3(batch_size,lr,train_iter,test_iter,net,train,epochs,loss,l2_penalty,params):
    for epoch in range(epochs):
        acc_train,n=0.0,0
        for X,y in train_iter:
            y_hat=net(X)
            with ag.record():
                l=loss+l2_penalty(params[0])##损失函数加上L2惩罚项
            l.backward()
            sgd(params,lr,batch_size)
            n+=y.size
            train_acc+=(y_hat.rgmax(axis=1)==y).sum().asscalar()
        test_acc=evaluate_accuracy(test_iter,net)
        print("NO.%d: train_loss=%.3f, test_loss="%(epoch+1,train_acc/n,test_acc))     
```
### b、dropout法
为了保证结果稳定，只在训练阶段使用dropout法
```python
from mxnet import autograd as ag
from mxnet import nd

def dropout(X,drop_prob):
    assert 0<=drop_prob<=1
    keep_prob=1-drop_prob
    if keep_prob ==0:
        return X.zeros_like()
    ##小于keep_prob时为0，大于keep_prob时为1
    mask=nd.random.uniform(0,1,X.shape)<keep_prob
    return mask*X/keep_prob

num_inputs,num_outputs,num_hiddens1,num_hiddens2=784,10,256,256

w1=nd.random.normal(scale=0.01,shape=(num_inputs,num_hiddens1))
b1=nd.zeros(num_hiddens1)
w2=nd.random.normal(scale=0.01,shape=(num_hiddens1,num_hiddens2))
b2=nd.zeros(num_hiddens2)
w3=nd.random.normal(scale=0.01,shape=(num_hiddens2,num_outputs))
b3=nd.zeros(num_outputs)

params=[w1,b1,w2,b2,w3,b3]
for param in params:
    param.attach_grad()

drop_prob1, drop_prob2=0.2,0.5

def relu(X):
    return nd.maximum(X,0)

def net(X):
    X=X.reshape(-1,num_inputs)
    h1=relu(nd.dot(X,w1)+b1)
    if ag.is_training():
    ##判断是否为训练模式后再进行dropout
        h1=dropout(h1,drop_prob1)
    h2=relu(nd.dot(h1,w2)+b2)
    if ag.is_training():
        h2=dropout(h2,drop_prob2)
    return nd.dot(h2,w3)+b3
```
---
## 6、用mxnet直接生成模型
### 6.1、神经网络的构建
```python
from mxnet.gluon import nn
from mxnet import nd
from mxnet import autograd as ag

class MySequential(nn.Block):
    def __init__(self,**kwargs):
        super(MySequential,self).__init__(**kwargs)
        
    def add(self,block):
        self._children[block.name]=block
        
    def forward(self,x):
        for block in self._children.values():
            x=block(x)
        return x
    

class FancyMLP(nn.Block):
    def __init__(self,**kwargs):
        super(FancyMLP,self).__init__(**kwargs)
        self.rand_weight=self.params.get_constant("rand_weight",nd.random.uniform(shape=(10,10)))
        self.dense=nn.Dense(10,activation="relu")
    
    def forward(self,x):
        x=self.dense(x)
        x=nd.relu(nd.dot(x,self.rand_weight.data())+1)
        x=self.dense(x)
        while x.norm().asscalar()>1:
            x/=2
        if x.norm().asscalar()<0.8:
            x*=10
        return x.sum()

class MLP(nn.Block):
    def __init__(self,**kwargs):
        super(MLP,self).__init__(**kwargs)
        self.hidden=nn.Dense(20,activation="relu")
        self.output=nn.Dense(10)
        
    def forward(self,x):
        return self.output(self.hidden(x))
    
##错误实例
class NestMLP(nn.Block):
    def __init__(self,**kwargs):
        super(NestMLP,self).__init__(**kwargs)
        self.rand_weight=self.params.get_constant(
        'rand_weight',nd.random.uniform(shape=(10,10)))
        self.dense=nn.Dense(20,activation="relu")
    
    def add(self,block):
        self._children[block.name]=block
    
    def forward(self,x):
        for block in self._children.values():
            x=block(x)
##错误实例

net=nn.Sequential()
net.add(MLP(),nn.Dense(10),FancyMLP())
net.initialize()
X=nd.random.uniform(shape=(20,20))
net(X)
```
* 注意：以上NestMLP的定义会导致该层神经网络作废，因为前向输出只有输出要加的神经网络层！！！
### 6.2、模型参数的访问、初始化和共享
```python
#访问
from mxnet.gluon import nn
from mxnet import nd,init

net=nn.Sequential()
net.add(nn.Dense(256,activation="relu"))
net.add(nn.Dense(10))
net.initialize()

X=nd.random.uniform(shape=(2,20))
Y=net(X)

import sys

get_net=nn.Sequential()
get_net.add(nn.Dense(256,activation="relu"))
get_net.add(nn.Dense(10))

try:
    get_net(X)
except RuntimeError as err:
    sys.stderr.write(str(err))    

net[0].params

net[0].weight,net[0].bias

net[0].weight.data()

net[0].bias.data()

net[0].weight.grad()

net.collect_params()

net.collect_params(".*weight")

#初始化
class MyInit(init.Initializer):
    def _init_weight(self,name,data):
        print("Init",name,data.shape)
        data[:]=nd.random.uniform(low=-10,high=10,shape=data.shape)
        data*=data.abs()>=5
        
net.initialize(MyInit(),force_reinit=True)
net[0].weight.data()

net[0].weight.set_data(net[0].weight.data()+1) ##用set_data使得参数全部+1
net[0].weight.data()

#参数共享
from mxnet.gluon import nn
from mxnet import nd

net =nn.Sequential()
shared=nn.Dense(8)
net.add(nn.Dense(8,activation="relu"),
       shared,
       nn.Dense(8,activation="relu",params=shared.params),
       nn.Dense(10))
net.initialize()
X=nd.random.uniform(shape=(2,20))

net(X)

net[1].weight.data()==net[2].weight.data()

#取消延后初始化
from mxnet import nd,init
from mxnet.gluon import nn

net=nn.Sequential()
net.add(nn.Dense(256,activation="relu",in_units=20))
net.add(nn.Dense(10,in_units=256))

class MyInit(init.Initializer):
    def _init_weight(self,name,data):
        print('Init',name,data.shape)

net.initialize(init=MyInit())
```