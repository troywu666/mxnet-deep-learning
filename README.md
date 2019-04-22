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
```