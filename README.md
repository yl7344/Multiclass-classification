# Multiclass-classification
├── notebooks  <br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── notebooks on how to use the .py file <br/>
├── pdf <br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── print view of notebook outputs and project report <br/>
├── CNN.py <br/>
├── ResNet.py <br/>
├── ResNet1.py <br/>
├── data.py  <br/>
├── evaluate.py <br/>
└── train.py <br/>

# I. Dependencies
1.&nbsp;Python 3.6.9<br/>
2.&nbsp;Keras 2.4.3<br/>
3.&nbsp;Numpy 1.19.4<br/>
4.&nbsp;Matplotlib 3.2.2<br/>
5.&nbsp;Pytorch<br/>
6.&nbsp;Google Colab GPUs<br/>
7.&nbsp;NYU HPC<br/>

# II. CNN.py
Provide simple CNN for training splited classes.<br/>

# III. ResNet.py & ResNet1.py
ResNet Structure from https://arxiv.org/abs/1512.03385v1, ResNet1.py provides modified structure. <br/>

# IV. data.py
Provide three functions to load data from Cifar-100.<br/>
loadData(batch_size) returns original data set.<br/>
loadData_byBigClass(batch_size) returns whole data set but their lables are superclasses.<br/>
loadData_minClass1(batch_size,class_num) returns data points under selected superclass. <br/>

# V. evaluate.py
Provides functions to return the accuracy of the model.

# VI. train.py
provides three functions to train networks on three different data sets.
