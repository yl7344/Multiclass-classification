import ResNet1
import data
import evaluate
import ResNet
import torch
import CNN
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model,train,test,loss_fn,optimizer,max_iter,watch_iter):
    total_iter = 0
    loss = 0.0
    train_loss_his = []
    test_loss_his = []
    while total_iter < max_iter:
        for batch in train:
            total_iter += 1
            train_inputs, train_labels = batch
            train_inputs, train_labels = train_inputs.to(device), train_labels.to(device)
            train_outputs = model(train_inputs)
            l = loss_fn(train_outputs, train_labels)
            loss += l.item()
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            if total_iter % watch_iter == 0:
                train_loss = loss / watch_iter
                train_loss_his.append(train_loss)
                loss = 0.0
                for batch in test:
                    test_inputs, test_labels = batch
                    test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)
                    test_outputs = model(test_inputs)
                    l = loss_fn(test_outputs, test_labels)
                    loss += l.item()
                test_loss_his.append(loss)
                txt = f'iter: {total_iter: 6d}, train loss: {train_loss}, test_loss: {loss}'
                print(txt)
                print('accuracy: ' + str(evaluate.evaluate(model,test)*100) + '%')
                loss = 0.0
    return train_loss_his,test_loss_his

def train_BigNet(total_iter, epochs):
    # [3, 4, 6, 3] basic block size, with up to 512 chanels.
    net = ResNet.resnet34(100).to(device)

    # Load Whole Dataset
    trainDataLoader, testDataLoader =  data.loadData(250)

    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)  

    
    train_loss_his,test_loss_his = train(net,trainDataLoader,testDataLoader,loss,optimizer,total_iter,epochs)
    return net,train_loss_his,test_loss_his

def train_smallNet(total_iter, epochs, numbers):
    # [3, 4, 6, 3] basic block size, with up to 128 chanels.
    nets = [ResNet.ResNet(ResNet.BasicBlock, [1, 1, 1, 1],100).to(device) for i in range(numbers)]

    # Load Whole Dataset
    trainDataLoader, testDataLoader =  data.loadData(250)

    loss = nn.CrossEntropyLoss()
    optimizer = [torch.optim.Adam(nets[i].parameters(), lr=0.0001)  for i in range(numbers)]

    train_loss_his = [[] for i in range(numbers)]
    test_loss_his = [[] for i in range(numbers)]
    for i in range(numbers):
        #if i != 0:
        #    nets[i].load_state_dict(nets[i-1].state_dict())
        train_loss_his[i],test_loss_his[i] = train(nets[i],trainDataLoader,testDataLoader,loss,optimizer[i],total_iter,epochs)

    return nets, train_loss_his,test_loss_his
