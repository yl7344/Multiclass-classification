from torchvision.datasets import CIFAR100
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
import torch

# Normalize training set together with augmentation
transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
        ])

# Normalize test set same as training set without augmentation
transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
        ])

    
# Load data
train_data = CIFAR100(download=True,root="./Cifar100",transform=transform_train)
test_data = CIFAR100(root="./Cifar100",train=False,transform=transform_test)

def loadData(batch_size):
    # Compress data to dataloader dict
    trainloader = DataLoader(train_data, 
                            batch_size=batch_size, 
                            shuffle=True, 
                            num_workers=4,
                            )

    testloader = DataLoader(test_data , 
                            batch_size=batch_size, 
                            shuffle=False, 
                            num_workers=4,
                            )

    return trainloader, testloader

def loadData_byBigClass(batch_size):
    bigClass = 	[['beaver', 'dolphin', 'otter', 'seal', 'whale'],
	              ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'],
	              ['orchid', 'poppy', 'rose', 'sunflower', 'tulip'],
                ['bottle', 'bowl', 'can', 'cup', 'plate'],
	              ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper'],
	              ['clock', 'keyboard', 'lamp', 'telephone', 'television'],
                ['bed', 'chair', 'couch', 'table', 'wardrobe'],
	              ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
	              ['bear', 'leopard', 'lion', 'tiger', 'wolf'],
                ['bridge', 'castle', 'house', 'road', 'skyscraper'],
	              ['cloud', 'forest', 'mountain', 'plain', 'sea'],
	              ['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'],
	              ['fox', 'porcupine', 'possum', 'raccoon', 'skunk'],
	              ['crab', 'lobster', 'snail', 'spider', 'worm'],
	              ['baby', 'boy', 'girl', 'man', 'woman'],
	              ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
	              ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
	              ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree'],
	              ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train'],
	              ['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor']]
    classList = train_data.classes

    # get train data
    train_data_inputs = []
    for i in range(len(train_data)):
        train_data_inputs.append(train_data[i][0])
    train_data_inputs = torch.stack((train_data_inputs)).float()
    train_data_labels = train_data.targets

    # convert label to big class
    for i,n in enumerate(train_data_labels):
      cls = classList[n]
      for j,m in enumerate(bigClass):
        if cls in m:
          train_data_labels[i] = j
          
    train_data_labels = torch.tensor(train_data_labels,dtype=torch.long)

    # get test data
    test_data_inputs = []
    for i in range(len(test_data)):
        test_data_inputs.append(test_data[i][0])
    test_data_inputs = torch.stack((test_data_inputs)).float()

    test_data_labels = test_data.targets
    # convert label to big class
    for i,n in enumerate(test_data_labels):
      cls = classList[n]
      for j,m in enumerate(bigClass):
        if cls in m:
          test_data_labels[i] = j
          
    test_data_labels = torch.tensor(test_data_labels,dtype=torch.long)

    trainloader = DataLoader(torch.utils.data.TensorDataset(train_data_inputs,train_data_labels), 
                            batch_size=batch_size, 
                            shuffle=True, 
                            num_workers=4,
                            )

    testloader = DataLoader(torch.utils.data.TensorDataset(test_data_inputs,test_data_labels) , 
                            batch_size=batch_size, 
                            shuffle=False, 
                            num_workers=4,
                            )
    #print(test_data_labels)
    return trainloader, testloader


def loadData_minClass1(batch_size,class_num):
    bigClass = 	[['beaver', 'dolphin', 'otter', 'seal', 'whale'],
	              ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'],
	              ['orchid', 'poppy', 'rose', 'sunflower', 'tulip'],
                ['bottle', 'bowl', 'can', 'cup', 'plate'],
	              ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper'],
	              ['clock', 'keyboard', 'lamp', 'telephone', 'television'],
                ['bed', 'chair', 'couch', 'table', 'wardrobe'],
	              ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
	              ['bear', 'leopard', 'lion', 'tiger', 'wolf'],
                ['bridge', 'castle', 'house', 'road', 'skyscraper'],
	              ['cloud', 'forest', 'mountain', 'plain', 'sea'],
	              ['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'],
	              ['fox', 'porcupine', 'possum', 'raccoon', 'skunk'],
	              ['crab', 'lobster', 'snail', 'spider', 'worm'],
	              ['baby', 'boy', 'girl', 'man', 'woman'],
	              ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
	              ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
	              ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree'],
	              ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train'],
	              ['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor']]

    classList = train_data.classes

    # train set
    train_data_inputs = train_data.data
    train_data_labels = train_data.targets
    train_data_labels_big = train_data_labels.copy()

    train_data_inputs_mini = []
    train_data_labels_mini = []
    train_data_dict = []

    for i,n in enumerate(train_data_labels):
        cls = classList[n]
        for j,m in enumerate(bigClass):
            if cls in m:
                train_data_labels_big[i] = j
          
    for i in range(len(train_data_inputs)):
        if train_data_labels_big[i] == class_num:
            if not train_data_labels[i] in train_data_dict:
                train_data_dict.append(train_data_labels[i])
            train_data_inputs_mini.append(train_data[i][0])
            train_data_labels_mini.append(train_data_dict.index(train_data_labels[i]))
        #else:
            #train_data_inputs_mini.append(train_data[i][0])
            #train_data_labels_mini.append(5)
    train_data_inputs_mini = torch.stack((train_data_inputs_mini)).float()
    train_data_labels_mini = torch.tensor(train_data_labels_mini,dtype=torch.long)


    # test set
    test_data_inputs = test_data.data
    test_data_labels = test_data.targets
    test_data_labels_big = test_data_labels.copy()

    test_data_inputs_mini = []
    test_data_labels_mini = []

    for i,n in enumerate(test_data_labels):
        cls = classList[n]
        for j,m in enumerate(bigClass):
            if cls in m:
                test_data_labels_big[i] = j
          
    for i in range(len(test_data_inputs)):
        if test_data_labels_big[i] == class_num:
            if not test_data_labels[i] in train_data_dict:
                test_data_dict.append(train_data_dict[i])
            test_data_inputs_mini.append(test_data[i][0])
            test_data_labels_mini.append(train_data_dict.index(test_data_labels[i]))
        #else:
            #test_data_inputs_mini.append(test_data[i][0])
            #test_data_labels_mini.append(5)
    
    test_data_inputs_mini = torch.stack((test_data_inputs_mini)).float()
    test_data_labels_mini = torch.tensor(test_data_labels_mini,dtype=torch.long)
          


    trainloader = DataLoader(torch.utils.data.TensorDataset(train_data_inputs_mini,train_data_labels_mini), 
                            batch_size=batch_size, 
                            shuffle=True, 
                            num_workers=4,
                            )

    testloader = DataLoader(torch.utils.data.TensorDataset(test_data_inputs_mini,test_data_labels_mini) , 
                            batch_size=batch_size, 
                            shuffle=False, 
                            num_workers=4,
                            )

    #print(test_data_labels_mini)
    return trainloader, testloader