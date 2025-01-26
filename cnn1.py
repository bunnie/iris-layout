import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np

import iris_dataset
import cifar

# Dataset generation notes
#  For any brand new block, you need to extract an idealized PNG of the layout from design
#  data (e.g., GDS file)
#    1. Extract the GDS of the layer & sub-block of interest using klayout. Put it in
#       imaging/blockname-layer.png, e.g. imaging/wrapped_snn_network-poly.gds. The techfile
#       argument is required and is "--tech sky130" for the open source data set. Note that
#       the default layer is "poly" (which is correct for SKY130)
#    2. Run "gds_to_png.py". This will automatically search for all .gds files in imaging/
#       and generate idealized versions of the layers for reference alignment.
#
#  With the block image, GDS data and idealized layout image, you can now create the data set:
#    1. Run "extract_dataset.py" with the names of the blocks that you want to generate
#       data for, i.e. "--names wrapped_snn_network"
#
#    This will generate a .pkl file with the dataset, and a .meta file with a description
#    of the training data set.

# Current strategy:
#   Just try to tell between ff, logic, fill, other
#      - Reduce input channels from RGB to just gray - how to do that? This
#        should reduce the # of parameters we need to tune
#      - Refine the CNN to match our use case: right now the intermediate layers
#        are optimized for a task that's not ours (handwriting recognition)
#      - Maybe need to eliminate extremely small fill from the training set?
#      - Alternatively, do we specify a cell size? Need to think about what
#        that even means.
#          - Maybe what we want in the end is a classifier that
#            given a patch of image, guesses how many of what type of cell are in
#            a region with a certain probability?
#          - The underlying issue is that cell sizes are quite different in scale,
#            and the size of the cell matters. The problem is the current CNN
#            is designed explicitly to disregard scale (written numbers have
#            the same meaning regardless of size), so again, need to tune the CNN
#            to throw away the part that allows us to scale an object.

PATH = './iris_net.pth'

# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        #self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc1 = nn.Linear(1040, 120) # BUT WHY
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 3) # set to number of classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ == "__main__":
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 4

    #debugset = cifar.CIFAR10(root='./data', train=True, download=True, transform=transform)
    #debugloader = torch.utils.data.DataLoader(debugset, batch_size=batch_size, shuffle=True, num_workers=2)
    import pickle
    from typing import Any
    data: Any = []
    targets = []

    trainset = iris_dataset.Iris(root='./imaging', train=True,
                                            download=True, transform=transform)
    print(len(trainset.classes))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    testset = iris_dataset.Iris(root='./imaging', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

    classes = ('ff', 'logic', 'fill')

    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    # print images
    print('Image check: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))
    imshow(torchvision.utils.make_grid(images))

    if True:
        net = Net()
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # Assuming that we are on a CUDA machine, this should print a CUDA device:
        print(device)
        net.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

        for epoch in range(4):  # loop over the dataset multiple times

            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                # inputs, labels = data
                inputs, labels = data[0].to(device), data[1].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:    # print every 2000 mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                    running_loss = 0.0

        print('Finished Training')

        torch.save(net.state_dict(), PATH)
    if True:
        net = Net()
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # Assuming that we are on a CUDA machine, this should print a CUDA device:
        print(device)
        net.load_state_dict(torch.load(PATH, weights_only=True))
        net.to(device)

        dataiter = iter(testloader)
        images, labels = next(dataiter)

        # print images
        print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))
        imshow(torchvision.utils.make_grid(images))

        images_cuda = images.to(device)
        outputs = net(images_cuda)
        _, predicted = torch.max(outputs, 1)

        print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
                                    for j in range(4)))

        correct = 0
        total = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                # calculate outputs by running images through the network
                images_cuda = images.to(device)
                labels_cuda = labels.to(device)
                outputs = net(images_cuda)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels_cuda.size(0)
                correct += (predicted == labels_cuda).sum().item()

        print(f'Accuracy of the network on the test images: {100 * correct // total} %')

        # prepare to count predictions for each class
        correct_pred = {classname: 0 for classname in classes}
        total_pred = {classname: 0 for classname in classes}

        # again no gradients needed
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images_cuda = images.to(device)
                labels_cuda = labels.to(device)
                outputs = net(images_cuda)
                _, predictions = torch.max(outputs, 1)
                # collect the correct predictions for each class
                for label, prediction in zip(labels_cuda, predictions):
                    if label == prediction:
                        correct_pred[classes[label]] += 1
                    total_pred[classes[label]] += 1


        # print accuracy for each class
        for classname, correct_count in correct_pred.items():
            accuracy = 100 * float(correct_count) / total_pred[classname]
            print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')