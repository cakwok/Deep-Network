import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import sys
import os           #for importing files by all filenames
from PIL import Image
from Q1a_CnnCoreStructure import *
import Q1a_CnnCoreStructure
import csv
from torch.nn.functional import normalize #to normalize a tensor


#Question 3A Create a greek symbol dataset
def ConvertGreekImages(folder):


    f1 = open('GreekPixels.csv', 'w')
    f1.write("Greek Letter Intensity \n")

    f2 = open('GreekLabel.csv', 'w')
    f2.write('Greek Label \n')


    for filename in os.listdir(folder):

        img = cv.imread(os.path.join(folder, filename), 0) #!! using imread has caused image becomes 3 channels!!  0 represents read as grayscale

        if img is not None:

            ret, img_T = cv.threshold(img, 180, 255, cv.THRESH_BINARY_INV)
            img_T = cv.resize(img_T, (28, 28))
            img_T = cv.cvtColor(img_T, cv.COLOR_BGR2RGB)

            print (filename)
            plt.imshow(img_T)
            plt.show()

            for r in range(28):
                for c in range(28):
                    pixel = str(img_T[r][c][0]) + ", "      #third [] indicates channel.  in grayscale, 3 channels show the same value
                    f1.write(pixel)
            f1.write("\n")


            if filename.split("_")[0] == "alpha":
                f2.write("0" + "\n")
            elif  filename.split("_")[0] == "beta":
                f2.write("1" + "\n")
            elif  filename.split("_")[0] == "gamma":
                f2.write("2" + "\n")

    f1.close()
    f2.close()

    return

class NeuralNetwork_FirstLayer(Q1a_CnnCoreStructure.NeuralNetwork):
    def __init__(self):
         NeuralNetwork.__init__(self)

    def forward(self, x):
        x = F.relu( F.max_pool2d( self.conv1(x), 2 ) )
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2)) #A max pooling of dropout layer with a 2x2 window and a ReLU function applied
        x = x.view(-1, 320)                             #Fatten a tensor, since the input channel of previous layer is 320.
        x = F.relu(self.fc1(x))
        print ("Submodel(Q1a_CnnCoreStructure.NeuralNetwork)")
        return x

def trucate(network, test_loader):
    network_output = []
    network.eval()
    with torch.no_grad():
        for data, target in test_loader:
            output = network(data)
            network_output.append(output)
    return network_output[0]

def load_learnt_network_status(network, learning_rate, momentum):

    continued_optimizer = optim.SGD(network.parameters(), lr=learning_rate,  momentum=momentum)

    #load the internal state of the network and optimizer when we last saved them.
    network_state_dict = torch.load('./results/model.pth')
    network.load_state_dict(network_state_dict)

    optimizer_state_dict = torch.load('./results/optimizer.pth')
    continued_optimizer.load_state_dict(optimizer_state_dict)

    return network, continued_optimizer

def loadGreekSymbols():

    GreekSymbolPixel = []
    GreekSymbolLabel = []

    with open('GreekPixels.csv') as f1:
        header = next(f1)
        reader = csv.reader(f1)                 #need to read as csv, otherwise it's typed as a string
        GreekSymbolPixel = list(reader)

    with open('GreekLabel.csv') as f1:
        header = next(f1)
        for line in f1:
            GreekSymbolLabel.append(line.strip())   #.strip()!  otherwise "\n" would be included in the list

    return GreekSymbolPixel, GreekSymbolLabel

def ProjectGreekSymbols(network, HandwrittingImages_tensor, GroundTruth):           #Question 1G, test my handwritting

    network.eval()
    pred_list = []
    elementVector = []

    with torch.no_grad():
        for data in HandwrittingImages_tensor:
            output = network(data)
            #pred = output.data.max(1, keepdim=True)[1]
            #pred_list.append(pred)
            elementVector.append(output)

    return elementVector

def print_ssd(GreekSymbolLabel, symbol, elementVector, label):
    
    ssd_list = []

    for index, labels in enumerate(GreekSymbolLabel):
        ssd = np.sum((np.array(symbol, dtype=np.float32) - np.array(elementVector[index], dtype=np.float32))**2)
        ssd_list.append([GreekSymbolLabel[index], ssd])         #create a list of list

    alpha_ssd_list_sorted = sorted(ssd_list,key=lambda l:l[0], reverse=False) #sort 2D array

    print(label + " SSD with other symbols")

    for items in alpha_ssd_list_sorted:
        print(items[0], items[1])

    print ("\n")
    
    return

def main(argv):

    #ConvertGreekImages('greek-1')      #Dont remove!!!!!!!!!!

    batch_size_train = 64           #Num of training examples in 1 batch
    batch_size_test = 1000          #Num of testing examples in 1 batch
    learning_rate = 0.01            #How much to shift in each gradient descent
    momentum = 0.5

    test_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('./files/', train=False, download=False,      #download remarked False after downloaded once
    transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.1307,), (0.3081,))])), batch_size=batch_size_test, shuffle=False)

    showNNFirstDenseLayer = NeuralNetwork_FirstLayer()
    print(showNNFirstDenseLayer)

    showNNFirstDenseLayer, continued_optimizer = load_learnt_network_status(showNNFirstDenseLayer, learning_rate, momentum)

    showNNFirstDenseLayer_truncated_output = trucate(showNNFirstDenseLayer, test_loader)
    print("showNNFirstDenseLayer_truncated_output")
    print(showNNFirstDenseLayer_truncated_output.size())

    # ---- Question 1C, project geek symbols into the embedding space
    # ---- read csv
    GreekSymbolPixel, GreekSymbolLabel = loadGreekSymbols()
    print (len(GreekSymbolPixel))
    print (type(GreekSymbolPixel[0]))
    print (GreekSymbolPixel[0], "\n")
    print (GreekSymbolPixel[1])
    print (GreekSymbolLabel)

    # ---- convert 1D list into tensor format
    GreekSymbolPixel_tensor = []

    for i in GreekSymbolPixel:
        tensor = torch.FloatTensor(list(map(int, i))).resize(1, 28,28)       #convert a list into tensor, convert a list of strings into list of integers
        norm = normalize(tensor, p=3.0)                                      #normalize a tensor
        GreekSymbolPixel_tensor.append(norm)
        #GreekSymbolPixel_tensor.append(torch.FloatTensor(list(map(int, i))).resize(1, 28,28))
    print (GreekSymbolPixel_tensor[0])
    print (GreekSymbolPixel_tensor[0].size())

    # ---- Get a set of 27 x 50 element vectors
    elementVector = ProjectGreekSymbols(showNNFirstDenseLayer, GreekSymbolPixel_tensor, GreekSymbolLabel)
    print ("elementVector")
    print (elementVector[0])
    print (len(elementVector), elementVector[0].size())

    # ---- Question 3D compute distances in the embedding space
    alpha_elementVector = []
    beta_elementVector = []
    gamma_elementVector = []
    alpha_ssd = []
    beta_ssd = []
    gamma_ssd = []

    #Compute SSD between same letter
    for index, labels in enumerate(GreekSymbolLabel):
        if labels == "0":
            if len(alpha_elementVector) != 0:
                ssd = np.sum((np.array(alpha_elementVector, dtype=np.float32) - np.array(elementVector[index], dtype=np.float32))**2)
                print (ssd)
                alpha_ssd.append(ssd)
            else:
                alpha_elementVector = elementVector[index]
        elif labels == "1":
            if len(beta_elementVector) != 0:
                ssd = np.sum((np.array(beta_elementVector, dtype=np.float32) - np.array(elementVector[index], dtype=np.float32))**2)
                print (ssd)
                beta_ssd.append(ssd)
            else:
                beta_elementVector = elementVector[index]
        elif labels == "2":
            if len(gamma_elementVector) != 0:
                ssd = np.sum((np.array(gamma_elementVector, dtype=np.float32) - np.array(elementVector[index], dtype=np.float32))**2)
                print (ssd)
                gamma_ssd.append(ssd)
            else:
                gamma_elementVector = elementVector[index]

    print("alpha_ssd")
    print(alpha_ssd)

    print("beta ssd")
    print(beta_ssd)

    print("gamma_ssd")
    print(gamma_ssd)

    #Compute SSD between different letter
    print("Alpha SSD with other symbols")

    ssd_list = []

    #--- print out alpha ssd
    '''
    for index, labels in enumerate(GreekSymbolLabel):
        ssd = np.sum((np.array(alpha_elementVector, dtype=np.float32) - np.array(elementVector[index], dtype=np.float32))**2)
        ssd_list.append([GreekSymbolLabel[index], ssd])         #create a list of list

    alpha_ssd_list_sorted = sorted(ssd_list,key=lambda l:l[0], reverse=False) #sort 2D array

    print("Alpha SSD with other symbols")

    for items in alpha_ssd_list_sorted:
        print(items[0], items[1])
    '''
    print_ssd(GreekSymbolLabel, alpha_elementVector, elementVector, "alpha")
    print_ssd(GreekSymbolLabel, beta_elementVector, elementVector, "beta")

    #--- print out beta ssd
    '''
    ssd_list = []
    for index, labels in enumerate(GreekSymbolLabel):
        ssd = np.sum((np.array(beta_elementVector, dtype=np.float32) - np.array(elementVector[index], dtype=np.float32))**2)
        ssd_list.append([GreekSymbolLabel[index], ssd])         #create a list of list

    beta_ssd_list_sorted = sorted(ssd_list,key=lambda l:l[0], reverse=False) #sort 2D array

    print("Beta SSD with other symbols")

    for items in beta_ssd_list_sorted:
        print(items[0], items[1])
    '''
    return

if __name__ == "__main__":
    main(sys.argv)
