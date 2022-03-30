'''
Wing Man Casca, Kwok
CS5330 Project 5 - Recognition using Deep Networks
'''
#import statements
import torch
import torchvision                 #Load Dataloader, 1. transformation eg cropping/normalization 2. GPU functions
import sys
import matplotlib.pyplot as plt     #Plot Graph

# main function
def main(argv):
    #------ Question 1A Get the MNIST digit data set
    n_epochs = 3                    #Epoch means num of loops here
    batch_size_train = 64           #Num of training examples in 1 batch
    batch_size_test = 1000          #Num of testing examples in 1 batch
    learning_rate = 0.01
    momentum = 0.5
    log_interval = 10

    random_seed = 1
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)

    #------ Download Training and Testing Dataset.  * Mind the PATH here!
    train_loader = torch.utils.data.DataLoader(
      torchvision.datasets.MNIST('./files/', train=True, download=False,    #download remarked False after downloaded once
      transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
      torchvision.transforms.Normalize((0.1307,), (0.3081,))])),            #0.1307, 0.3081 = Global mean, Global SD
      batch_size=batch_size_train, shuffle=False)

    test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./files/', train=False, download=False,      #download remarked False after downloaded once
    transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.1307,), (0.3081,))])), batch_size=batch_size_test, shuffle=False)
    #---------------------------------------------

    #enumerate is for loop like, but it returns indexes
    #the first enumerate statement defines the iteration
    examples = enumerate(test_loader)

    #next - read the first enumerated element
    batch_idx, (example_data, example_targets) = next(examples)

    #See one test data batch consists of torch.Size([1000, 1, 28, 28])
    #1000 examples of 28x28 pixels in grayscale (i.e. no rgb channels, hence the one)
    print (example_data.shape)

    fig = plt.figure()
    for i in range(6):
        plt.subplot(2, 3, i+1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation = 'none')
        plt.title("Ground Truth: {}".format(example_targets[i]))
        plt.xticks([])          #remove xtick
        plt.yticks([])          #remove ytick
    plt.show()

    return

if __name__ == "__main__":
    main(sys.argv)
