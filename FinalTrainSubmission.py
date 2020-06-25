#Specify all the import statements
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns
import argparse


#Define argparse arguements
parser = argparse.ArgumentParser(description='Train.py')

# Inventory of command line arguements
parser.add_argument('--dir', dest="directory", action="store", type=str, default="checkpoint.pth")
parser.add_argument('--lr', dest="learningrate", action="store", type=float, default=0.001)
parser.add_argument('--eps', dest="epochs", action="store", type=int, default=1)
parser.add_argument('--drop', dest = "dropout", action = "store",type=float, default = 0.5)
parser.add_argument('--gpuorcpu', dest="gpuorcpu", action="store", default="gpu")
parser.add_argument('--hidden', type=int, dest="hiddenunits", action="store", default=6272)
parser.add_argument('--model', dest="premodel", action="store", help="input model here", type = str)
parser.add_argument('datadir', nargs='*', action="store", default="./flowers/")
args = parser.parse_args()

#Load in the data
data_dir = args.datadir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# TODO: Define your transforms for the training, validation, and testing sets
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.Resize([224, 224]),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

validation_transforms = transforms.Compose([transforms.Resize([224, 224]),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize([224, 224]),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])


# TODO: Load the datasets with ImageFolder
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
validation_data = datasets.ImageFolder(valid_dir, transform=validation_transforms)
test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

# TODO: Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
validationloader = torch.utils.data.DataLoader(validation_data, batch_size=64)
testloader = torch.utils.data.DataLoader(test_data, batch_size=64)


#Load the pre-specified architecture
model =  getattr(models,args.premodel)(pretrained=True)
model

# Use GPU if it's requested (and available)
if args.gpuorcpu == 'gpu':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")

#Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False

#In_features_dictionary
if args.premodel == 'vgg19':
    input_layer = int(25088)
elif args.premodel == 'alexnet':
    input_layer = int(9216)
elif args.premodel == 'resnet34':
    input_layer = int(512)
else:
    input_layer = int(25088)


#Specify model hidden layers and features
from collections import OrderedDict
classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(int(input_layer), args.hiddenunits)),
                          ('relu', nn.ReLU()),
                          ('dropout', nn.Dropout(p=args.dropout)),
                          ('fc2', nn.Linear(args.hiddenunits, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))

model.classifier = classifier
criterion = nn.NLLLoss()

# Only train the classifier parameters, feature parameters are frozen
optimizer = optim.Adam(model.classifier.parameters(), lr=args.learningrate)
model.to(device);


#Train the classifier layers using backpropagation using the pre-trained network to get the features
#Track the loss and accuracy on the validation set to determine the best hyperparameters

epochs = args.epochs
steps = 0
running_loss = 0
print_every = 5
for epoch in range(epochs):
    for inputs, labels in trainloader:
        steps += 1
        # Move input and label tensors to the default device
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in validationloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)

                    test_loss += batch_loss.item()

                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Validation loss: {test_loss/len(validationloader):.3f}.. "
                  f"Validation accuracy: {accuracy/len(validationloader):.3f}")
            running_loss = 0
            model.train()

#Save the checkpoint
#model_input_dict = {Resnet: model.fc.in_features, VGG16: model.classifier[0].in_features, Densenet: model.classifier.in_features,    SqueezeNet: model.classifier[1].in_channels}

model.class_to_idx = train_data.class_to_idx

model.cpu()
torch.save({'arch': str(args.premodel),
            'state_dict': model.state_dict(),
            'input_size': int(input_layer),
            'output_size': 102,
            'hidden_units': args.hiddenunits,
            'class_to_idx': model.class_to_idx},
            args.directory)
