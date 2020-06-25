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
from collections import OrderedDict
import sys
import json


#Define the required arguements
parser = argparse.ArgumentParser(description='predict.py')

#Specify arguements
#parser.add_argument('input', type=str)
parser.add_argument('--directory', dest="directory", action="store", type=str, default="checkpoint.pth")
parser.add_argument('--topk', dest="topk", action="store", type = int, default=5)
parser.add_argument('--gpuorcpu', dest="gpuorcpu", action="store", default="gpu")
parser.add_argument('datadir', nargs='*', action="store", default="./flowers/")


args = parser.parse_args()

#Load in the model

chpt = torch.load(args.directory)

predict_model = getattr(models,chpt['arch'])(pretrained=True)
for param in predict_model.parameters():
    param.requires_grad = False

predict_model.class_to_idx = chpt['class_to_idx']

# Create the classifier

classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(chpt['input_size'], chpt['hidden_units'])),
                              ('relu', nn.ReLU()),
                              ('fc2', nn.Linear(chpt['hidden_units'], chpt['output_size'])),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))
predict_model.classifier = classifier

predict_model.load_state_dict(chpt['state_dict'])

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

def process_image(image):
    '''
    Takes a PIL input, makes some formatting changes and then converts it to a numpy array
    '''
    # Open the image & resize using thumbnail
    pil_image = Image.open(image)
    pil_image.load()


    #Resizing the image and converting to a thumbnail format
    if pil_image.size[1] > pil_image.size[0]:
        pil_image.thumbnail((256, 5000))
    else:
        pil_image.thumbnail((5000, 256))


    #Cropping.  Note per online resources PIL Crop takes the form im1 = im.crop((left, top, right, bottom))
    size = pil_image.size
    left = size[0]//2 -(224/2)
    top = size[1]//2 - (224/2)
    right = size[0]//2 +(224/2)
    bottom = size[1]//2 + (224/2)
    pil_image = pil_image.crop((left,
                     top,
                     right,
                     bottom
                    ))

    #Normalizing
    pil_image = np.array(pil_image)/255
    std = np.array([0.229, 0.224, 0.225])
    mean = np.array([0.485, 0.456, 0.406])
    pil_image = (pil_image - mean)/std

    pil_image = pil_image.transpose((2, 0, 1))

    return pil_image


def predict(image_path, model, topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    #Manipulate the images so that they are torch ready
    prior_image_output = process_image(image_path)
    model.eval()
    torch_image = torch.zeros(1, 3, 224, 224)
    torch_image[0] = torch.from_numpy(prior_image_output)

    #Move the model and image to CUDA
    model.to('cuda')
    torch_image[0] = torch_image[0].to('cuda')

    #Run the data through the network model
    torch_image = torch_image.to('cuda')
    with torch.no_grad():
        inference = model(torch_image)

    #Take the inference and move it back to the CPU
    inference = inference.to('cpu')

    #Calculate the top probabilities
    probabilities = torch.exp(inference).topk(topk)
    top_probs, top_class = probabilities

    #Seperate top_probs and top_class into a usable format with probs and classes as the keys
    idx_to_class = dict(zip(model.class_to_idx.values(), model.class_to_idx.keys()))
    probs = top_probs[0].numpy()
    classes = [idx_to_class[i] for i in top_class[0].numpy()]

      #Extract the top probabilities and classes
    inferred_flower_names = np.array([cat_to_name[i] for i in classes])
    actual_flower_name = cat_to_name[image_path.split('/')[2]]

    #Derive the name of the real flower from the test JPEG by splitting its # reference
    extract_flower_number = image_path.split('/')[2]
    index = [extract_flower_number]
    actual_flower_name = ([cat_to_name[i] for i in index])

    return probs, classes, inferred_flower_names, actual_flower_name

probs, classes, inferred_flower_names, actual_flower_name = predict('flowers/test/1/image_06743.jpg', predict_model, args.topk)


for i in range (int(args.topk)):
    print("There is a {} probability that the the flower is a {}".format(probs[i], inferred_flower_names[i]))

print("The flower is actually a {}".format(actual_flower_name))
