import argparse
import json
import sys

import numpy as np
import torch
from PIL import Image
from torchvision import models

parser = argparse.ArgumentParser(
    description="Predict flower name from an image along with the probability of that name")
parser.add_argument("dir", metavar="dir", type=str)
parser.add_argument("checkpoint", metavar="checkpoint", type=str)
parser.add_argument("--top_k", type=int, default=3, dest="top_k")
parser.add_argument("--category_names", type=str,
                    default="cat_to_name.json", dest="category_names")
parser.add_argument("--gpu", dest="gpu", action="store_true", help="Use GPU?")

args = parser.parse_args()
dir = args.dir
checkpoint = args.checkpoint
top_k = args.top_k
category_names = args.category_names
gpu = args.gpu

device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)

    arch = checkpoint["arch"]
    if arch == "alexnet":
        model = models.alexnet(pretrained=True)
        classifier = "classifier"
    elif arch == "vgg11":
        model = models.vgg11(pretrained=True)
        classifier = "classifier"
    elif arch == "vgg13":
        model = models.vgg13(pretrained=True)
        classifier = "classifier"
    elif arch == "vgg16":
        model = models.vgg16(pretrained=True)
        classifier = "classifier"
    elif arch == "vgg19":
        model = models.vgg19(pretrained=True)
        classifier = "classifier"
    elif arch == "resnet18":
        model = models.resnet18(pretrained=True)
        classifier = "fc"
    elif arch == "resnet34":
        model = models.resnet34(pretrained=True)
        classifier = "fc"
    elif arch == "resnet50":
        model = models.resnet50(pretrained=True)
        classifier = "fc"
    elif arch == "resnet101":
        model = models.resnet101(pretrained=True)
        classifier = "fc"
    elif arch == "resnet152":
        model = models.resnet152(pretrained=True)
        classifier = "fc"
    elif arch == "densenet121":
        model = models.densenet121(pretrained=True)
        classifier = "classifier"
    elif arch == "densenet169":
        model = models.densenet169(pretrained=True)
        classifier = "classifier"
    elif arch == "densenet201":
        model = models.densenet201(pretrained=True)
        classifier = "classifier"
    elif arch == "densenet161":
        model = models.densenet161(pretrained=True)
        classifier = "classifier"
    else:
        print(f"Error: Unknown architecture: {arch}")
        sys.exit()

    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False

    # Define classifier
    if classifier == "classifier":
        model.classifier = checkpoint["classifier"]
    elif classifier == "fc":
        model.fc = checkpoint["classifier"]

    model.load_state_dict(checkpoint["state_dict"])
    model.class_to_idx = checkpoint["class_to_idx"]

    model.to(device)

    return model


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    with Image.open(image) as pil_image:
        # Resizing
        w = pil_image.size[0]
        h = pil_image.size[1]
        pil_image.thumbnail((256 if h > w else w, 256 if w > h else h))

        # Cropping
        w = pil_image.size[0]
        h = pil_image.size[1]
        pil_image = pil_image.crop(
            ((w - 224)/2, (h - 224)/2, (w + 224)/2, (h + 224)/2))

        # Normalizing
        np_image = np.array(pil_image) / 225
        mean = np.array([.485, .456, .406])
        std = np.array([.229, .224, .225])
        np_image = (np_image - mean) / std
        np_image = np_image.transpose(2, 0, 1)

        return torch.from_numpy(np_image)


def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    with torch.no_grad():
        model.eval()
        image = process_image(image_path)
        image = image.reshape(-1,
                              image.shape[0], image.shape[1], image.shape[2]).to(device)

        logps = model(image.float())
        ps = torch.exp(logps)

        probs, classes = ps.topk(topk)

        probs = probs.cpu().numpy().tolist()[0]
        classes = classes.cpu().numpy().tolist()[0]
        class_to_idx_invert = {
            model.class_to_idx[i]: i for i in model.class_to_idx}

        for i, val in enumerate(classes):
            classes[i] = class_to_idx_invert[val]

        return probs, classes


if __name__ == "__main__":
    try:
        with open(category_names, 'r') as f:
            cat_to_name = json.load(f)
        model = load_checkpoint(checkpoint)
        probs, classes = predict(dir, model, top_k)
        print("Predicted flower name(s):", [cat_to_name[i] for i in classes])
        print("Associated probabilities:", probs)
    except FileNotFoundError as e:
        print(e)
    except PermissionError as e:
        print(e)
    sys.exit(0)
