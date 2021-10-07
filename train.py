import argparse
import os
import sys

import torch
from torch import nn, optim
from torch.optim import optimizer
from torchvision import datasets, models, transforms

parser = argparse.ArgumentParser(description="Trains a neural network")
parser.add_argument('data_dir', metavar='dir', type=str,
                    help="Directory to the dataset to be trained on")
parser.add_argument("--save_dir", dest="save_dir", default=".",
                    type=str, help="Directory to save checkpoints")
parser.add_argument("--arg", dest="arch", default="vgg16",
                    type=str, help="Pretrained architecture to use")
parser.add_argument("--learning_rate", dest="learning_rate",
                    default=0.001, type=float, help="Learning rate to use")
parser.add_argument("--hidden_units", dest="hidden_units",
                    type=int, default=512, help="Number of hidden units to use")
parser.add_argument("--epochs", dest="epochs", type=int,
                    default=5, help="Number of epochs to train model")
parser.add_argument("--gpu", dest="gpu", action="store_true", help="Use GPU?")

args = parser.parse_args()
data_dir = args.data_dir
save_dir = args.save_dir
arch = args.arch
learning_rate = args.learning_rate
hidden_units = args.hidden_units
epochs = args.epochs
gpu = args.gpu

device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")


def build_model(arch, hidden_units):
    if arch == "alexnet":
        model = models.alexnet(pretrained=True)
        in_features = model.classifier[1].in_features
        classifier_name = "classifier"
    elif arch == "vgg11":
        model = models.vgg11(pretrained=True)
        in_features = model.classifier[0].in_features
        classifier_name = "classifier"
    elif arch == "vgg13":
        model = models.vgg13(pretrained=True)
        in_features = model.classifier[0].in_features
        classifier_name = "classifier"
    elif arch == "vgg16":
        model = models.vgg16(pretrained=True)
        in_features = model.classifier[0].in_features
        classifier_name = "classifier"
    elif arch == "vgg19":
        model = models.vgg19(pretrained=True)
        in_features = model.classifier[0].in_features
        classifier_name = "classifier"
    elif arch == "resnet18":
        model = models.resnet18(pretrained=True)
        in_features = model.fc.in_features
        classifier_name = "fc"
    elif arch == "resnet34":
        model = models.resnet34(pretrained=True)
        in_features = model.fc.in_features
        classifier_name = "fc"
    elif arch == "resnet50":
        model = models.resnet50(pretrained=True)
        in_features = model.fc.in_features
        classifier_name = "fc"
    elif arch == "resnet101":
        model = models.resnet101(pretrained=True)
        in_features = model.fc.in_features
        classifier_name = "fc"
    elif arch == "resnet152":
        model = models.resnet152(pretrained=True)
        in_features = model.fc.in_features
        classifier_name = "fc"
    elif arch == "densenet121":
        model = models.densenet121(pretrained=True)
        in_features = model.classifier.in_features
        classifier_name = "classifier"
    elif arch == "densenet169":
        model = models.densenet169(pretrained=True)
        in_features = model.classifier.in_features
        classifier_name = "classifier"
    elif arch == "densenet201":
        model = models.densenet201(pretrained=True)
        in_features = model.classifier.in_features
        classifier_name = "classifier"
    elif arch == "densenet161":
        model = models.densenet161(pretrained=True)
        in_features = model.classifier.in_features
        classifier_name = "classifier"
    else:
        print(f"Error: Unknown architecture: {arch}")
        sys.exit()

    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False

    # Define classifier
    classifier = nn.Sequential(
        nn.Linear(in_features, hidden_units),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(hidden_units, hidden_units),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(hidden_units, 102),
        nn.LogSoftmax(dim=1)
    )

    if classifier_name == "classifier":
        model.classifier = classifier
        optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    elif classifier_name == "fc":
        model.fc = classifier
        optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)

    return model, optimizer


def train_model(model, epochs, dataloaders, optimizer, criterion):
    steps = 0
    running_loss = 0
    print_every = 20
    model.to(device)
    for epoch in range(epochs):
        for images, labels in dataloaders["train"]:
            steps += 1
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model(images)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()
                test_loss = 0
                accuracy = 0

                for images, labels in dataloaders["valid"]:
                    images, labels = images.to(device), labels.to(device)

                    logps = model(images)
                    loss = criterion(logps, labels)
                    test_loss += loss.item()

                    ps = torch.exp(logps)
                    _, top_class = ps.topk(1, dim=1)
                    equality = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equality.type(torch.FloatTensor)).item()

                print(f"Epoch: {epoch+1}/{epochs} ",
                      f"Training Loss: {running_loss/print_every:.3f} ",
                      f"Validation Loss: {test_loss/len(dataloaders['valid']):.3f} ",
                      f"Validation Accuracy: {accuracy/len(dataloaders['valid']):.3f}")
                running_loss = 0
                model.train()


def generate_data(dir):
    train_dir = os.path.join(dir, "train")
    valid_dir = os.path.join(dir, "valid")
    test_dir = os.path.join(dir, "test")
    data_transforms = {
        "train": transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
        "valid": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
        "test": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    }

    # Load the datasets with ImageFolder
    image_datasets = {
        "train": datasets.ImageFolder(train_dir, transform=data_transforms["train"]),
        "valid": datasets.ImageFolder(valid_dir, transform=data_transforms["valid"]),
        "test": datasets.ImageFolder(test_dir, transform=data_transforms["test"])
    }

    # Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {
        "train": torch.utils.data.DataLoader(image_datasets["train"], batch_size=64, shuffle=True),
        "valid": torch.utils.data.DataLoader(image_datasets["valid"], batch_size=64, shuffle=True),
        "test": torch.utils.data.DataLoader(image_datasets["test"], batch_size=64, shuffle=True)
    }

    return data_transforms, image_datasets, dataloaders


def save_model(save_dir, model, image_datasets):
    model.class_to_idx = image_datasets["train"].class_to_idx
    checkpoint = {
        "input_size": 25088,
        "output_size": 102,
        "classifier": model.classifier,
        "state_dict": model.state_dict(),
        "class_to_idx": model.class_to_idx,
        "arch": arch
        # "optimizer": optimizer.state_dict(),
        # "epochs": epochs,
    }

    torch.save(checkpoint, os.path.join(save_dir, "checkpoint.pth"))


if __name__ == "__main__":
    print("--------LOADING DATA--------")
    _, image_datasets, dataloaders = generate_data(data_dir)
    print("Data loaded successfully")

    print("--------BUILDIING MODEL--------")
    model, optimizer = build_model(arch, hidden_units)
    print("Model successfully built")

    criterion = nn.NLLLoss()

    print("--------TRAINING MODEL--------")
    print(f"Training model with {epochs} epochs")
    train_model(model, epochs, dataloaders, optimizer, criterion)
    print("Model successfully trained")

    print("--------SAVING MODEL--------")
    save_model(save_dir, model, image_datasets)
    print(f"Model saved to {os.path.join(save_dir, 'checkpoint.pth')}")
