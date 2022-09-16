import torch, torchvision
import os
import random
import logging #Se renseigner, par encore utilisé
import dataset

import constants as cst
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F

import loss_fn
from unet import UNET

# First version of the training loop


if __name__ == "__main__":
    random.seed(cst.SEED)
    torch.manual_seed(cst.SEED)
    np.random.seed(cst.SEED)

    SIZE = (384, 512)
    DEVICE = torch.device("cpu")

    transform = transforms.Compose([transforms.Resize(SIZE),
                                    transforms.Pad((0, 64, 0, 64))])
    untransform = transforms.Compose([transforms.CenterCrop(SIZE),
                                     transforms.Resize((1932, 2576))])

    # Datasets and loaders
    training_set = dataset.ZebrafishDataset(cst.TRAIN, cst.VMASK)
    validation_set = dataset.ZebrafishDataset(cst.VALIDATE, cst.VMASK)
    testing_set = dataset.ZebrafishDataset(cst.TEST, cst.VMASK)

    training_loader = torch.utils.data.DataLoader(training_set,
                                                  batch_size=cst.BATCH_SIZE,
                                                  shuffle=True,
                                                  num_workers=cst.WORKERS)

    validation_loader = torch.utils.data.DataLoader(validation_set,
                                                    batch_size=cst.BATCH_SIZE,
                                                    shuffle=True,
                                                    num_workers=cst.WORKERS)

    testing_loader = torch.utils.data.DataLoader(testing_set,
                                                 batch_size=1,
                                                 shuffle=True,
                                                 num_workers=cst.WORKERS)

    # (Channels x Classes)
    model = UNET(3, 2)
    best_model = UNET(3, 2)
    best_model = model
    model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    criterion_string = "CE"

    if cst.LOSS == "Dice":
        print("Dice")
        criterion = loss_fn.DiceLoss()
        criterion_string = "DCE"
    if cst.LOSS == "IOU":
        print("IOU")
        criterion = loss_fn.IoULoss()
        criterion_string = "IOU"
    if cst.LOSS == "Tversky":
        print("Twersky")
        criterion = loss_fn.TverskyLoss()
        criterion_string = "Tversky"

    optimiser = torch.optim.Adam(model.parameters(), lr=cst.LEARNING_RATE, weight_decay=cst.WEIGHT_DECAY)
    optimiser_string = "ADAM" + "_" + "LR" + str(cst.LEARNING_RATE) + "_" + "WD" + str(cst.WEIGHT_DECAY)

    if cst.OPTIMIZER == "SGD":
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=cst.LEARNING_RATE,
                                    momentum=cst.MOMENTUM,
                                    weight_decay=cst.WEIGHT_DECAY)
        optimiser_string = "SGD" + "_" + "LR" + str(cst.LEARNING_RATE) + "_" + "M" + str(cst.MOMENTUM)
        optimiser_string += "_" + "WD" + str(cst.WEIGHT_DECAY)

    model.eval()
    with torch.no_grad():
        val_loss = []
        for images, masks, names in validation_loader:
            images = transform(images)
            outputs = model(images.to(DEVICE))
            outputs = untransform(outputs)

            masks = masks.type(torch.LongTensor)
            masks = torch.squeeze(masks, 1)

            if cst.LOSS == "CE":
                vloss = criterion(outputs, masks.to(DEVICE))
            else:
                vloss = criterion(outputs, F.one_hot(masks, 2).permute(0, 3, 1, 2).float())

            loss = vloss.detach().item()
            val_loss.append(loss)

        loss = np.mean(val_loss)
        print("Validation loss before training: {}".format(loss))

    best_val = loss
    best_epoch = 0

    params_string = "Params" + "_" + "Epoch" + str(cst.EPOCHS) + "_" + "BS" + str(cst.BATCH_SIZE)
    params_string += "_" + "W" + str(cst.WORKERS)

    epochs_train_losses = []
    epochs_val_losses = []
    for i in range(cst.EPOCHS):
        print("Starting epoch {}".format(i+1), end=". ")

        model.train()
        train_loss = []
        for images, masks, names in training_loader:
            images = transform(images)
            outputs = model(images.to(DEVICE))
            outputs = untransform(outputs)

            masks = masks.type(torch.LongTensor)
            masks = torch.squeeze(masks, 1)

            if cst.LOSS == "CE":
                tloss = criterion(outputs, masks.to(DEVICE))
            else:
                tloss = criterion(outputs, F.one_hot(masks, 2).permute(0, 3, 1, 2).float())

            loss = tloss.detach().item()
            train_loss.append(loss)

            optimiser.zero_grad()
            tloss.backward()
            optimiser.step()

        loss = np.mean(train_loss)
        epochs_train_losses.append(loss)
        print("Trained: {}".format(loss), end=". ")

        model.eval()
        with torch.no_grad():
            val_loss = []
            for images, masks, names in validation_loader:
                images = transform(images)
                outputs = model(images.to(DEVICE))
                outputs = untransform(outputs)

                masks = masks.type(torch.LongTensor)
                masks = torch.squeeze(masks, 1)

                if cst.LOSS == "CE":
                    vloss = criterion(outputs, masks.to(DEVICE))
                else:
                    vloss = criterion(outputs, F.one_hot(masks, 2).permute(0, 3, 1, 2).float())

                loss = vloss.detach().item()
                val_loss.append(loss)

            loss = np.mean(val_loss)
            epochs_val_losses.append(loss)
            print("Validation: {}.".format(loss))

            if loss < best_val:
                best_val = loss
                best_model = model
                best_epoch = i+1

    print("Training: {}".format(epochs_train_losses))
    print("Validating: {}".format(epochs_val_losses))
    print("Best score: {}".format(best_val))

    name = "upper_vertebrae_" + criterion_string + "_" + optimiser_string + "_" + params_string
    model_name = name + ".pth"
    best_name = name + "_" + "BEST" + "_Epoch" + str(best_epoch) + "_" + "Val" + str(best_val) + ".pth"
    model_filepath = os.path.join(cst.MODEL, model_name)
    best_filepath = os.path.join(cst.MODEL, best_name)
    torch.save(model.state_dict(), model_filepath)
    torch.save(best_model.state_dict(), best_filepath)

    """
    index = [i+1 for i in range(cst.EPOCHS)]
    plt.plot(index, epochs_train_losses, label="Training")
    plt.plot(index, epochs_val_losses, label="Validation")
    plt.title(cst.LOSS)
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.legend()
    plot_name = name + ".png"
    plt.savefig(plot_name)
    """
