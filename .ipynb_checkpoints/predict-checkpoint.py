import unet
import torch, torchvision
import torch.nn as nn
import constants as cst
import os
import dataset
import torchvision.transforms as transforms
from PIL import Image

#Load a model from directory and generate a mask
def load_model(filepath):
    net = unet.UNET(3, 2)
    net.cpu()
    net.load_state_dict(torch.load(filepath, map_location='cpu'))
    return net

def predict_img(net, img, device, transform, out_threshold=0.5):
    with torch.no_grad():
        x = img
        logits = net(x.to(device))
        logits = transform(logits)
        y_pred = nn.Softmax(dim=1)(logits)
        proba = y_pred.detach().cpu().squeeze(0).numpy()[1, :, :]
        return proba > out_threshold


if __name__ == "__main__":
    SIZE = (384, 512)
    DEVICE = torch.device("cpu")

    name = "upper_vertebrae_CE_ADAM_LR0.002_WD0_Params_Epoch15_BS4_W0_BEST_Epoch13_Val0.00764998427725264"
    save_name = name + "_" + "TH" + str(cst.THRESHOLD)
    model_name = name + ".pth"
    save_path1 = os.path.join(cst.PREDICT, "ALL" + "_" + save_name)
    save_path2 = os.path.join(cst.PREDICT, save_name)
    os.makedirs(save_path1, exist_ok=True)
    os.makedirs(save_path2, exist_ok=True)

    model_path = os.path.join(cst.MODEL, model_name)
    model = load_model(model_path)

    testing_set = dataset.ZebrafishDataset(cst.TEST, cst.VMASK)
    testing_loader = torch.utils.data.DataLoader(testing_set,
                                                 batch_size=1,
                                                 shuffle=True,
                                                 num_workers=cst.WORKERS)

    transform = transforms.Compose([transforms.Resize(SIZE),
                                    transforms.Pad((0, 64, 0, 64))])
    untransform = transforms.Compose([transforms.CenterCrop(SIZE),
                                      transforms.Resize((1932, 2576))])

    i = 1
    for img, mask, name in testing_loader:
        img_name = name[0]
        img_name = img_name[:-4]
        pred = predict_img(model, transform(img), DEVICE, untransform, cst.THRESHOLD)
        curr_name = img_name + "_predicted.jpg"
        real = img_name + "_real.jpg"
        drawn = Image.fromarray(pred)
        drawn.save(os.path.join(save_path1, curr_name))
        drawn.save(os.path.join(save_path2, img_name + ".jpg"))
        torchvision.utils.save_image(mask, fp=os.path.join(save_path1, real))
