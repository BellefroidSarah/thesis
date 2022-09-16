import dataset
import torch
import os
import constants as cst
import metrics
import numpy as np

# Name of the repository containing the predictions
run_name = "lateral_vertebrae_CE_ADAM_LR0.002_WD0_Params_Epoch15_BS4_W0_BEST_Epoch13_Val0.0060904803685843945_TH0.5"
prediction_dir = os.path.join(cst.PREDICT, run_name)
data = dataset.ZebrafishDataset(prediction_dir, cst.VMASK)
loader = torch.utils.data.DataLoader(data,
                                     batch_size=1,
                                     shuffle=True,
                                     num_workers=cst.WORKERS)

precisions = []
recalls = []
F1s = []
IOUs = []
for prediction, groundtruth, _ in loader:
    prediction = torch.squeeze(prediction)
    prediction = torch.squeeze(prediction)
    groundtruth = torch.squeeze(groundtruth)
    groundtruth = torch.squeeze(groundtruth)

    prec = metrics.precision(prediction, groundtruth)
    rec = metrics.recall(prediction, groundtruth)
    F1 = metrics.F1Score(prediction, groundtruth)
    IOU = metrics.IOUScore(prediction, groundtruth)

    precisions.append(prec)
    recalls.append(rec)
    F1s.append(F1)
    IOUs.append(IOU)

print("Name of the run: {}".format(run_name))
print("Precision: {}".format(np.mean(precisions)))
print("Recall: {}".format(np.mean(recalls)))
print("F1/Dice score: {}".format(np.mean(F1s)))
print("IoU: {}".format(np.mean(IOUs)))
