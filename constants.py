import os
ID_PROJECT = 153858703

TERM_IDS = [153931462,
            153931470,
            153931478,
            153931484,
            153931501,
            153931509,
            153931524,
            153931530,
            153931540,
            153931546,
            153931563,
            153931571,
            153931579,
            153931585,
            153931595,
            153931601,
            153931611,
            153931631,
            153931639,
            153931652,
            153931674,
            153931687,
            153931697,
            153931717,
            153931725,
            153931738,
            153931750]

TERM_SURNAME = ["M1",
                "M2",
                "D1",
                "D2",
                "aa1",
                "aa2",
                "EN1",
                "EN2",
                "CH1",
                "CH2",
                "Br2a",
                "Br2b",
                "Br1a",
                "Br1b",
                "Hm1",
                "Hm2",
                "Op1",
                "Op2",
                "P",
                "N",
                "CB1",
                "CB2",
                "CL1",
                "CL2",
                "Oc1",
                "Oc2",
                "Vertebrae"]

TERM_NAMES = ["m1",  # 0
              "m2",
              "d1",
              "d2",
              #"aa1",
              #"aa2",  # 5
              "en1",
              "en2",
              "ch1",
              "ch2",
              "br2a",  # 10
              "br2b",
              "br1a",
              "br1b",
              "hm1",
              "hm2",  # 15
              "op1",
              "op2",
              "p",
              "n",
              "cb1",  # 20
              "cb2",
              "cl1",
              "cl2",
              "oc1",
              "oc2",  # 25
              "vc"]

V_TERM_NAMES = ["m1", #Those only in the head
                "m2",
                "d1",
                "d2",
                "en1",
                "en2",
                "ch1",
                "ch2",
                "br2a",
                "br2b",
                "br1a",
                "br1b",
                "hm1",
                "hm2",
                "op1",
                "op2",
                "p",
                "n",
                "cb1",
                "cb2",
                "cl1",
                "cl2",
                "oc1",
                "oc2"]

LAT_TERM_NAMES = ["VC"]

SEED = 1

# DIR = "data"
DIR = os.getcwd()
DATA = os.path.join(DIR, "data")
UNLABELED = "unlabeled"
LABELED = "labeled"
MODEL = os.path.join(DIR, "models")
PREDICT = os.path.join(DIR, "predictions")
PLOTS = os.path.join(DIR, "plots")

IMG_WIDTH = 2576
IMG_HEIGHT = 1932

BATCH_SIZE = 4
LEARNING_RATE = 0.0001
MOMENTUM = 0
WEIGHT_DECAY = 0
EPOCHS = 250
WORKERS = 0
FOLDS = 5

# Optimiser et loss peuvent devenir des arguments
OPTIMIZER = "Adam"
WEIGHTING = True
LOSS = "Focal"

MEAN = [0.78676176, 0.50835603, 0.78414893]
STD = [0.16071789, 0.24160224, 0.12767686]
THRESHOLD = 0.5
