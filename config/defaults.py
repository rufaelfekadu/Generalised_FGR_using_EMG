from yacs.config import CfgNode as CN

_C = CN()

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------

_C.MODEL = CN()
_C.MODEL.NAME = "FGR"
_C.MODEL.NUM_CLASSES = 2
_C.MODEL.IN_CHANNELS = 1
_C.MODEL.OUT_CHANNELS = 1
_C.MODEL.DROPOUT = 0.2
_C.MODEL.NUM_LAYERS = 1
_C.MODEL.HIDDEN_DIM = 64
_C.MODEL.ATTENTION_HEADS = 1

# -----------------------------------------------------------------------------
# Model.Transformer
# -----------------------------------------------------------------------------

_C.MODEL.TRANSFORMER = CN()
_C.MODEL.TRANSFORMER.NUM_LAYERS = 1
_C.MODEL.TRANSFORMER.IMAGE_SIZE = 4
_C.MODEL.TRANSFORMER.PATCH_SIZE = 2
_C.MODEL.TRANSFORMER.HIDDEN_DIM = 64
_C.MODEL.TRANSFORMER.ATTENTION_HEADS = 1
_C.MODEL.TRANSFORMER.DROPOUT = 0.2

# -----------------------------------------------------------------------------
# DATA
# -----------------------------------------------------------------------------

_C.DATA = CN()
_C.DATA.PATH = "../data/doi_10"
_C.DATA.SUBJECT = 1
_C.DATA.POSITION = 1
_C.DATA.SESSION = 1

# -----------------------------------------------------------------------------
# TRAIN
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.BATCH_SIZE = 32
_C.TRAIN.NUM_EPOCHS = 100
_C.TRAIN.LR = 0.001
_C.TRAIN.WEIGHT_DECAY = 0.0001
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.NUM_WORKERS = 4
_C.TRAIN.BETAS = (0.5, 0.999)
_C.TRAIN.SAVE_FREQ = 5
_C.TRAIN.TEST_FREQ = 10
_C.TRAIN.NUM_VAL = 3
_C.TRAIN.SEED = 0
_C.TRAIN.CHECKPOINT = True
_C.TRAIN.N_SPLITS = 5
_C.TRAIN.LAM = 0.25
_C.TRAIN.THR = 0.79
_C.TRAIN.THR_DOMAIN = 0.87
_C.TRAIN.ALPHA = 0.25
_C.TRAIN.DISC_TRAIN_FREQ = 2
_C.TRAIN.DEVICE = "cuda"

# -----------------------------------------------------------------------------
# OUTPUT
# -----------------------------------------------------------------------------

_C.OUTPUT = CN()
_C.OUTPUT.LOG_DIR = "outputs/single_position"
