import argparse
from pathlib import Path

from PIL import Image
from psgan import Inference
from fire import Fire
import numpy as np

import faceutils as futils
from psgan import PostProcess
from fvcore.common.config import CfgNode


_C = CfgNode()


# Paths for logging and saving
_C.LOG = CfgNode()
_C.LOG.LOG_PATH = 'log/'
_C.LOG.SNAPSHOT_PATH = 'snapshot/'
_C.LOG.VIS_PATH = 'visulization/'
_C.LOG.SNAPSHOT_STEP = 1024
_C.LOG.LOG_STEP = 8
_C.LOG.VIS_STEP = 2048

# Data settings
_C.DATA = CfgNode()
_C.DATA.PATH = './data'
_C.DATA.NUM_WORKERS = 4
_C.DATA.BATCH_SIZE = 1
_C.DATA.IMG_SIZE = 256

# Training hyper-parameters
_C.TRAINING = CfgNode()
_C.TRAINING.G_LR = 2e-4
_C.TRAINING.D_LR = 2e-4
_C.TRAINING.BETA1 = 0.5
_C.TRAINING.BETA2 = 0.999
_C.TRAINING.C_DIM = 2
_C.TRAINING.G_STEP = 1
_C.TRAINING.NUM_EPOCHS = 50
_C.TRAINING.NUM_EPOCHS_DECAY = 0

# Loss weights
_C.LOSS = CfgNode()
_C.LOSS.LAMBDA_A = 10.0
_C.LOSS.LAMBDA_B = 10.0
_C.LOSS.LAMBDA_IDT = 0.5
_C.LOSS.LAMBDA_CLS = 1
_C.LOSS.LAMBDA_REC = 10
_C.LOSS.LAMBDA_HIS = 1
_C.LOSS.LAMBDA_SKIN = 0.1
_C.LOSS.LAMBDA_EYE = 1
_C.LOSS.LAMBDA_HIS_LIP = _C.LOSS.LAMBDA_HIS
_C.LOSS.LAMBDA_HIS_SKIN = _C.LOSS.LAMBDA_HIS * _C.LOSS.LAMBDA_SKIN
_C.LOSS.LAMBDA_HIS_EYE = _C.LOSS.LAMBDA_HIS * _C.LOSS.LAMBDA_EYE
_C.LOSS.LAMBDA_VGG = 5e-3

# Model structure
_C.MODEL = CfgNode()
_C.MODEL.G_CONV_DIM = 64
_C.MODEL.D_CONV_DIM = 64
_C.MODEL.G_REPEAT_NUM = 6
_C.MODEL.D_REPEAT_NUM = 3
_C.MODEL.NORM = "SN"
_C.MODEL.WEIGHTS = "assets/models"


# Preprocessing
_C.PREPROCESS = CfgNode()
_C.PREPROCESS.UP_RATIO = 0.6 / 0.85  # delta_size / face_size
_C.PREPROCESS.DOWN_RATIO = 0.2 / 0.85  # delta_size / face_size
_C.PREPROCESS.WIDTH_RATIO = 0.2 / 0.85  # delta_size / face_size
_C.PREPROCESS.LIP_CLASS = [7, 9]
_C.PREPROCESS.FACE_CLASS = [1, 6]
_C.PREPROCESS.LANDMARK_POINTS = 68

# Postprocessing
_C.POSTPROCESS = CfgNode()
_C.POSTPROCESS.WILL_DENOISE = False


def get_config()->CfgNode:
    return _C


def main(save_path='transferred_image.png'):
    parser =  parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source_path",
        default="./result/test_image/non-makeup/xfsy_0106.png",
        metavar="FILE",
        help="path to source image")
    parser.add_argument(
        "--reference_dir",
        default="result/test_image/makeup",
        help="path to reference images")
    parser.add_argument(
        "--speed",
        action="store_true",
        help="test speed")
    parser.add_argument(
        "--device",
        default="cpu",
        help="device used for inference")
    parser.add_argument(
        "--model_path",
        default="result/model/G.pth",
        help="model for loading")

    args = parser.parse_args()

    # Using the second 
    
    config = get_config()
    config.freeze()
    inference = Inference(
        config, args.device, args.model_path)
    postprocess = PostProcess(config)

    source = Image.open(args.source_path).convert("RGB")
    reference_paths = list(Path(args.reference_dir).glob("*"))
    np.random.shuffle(reference_paths)
    for reference_path in reference_paths:
        if not reference_path.is_file():
            print(reference_path, "is not a valid file.")
            continue

        reference = Image.open(reference_path).convert("RGB")

        # Transfer the psgan from reference to source.
        image, face = inference.transfer(source, reference, with_face=True)
        source_crop = source.crop(
            (face.left(), face.top(), face.right(), face.bottom()))
        image = postprocess(source_crop, image)
        image.save(save_path)

        if args.speed:
            import time
            start = time.time()
            for _ in range(100):
                inference.transfer(source, reference)
            print("Time cost for 100 iters: ", time.time() - start)


if __name__ == '__main__':
    main()
