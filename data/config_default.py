from yacs.config import CfgNode as CN
from utils.general import increment_path, check_file
import torch, os
from pathlib import Path

_C = CN()

_C.config_file = 'defaults'
_C.weights = 'yolov5s.pt' #initial weights path
_C.cfg = ''  #'model.yaml path
_C.hyp = 'data/hyp.scratch.yaml'  #'hyperparameters path
_C.project = 'runs/train'  #'save to project/name
_C.name = 'exp'  #'save to project/name
_C.exist_ok = False  #'existing project/name ok, do not increment
_C.device = ''  #'cuda device, i.e. 0 or 0,1,2,3 or cpu
_C.single_cls = False  #'train multi_class data as single_class

# Variables for train.py
_C.TRAIN = CN()
_C.TRAIN.epochs = 300
_C.TRAIN.batch_size = 16  #'total batch size for all GPUs
_C.rect = False  #'rectangular training
_C.resume = False  #'resume most recent training
_C.nosave = False  #'only save final checkpoint
_C.notest = False  #'only test final epoch
_C.noautoanchor = False  #'disable autoanchor check
_C.evolve = False  #'evolve hyperparameters
_C.bucket = ''  #'gsutil bucket
_C.cache_images = False  #'cache images for faster training
_C.image_weights = False  #'use weighted image selection for training
_C.adam = False  #'use torch.optim.Adam() optimizer
_C.sync_bn = False  #'use SyncBatchNorm, only available in DDP mode
_C.local_rank = -1  #'DDP parameter, do not modify
_C.log_imgs = 16  #'number of images for W&B logging, max 100
_C.log_artifacts = False  #'log artifacts, i.e. final trained model
_C.workers = 8  #'maximum number of dataloader workers
_C.multi_scale = False  #'vary img_size +/- 50%%
_C.quad = False  #'quad dataloader

# Variables for test.py
_C.TEST = CN()
_C.TEST.batch_size = 32
_C.TEST.conf_thres = 0.001 # object confidence threshold
_C.TEST.iou_thres = 0.6 # IOU threshold for NMS
_C.TEST.task = 'val' # 'val', 'test', 'study'")
_C.TEST.verbose = False #report mAP by class
_C.TEST.save_txt = False #save results to *.txt
_C.TEST.save_hybrid = False #save label+prediction hybrid results to *.txt
_C.TEST.save_conf = False #save confidences in save_txt labels
_C.TEST.save_json = False #save a cocoapi-compatible JSON results file

# Variables for detect.py
_C.DETECT = CN()
_C.DETECT.source = 'data/images'
_C.DETECT.conf_three = 0.25, #object confidence threshold
_C.DETECT.iou_thres = 0.45 #IOU threshold for NMS
_C.DETECT.view_img = False #display results
_C.DETECT.exist_ok = False #existing project/name ok, do not increment
_C.DETECT.classes = [] #filter by class:  [0,2,3]
_C.DETECT.agnostic_nms = False
_C.DETECT.update = False #update all models')

# Dataset configuration
_C.DATASET = CN()
_C.DATASET.dataset = 'cityscapes'
_C.DATASET.download = ''
_C.DATASET.augment = True

# train and val data as 1) directory = path/images/, 2) file = path/images.txt, or 3) list = [path1/images/, path2/images/]
_C.DATASET.train = 'datasets/cityscapes/leftImg8bit/train_extra/'
_C.DATASET.val = 'datasets/cityscapes/leftImg8bit/val/'
_C.DATASET.test = ''
_C.DATASET.root = ''
_C.DATASET.channels = 4
_C.DATASET.img_size = [640, 640]  #'[train, test] image sizes

_C.DATASET.img_suffix = 'image'
_C.DATASET.label_suffix = 'label'
_C.DATASET.use_depth = False
_C.DATASET.depth_suffix = 'depth'
_C.DATASET.depth_ext = 'png'

# number of classes
_C.DATASET.nc = 10
_C.DATASET.void_classes = []
_C.DATASET.valid_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# class names
_C.DATASET.names = []

_C.SYSTEM = CN()

def get_cfg_defaults():
  """Get a yacs CNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  C = _C.clone()

  C.cfg, C.hyp = check_file(C.cfg), check_file(C.hyp)  # check files

  # Set DDP variables
  C.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
  C.global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1

  C.DATASET.img_size.extend([C.DATASET.img_size[-1]] * (2 - len(C.DATASET.img_size)))  # extend to 2 sizes (train, test)
  C.name = 'evolve' if C.evolve else C.name
  C.save_dir = increment_path(Path(C.project) / C.name, exist_ok =C.exist_ok | C.evolve)  # increment run

  C.TRAIN.total_batch_size = C.TRAIN.batch_size
  assert C.TRAIN.batch_size % C.world_size == 0, '--batch-size must be multiple of CUDA device count'
  C.TRAIN.batch_size = C.TRAIN.total_batch_size // C.world_size

  return C