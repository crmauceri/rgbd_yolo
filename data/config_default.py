from yacs.config import CfgNode as CN
from utils.general import increment_path, check_file
import torch, os
from pathlib import Path

_C = CN()

_C.weights = 'yolov5s.pt' #initial weights path
_C.cfg = ''  #'model.yaml path
_C.hyp = 'data/hyp.scratch.yaml'  #'hyperparameters path
_C.epochs = 300
_C.batch_size = 16  #'total batch size for all GPUs
_C.img_size = [640, 640]  #'[train, test] image sizes
_C.rect = False  #'rectangular training
_C.resume = False  #'resume most recent training
_C.nosave = False  #'only save final checkpoint
_C.notest = False  #'only test final epoch
_C.noautoanchor = False  #'disable autoanchor check
_C.evolve = False  #'evolve hyperparameters
_C.bucket = ''  #'gsutil bucket
_C.cache_images = False  #'cache images for faster training
_C.image_weights = False  #'use weighted image selection for training
_C.device = ''  #'cuda device, i.e. 0 or 0,1,2,3 or cpu
_C.multi_scale = False  #'vary img_size +/- 50%%
_C.single_cls = False  #'train multi_class data as single_class
_C.adam = False  #'use torch.optim.Adam() optimizer
_C.sync_bn = False  #'use SyncBatchNorm, only available in DDP mode
_C.local_rank = -1  #'DDP parameter, do not modify
_C.log_imgs = 16  #'number of images for W&B logging, max 100
_C.log_artifacts = False  #'log artifacts, i.e. final trained model
_C.workers = 8  #'maximum number of dataloader workers
_C.project = 'runs/train'  #'save to project/name
_C.name = 'exp'  #'save to project/name
_C.exist_ok = False  #'existing project/name ok, do not increment
_C.quad = False  #'quad dataloader

_C.DATASET = CN()
_C.DATASET.dataset = 'cityscapes'
_C.DATASET.download = ''

# train and val data as 1) directory = path/images/, 2) file = path/images.txt, or 3) list = [path1/images/, path2/images/]
_C.DATASET.train = 'datasets/cityscapes/leftImg8bit/train_extra/'
_C.DATASET.val = 'datasets/cityscapes/leftImg8bit/val/'
_C.DATASET.root = ''
_C.DATASET.channels = 4

_C.DATASET.img_suffix = 'image'
_C.DATASET.label_suffix = 'label'
_C.DATASET.depth_suffix = 'depth'

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

  C.img_size.extend([C.img_size[-1]] * (2 - len(C.img_size)))  # extend to 2 sizes (train, test)
  C.name = 'evolve' if C.evolve else C.name
  C.save_dir = increment_path(Path(C.project) / C.name, exist_ok =C.exist_ok | C.evolve)  # increment run

  C.total_batch_size = C.batch_size
  assert C.batch_size % C.world_size == 0, '--batch-size must be multiple of CUDA device count'
  C.batch_size = C.total_batch_size // C.world_size

  return C