from yacs.config import CfgNode as CN
from utils.general import increment_path, check_file
import torch, os
from pathlib import Path

_C = CN()

_C.weights = 'yolov5s.pt' #initial weights path
_C.C = ''  #'model.yaml path')
_C.data = 'data/coco128.yaml'  #'data.yaml path')
_C.hyp = 'data/hyp.scratch.yaml'  #'hyperparameters path')
_C.epochs = 300
_C.batch_size = 16  #'total batch size for all GPUs')
_C.img_size = [640, 640]  #'[train, test] image sizes')
_C.rect = True  #'rectangular training')
_C.resume = False  #'resume most recent training')
_C.nosave = True  #'only save final checkpoint')
_C.notest = True  #'only test final epoch')
_C.noautoanchor = True  #'disable autoanchor check')
_C.evolve = True  #'evolve hyperparameters')
_C.bucket = ''  #'gsutil bucket')
_C.cache_images = True  #'cache images for faster training')
_C.image_weights = True  #'use weighted image selection for training')
_C.device = ''  #'cuda device, i.e. 0 or 0,1,2,3 or cpu')
_C.multi_scale = True  #'vary img_size +/- 50%%')
_C.single_cls = True  #'train multi_class data as single_class')
_C.adam = True  #'use torch.optim.Adam() optimizer')
_C.sync_bn = True  #'use SyncBatchNorm, only available in DDP mode')
_C.local_rank = -1  #'DDP parameter, do not modify')
_C.log_imgs = 16  #'number of images for W&B logging, max 100')
_C.log_artifacts = True  #'log artifacts, i.e. final trained model')
_C.workers = 8  #'maximum number of dataloader workers')
_C.project = 'runs/train'  #'save to project/name')
_C.name = 'exp'  #'save to project/name')
_C.exist_ok = True  #'existing project/name ok, do not increment')
_C.quad = True  #'quad dataloader')

_C.SYSTEM = CN()

def get_C_defaults():
  """Get a yacs CNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  C = _C.clone()

  C.data, C.C, C.hyp = check_file(C.data), check_file(C.C), check_file(C.hyp)  # check files

  # Set DDP variables
  C.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
  C.global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1

  C.img_size.extend([C.img_size[-1]] * (2 - len(C.img_size)))  # extend to 2 sizes (train, test)
  C.name = 'evolve' if C.evolve else C.name
  C.save_dir = increment_path(Path(C.project) / C.name, exist_ok=C.exist_ok | C.evolve)  # increment run

  C.total_batch_size = C.batch_size
  assert C.batch_size % C.world_size == 0, '--batch-size must be multiple of CUDA device count'
  C.batch_size = C.total_batch_size // C.world_size

  return C