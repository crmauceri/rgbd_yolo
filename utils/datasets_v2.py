# Dataset utils and dataloaders


from dataloaders import make_data_loader
from dataloaders.config.defaults import get_cfg_defaults

def create_dataloader(opt, path, imgsz, stride, hyp=None, pad=0.0, prefix='',
                      void_classes=[], valid_classes=[], cache=False, rect=False, rank=-1):
    cfg = get_cfg_defaults()
    cfg.merge_from_file('../dataloaders/configs/sunrgbd.yaml')
    cfg.merge_from_list(['DATASET.ANNOTATION_TYPE', 'bbox',
                         'DATASET.NO_TRANSFORMS', True,
                         'TRAIN.BATCH_SIZE', 1])

    train_loader, val_loader, test_loader, num_class = make_data_loader(cfg)
    dataloader = train_loader
    return dataloader, dataloader.dataset
