import os
import yaml
from yacs.config import CfgNode as CN

_C = CN()

# Base config files
_C.BASE = ['']

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
_C.DATA.BATCH_SIZE = 128
_C.DATA.DATA_PATH = ''
_C.DATA.DATASET = 'sentinel2_apple' # 修改为项目名
_C.DATA.IMG_SIZE = 256              # 设为 256
_C.DATA.INTERPOLATION = 'bicubic'
_C.DATA.ZIP_MODE = False
_C.DATA.CACHE_MODE = 'part'
_C.DATA.PIN_MEMORY = True
_C.DATA.NUM_WORKERS = 8

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.TYPE = 'swin'
_C.MODEL.NAME = 'swin_tiny_patch4_window8_256'
_C.MODEL.PRETRAIN_CKPT = './pretrained_ckpt/best_model_1.pth'
_C.MODEL.RESUME = ''
_C.MODEL.NUM_CLASSES = 2 # 苹果园 vs 背景
_C.MODEL.DROP_RATE = 0.0
_C.MODEL.DROP_PATH_RATE = 0.1
_C.MODEL.LABEL_SMOOTHING = 0.1

# Swin Transformer parameters
_C.MODEL.SWIN = CN()
_C.MODEL.SWIN.PATCH_SIZE = 4
_C.MODEL.SWIN.IN_CHANS = 3      # 哨兵2 RGB 三波段
_C.MODEL.SWIN.EMBED_DIM = 96
_C.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
_C.MODEL.SWIN.DECODER_DEPTHS = [2, 2, 6, 2]
_C.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
_C.MODEL.SWIN.WINDOW_SIZE = 8   # 修改为 8，以适配 256 尺寸 (256/4/8=8)
_C.MODEL.SWIN.MLP_RATIO = 4.
_C.MODEL.SWIN.QKV_BIAS = True
_C.MODEL.SWIN.QK_SCALE = None
_C.MODEL.SWIN.APE = False
_C.MODEL.SWIN.PATCH_NORM = True
_C.MODEL.SWIN.FINAL_UPSAMPLE= "expand_first"

# -----------------------------------------------------------------------------
# PHF Strategy Settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.PHF = CN()
_C.TRAIN.PHF.ENABLE = True          # 是否开启分层微调
_C.TRAIN.PHF.INITIAL_PHASE = 'phase1'
_C.TRAIN.PHF.SWITCH_THRESHOLD = 0.4 # L_sem 低于此值时切换到 Phase 2
_C.TRAIN.PHF.PERC_WEIGHT = 0.1      # 感知损失 L_perc 的权重

# 基础训练设置
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.EPOCHS = 150
_C.TRAIN.WARMUP_EPOCHS = 10
_C.TRAIN.WEIGHT_DECAY = 0.05
_C.TRAIN.BASE_LR = 1e-4
_C.TRAIN.WARMUP_LR = 5e-7
_C.TRAIN.MIN_LR = 5e-6
_C.TRAIN.CLIP_GRAD = 5.0
_C.TRAIN.AUTO_RESUME = True
_C.TRAIN.ACCUMULATION_STEPS = 0
_C.TRAIN.USE_CHECKPOINT = False

_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.NAME = 'cosine'
_C.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 30
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1

_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = 'adamw'
_C.TRAIN.OPTIMIZER.EPS = 1e-8
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9

# -----------------------------------------------------------------------------
# 其他配置保持默认
# -----------------------------------------------------------------------------
_C.AUG = CN()
_C.AUG.COLOR_JITTER = 0.4
_C.AUG.AUTO_AUGMENT = 'rand-m9-mstd0.5-inc1'
_C.AUG.REPROB = 0.25
_C.AUG.REMODE = 'pixel'
_C.AUG.RECOUNT = 1
_C.AUG.MIXUP = 0.0 # 遥感影像分割建议先设为 0
_C.AUG.CUTMIX = 0.0
_C.AUG.CUTMIX_MINMAX = None
_C.AUG.MIXUP_PROB = 1.0
_C.AUG.MIXUP_SWITCH_PROB = 0.5
_C.AUG.MIXUP_MODE = 'batch'

_C.TEST = CN()
_C.TEST.CROP = True
_C.AMP_OPT_LEVEL = ''
_C.OUTPUT = './results'
_C.TAG = 'PHF_SwinUnet_Apple'
_C.SAVE_FREQ = 1
_C.PRINT_FREQ = 10
_C.SEED = 1234
_C.EVAL_MODE = False
_C.THROUGHPUT_MODE = False
_C.LOCAL_RANK = 0

def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)
    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(config, os.path.join(os.path.dirname(cfg_file), cfg))
    config.merge_from_file(cfg_file)
    config.freeze()

def update_config(config, args):
    _update_config_from_file(config, args.cfg)
    config.defrost()
    if args.opts: config.merge_from_list(args.opts)
    if args.batch_size: config.DATA.BATCH_SIZE = args.batch_size
    if args.resume: config.MODEL.RESUME = args.resume
    if args.tag: config.TAG = args.tag
    config.freeze()

def get_config(args):
    config = _C.clone()
    update_config(config, args)
    return config