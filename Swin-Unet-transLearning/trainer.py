import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from utils import DiceLoss


def set_training_phase(model, phase="phase1"):

    # 兼容 DataParallel
    curr_model = model.module if hasattr(model, 'module') else model

    # 先冻结所有参数
    for param in curr_model.parameters():
        param.requires_grad = False

    if phase == "phase1":
        # Phase 1: 仅激活深层和解码器
        logging.info(">>> PHF")
        # 解冻 Encoder 的后两层 (Stage 3 & 4)
        for param in curr_model.swin_unet.layers[2:].parameters():
            param.requires_grad = True
        for param in curr_model.swin_unet.layers_up.parameters():
            param.requires_grad = True
        if hasattr(curr_model.swin_unet, 'output'):
            for param in curr_model.swin_unet.output.parameters():
                param.requires_grad = True

    elif phase == "phase2":
        logging.info(">>> Fine-tuning")
        for param in curr_model.parameters():
            param.requires_grad = True


def trainer_synapse(args, model, snapshot_path):
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    max_epochs = args.max_epochs

    # 1. 数据准备 (针对 256 尺寸)
    db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    db_val = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="val",
                             transform=transforms.Compose(
                                 [RandomGenerator(output_size=[args.img_size, args.img_size])]))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    train_loader = DataLoader(db_train, batch_size=batch_size, shuffle=True,
                              num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn)
    val_loader = DataLoader(db_val, batch_size=batch_size, shuffle=False,
                            num_workers=8, pin_memory=True)

    # 2. 损失函数
    ce_loss = nn.CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)

    # 3. 初始化训练阶段：Phase 1
    current_phase = "phase1"
    set_training_phase(model, phase=current_phase)

    # 仅优化当前处于 Active 状态的参数
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                          lr=base_lr, momentum=0.9, weight_decay=0.0001)

    writer = SummaryWriter(snapshot_path + '/log')
    best_performance = 0.0

    # 4. 训练循环
    for epoch_num in range(max_epochs):
        model.train()
        epoch_l_sem = 0

        for i_batch, sampled_batch in tqdm(enumerate(train_loader), desc=f"Epoch {epoch_num}", total=len(train_loader)):
            image_batch, label_batch = sampled_batch['image'].cuda(), sampled_batch['label'].cuda()
            outputs, hidden_states = model(image_batch)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            l_sem = 0.5 * loss_ce + 0.5 * loss_dice
            l_perc = 0
            if len(hidden_states) >= 4:
                # 感知损失路径
                l_perc_mid = torch.mean(torch.abs(hidden_states[1] - hidden_states[2]))
                l_perc_deep = torch.mean(torch.abs(hidden_states[2] - hidden_states[3]))
                l_perc = 0.1 * (l_perc_mid + l_perc_deep)

            total_loss = l_sem + l_perc

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_l_sem += l_sem.item()

        avg_l_sem = epoch_l_sem / len(train_loader)
        logging.info(f'Epoch {epoch_num}: avg_l_sem: {avg_l_sem:.4f}, phase: {current_phase}')

        # --- Strategy Decision Block
        if current_phase == "phase1" and avg_l_sem < 0.4:
            current_phase = "phase2"
            set_training_phase(model, phase=current_phase)

            # 切换阶段后更新优化器并大幅降低学习率 (LR=0.1x)
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=base_lr * 0.1, momentum=0.9, weight_decay=0.0001)
            logging.info(f"PHF Decision: Switching to Phase 2 at Epoch {epoch_num}")

        if epoch_num % 1 == 0:
            model.eval()
            save_mode_path = os.path.join(snapshot_path, 'last_model.pth')
            torch.save(model.state_dict(), save_mode_path)

    writer.close()
    return "Training Finished"