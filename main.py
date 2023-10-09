import os
import random
import pandas as pd
import numpy as np
import time
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split

from Dataset import ImageMaskDataset, ImageOnlyDataset, rle_encode
from Model import UNet, DANN_UNet
from train_utils import calculate_iou, EarlyStopping

# GPU, TPU, 또는 CPU를 사용할 수 있도록 디바이스 설정
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps:0')
else:
    try:
        import torch_xla.core.xla_model as xm

        device = xm.xla_device()  # TPU (PyTorch XLA)
        print('TPU:', str(device))
    except ImportError:
        device = torch.device('cpu')

seed = 42
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)

if device == 'cuda':
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
elif device == 'mps:0':
    torch.mps.manual_seed(seed)

if __name__ == '__main__':
    print(device)

    # 이미지 전처리 및 데이터셋 생성을 위한 설정
    transform = A.Compose(
        [
            A.Resize(256, 256),
            A.Normalize(),
            ToTensorV2()
        ]
    )

    batch_size = 8
    learning_rate = 0.001
    train_csv = './data/train_source.csv'
    val_csv = './data/val_source.csv'
    target_csv = './data/train_target.csv'

    train_dataset = ImageMaskDataset(csv_file=train_csv, transform=transform)
    val_dataset = ImageMaskDataset(csv_file=val_csv, transform=transform)

    combined_dataset = ConcatDataset([train_dataset, val_dataset])
    target_dataset = ImageOnlyDataset(csv_file=target_csv, transform=transform)

    train_dataset, val_dataset = train_test_split(combined_dataset, test_size=0.2, random_state=42)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    target_dataloader = DataLoader(target_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # UNet 모델 관련 부분 제거
    dann_unet = DANN_UNet(num_classes=13).to(device)

    segmentation_criterion = nn.CrossEntropyLoss()
    segmentation_optimizer = optim.Adam(dann_unet.parameters(), lr=learning_rate)  # Optimizer를 dann_unet에 적용

    domain_criterion = nn.CrossEntropyLoss()
    domain_optimizer = optim.Adam(dann_unet.parameters(), lr=learning_rate)  # Optimizer를 dann_unet에 적용

    source_labels = torch.zeros(batch_size, dtype=torch.long).to(device)
    target_labels = torch.ones(batch_size, dtype=torch.long).to(device)

    patience = 5
    best_loss = None
    early_stop = False
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    segmentation_scheduler = ReduceLROnPlateau(segmentation_optimizer, mode='min', factor=0.1, patience=3)
    domain_scheduler = ReduceLROnPlateau(domain_optimizer, mode='min', factor=0.1, patience=3)

    for epoch in range(200):
        print(f'Epoch {epoch + 1}')
        time.sleep(0.1)
        dann_unet.train()
        loss_segmentation = 0
        loss_domain_source = 0
        loss_domain_target = 0

        target_iter = iter(target_dataloader)

        for images, masks in tqdm(train_dataloader):
            images = images.float().to(device)
            masks = masks.long().to(device)

            segmentation_optimizer.zero_grad()
            segmentation_outputs, _ = dann_unet(images)  # dann_unet으로 변경
            segmentation_loss = segmentation_criterion(segmentation_outputs, masks.squeeze(1))
            loss_segmentation += segmentation_loss.item()

            _, dann_source_outputs = dann_unet(images, alpha=1.0)

            domain_optimizer.zero_grad()
            source_labels.fill_(0)
            dann_source_loss = domain_criterion(dann_source_outputs, source_labels)
            loss_domain_source += dann_source_loss.item()

            try:
                target_images = next(target_iter)
            except StopIteration:
                target_iter = iter(target_dataloader)
                target_images = next(target_iter)

            target_images = target_images.float().to(device)
            current_batch_size = len(target_images)

            adjusted_target_labels = torch.ones(current_batch_size, dtype=torch.long).to(device)

            domain_optimizer.zero_grad()
            _, dann_target_outputs = dann_unet(target_images, alpha=-1.0)
            dann_target_loss = domain_criterion(dann_target_outputs, adjusted_target_labels)
            loss_domain_target += dann_target_loss.item()

        average_loss = (loss_segmentation + loss_domain_source + loss_domain_target) / len(train_dataloader)
        print(f'Average Loss: {average_loss}')
        time.sleep(0.1)

        if val_dataloader is not None:
            dann_unet.eval()
            val_loss_total = 0
            mean_iou_overall = 0
            class_iou = np.zeros(13)
            class_pixel_count = np.zeros(13)

            with torch.no_grad():
                for images, masks in tqdm(val_dataloader):
                    images = images.float().to(device)
                    masks = masks.long().to(device)

                    outputs, _ = dann_unet(images)  # dann_unet으로 변경
                    _, predicted = torch.max(outputs, dim=1)

                    val_loss_batch = segmentation_criterion(outputs, masks.squeeze(1))
                    val_loss_total += val_loss_batch.item()

                    iou_batch = []

                    for class_id in range(13):
                        iou_class = calculate_iou(masks.cpu().numpy(), predicted.cpu().numpy(), class_id)
                        iou_batch.append(iou_class)

                    mean_ious = np.mean(np.array(iou_batch))
                    mean_iou_overall += mean_ious

                mean_iou_overall /= len(val_dataloader)

                print(f'Validation Mean IoU: {mean_iou_overall}')

                val_avg_loss = val_loss_total / len(val_dataloader)

                print(f'Validation Loss: {val_avg_loss}')

                segmentation_scheduler.step(val_avg_loss)
                domain_scheduler.step(val_avg_loss)

                early_stopping(val_avg_loss, dann_unet)

                if  early_stopping.early_stop:
                    print("Early stopping")
                    break

    dann_unet.load_state_dict(torch.load('checkpoint.pth'))

    test_dataset = ImageMaskDataset(csv_file='./data/test.csv', transform=transform, infer=True)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)

    with torch.no_grad():
        dann_unet.eval()
        result = []
        for images in tqdm(test_dataloader):
            images = images.float().to(device)
            outputs, _ = dann_unet(images)
            outputs = torch.softmax(outputs, dim=1).cpu()
            outputs = torch.argmax(outputs, dim=1).numpy()

            for pred in outputs:
                pred = pred.astype(np.uint8)
                pred = Image.fromarray(pred[0])
                pred = pred.resize((960, 540), Image.NEAREST)
                pred = np.array(pred)

                for class_id in range(12):
                    class_mask = (pred == class_id).astype(np.uint8)
                    if np.sum(class_mask) > 0:
                        mask_rle = rle_encode(class_mask)
                        result.append(mask_rle)
                    else:
                        result.append(-1)

    submit = pd.read_csv('./data/sample_submission.csv')
    submit['mask_rle'] = result

    submit.to_csv('./data/dann_unet_submit.csv', index=False)