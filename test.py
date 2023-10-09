from PIL import Image
import pandas as pd
import numpy as np

from Dataset import CustomDataset, rle_encode
from Model import UNet
from train_utils import calculate_iou, EarlyStopping

import torch
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split

# Use GPU or TPU in PyTorch
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps:0')
else:
    try:
        import torch_xla
        import torch_xla.core.xla_model as xm

        device = xm.xla_device()
    except ImportError:
        device = torch.device('cpu')

if __name__ == '__main__':
    transform = A.Compose(
        [
            A.Resize(256, 256),
            A.Normalize(),
            ToTensorV2()
        ]
    )

    # 데이터셋 및 DataLoader 생성
    train_csv = './data/train_source.csv'
    val_csv = './data/val_source.csv'

    train_dataset = CustomDataset(csv_file=train_csv, transform=transform)
    val_dataset = CustomDataset(csv_file=val_csv, transform=transform)

    combined_dataset = ConcatDataset([train_dataset, val_dataset])

    train_dataset, val_dataset = train_test_split(combined_dataset, test_size=0.2, random_state=42)

    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)

    # 모델 초기화
    model = UNet().to(device)

    # 손실 함수와 옵티마이저 정의
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Initialize the EarlyStopping object
    early_stopping = EarlyStopping(patience=5, verbose=True)

    # Training loop
    for epoch in range(200):
        model.train()
        epoch_loss = 0
        for images, masks in tqdm(train_dataloader):
            images = images.float().to(device)
            masks = masks.long().to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks.squeeze(1))
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f'Epoch {epoch + 1}, Loss: {epoch_loss / len(train_dataloader)}')

        # Validation loop and mIoU calculation
        model.eval()
        val_loss = 0
        class_iou = np.zeros(13)  # One IoU value per class
        class_pixel_count = np.zeros(13)  # Total pixels per class
        with torch.no_grad():
            for images, masks in tqdm(val_dataloader):
                images = images.float().to(device)
                masks = masks.long().to(device)

                outputs = model(images)
                _, predicted = torch.max(outputs, 1)

                val_loss += criterion(outputs, masks.squeeze(1)).item()

                # Calculate IoU for each class
                for class_id in range(13):
                    class_iou[class_id] += calculate_iou(masks.cpu().numpy(), predicted.cpu().numpy(), class_id)
                    class_pixel_count[class_id] += np.sum(
                        (masks.cpu().numpy() == class_id) | (predicted.cpu().numpy() == class_id))

        # Calculate mIoU for each class and overall mIoU
        mean_iou = 0
        for class_id in range(13):
            if class_pixel_count[class_id] > 0:
                class_iou[class_id] /= class_pixel_count[class_id]
            mean_iou += class_iou[class_id]

        mean_iou /= 13  # Calculate mean IoU

        print(f'Validation Mean IoU: {mean_iou}')
        print(f'Validation Loss: {val_loss / len(val_dataloader)}')

        # Check if early stopping criteria are met
        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    # Load the best model checkpoint
    model.load_state_dict(torch.load('checkpoint.pth'))

    # 테스트 데이터에 대한 예측 및 결과 저장
    test_dataset = CustomDataset(csv_file='./data/test.csv', transform=transform, infer=True)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)

    with torch.no_grad():
        model.eval()
        result = []
        for images in tqdm(test_dataloader):
            images = images.float().to(device)
            outputs = model(images)
            outputs = torch.softmax(outputs, dim=1).cpu()
            outputs = torch.argmax(outputs, dim=1).numpy()

            for pred in outputs:
                pred = pred.astype(np.uint8)
                pred = Image.fromarray(pred)
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

    submit.to_csv('./data/baseline_submit.csv', index=False)
