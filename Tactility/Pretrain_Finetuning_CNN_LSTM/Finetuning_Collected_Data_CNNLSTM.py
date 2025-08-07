# 0. Import Packages

import os
import torch
import sys
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.models as models
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW
import torch
import torch.nn as nn
import torchvision.models as models
import cv2

# 1. Define critical parameters
contact_number = 2
resnet_depth = 50
lstm_layers = 3
leave_one_out = True
# + also change 6.1: path to data!

# 2. Recover Code from pretraining file
from Pretrain_Finetuning_CNN_LSTM.Pretraining_Online_Data_CNNLSTM import Hardness_CNN_LSTM

# 3. Define Dataset of own collected dataset. To collect data and create table, we refer the user to the paper and to the python file "Collect_Data":
# There, contact is defined based on two criteria: SSIM with a reference image and marker displacement. After contact, 8 images are collected
# at equally spaced depth distances of 0.25mm. We advise to collect about 40 images per object (40 poses), varying a little bit in contact position and yaw (see code)
# To define underneath Pytorch Dataset Class: the custom class is based on examples from pytorch code: https://docs.pytorch.org/tutorials/beginner/data_loading_tutorial.html
# class needs to overwrite two functions: __len__ and __get__item and should inherit from torch.utils.data.Dataset. It mimics setup of online dataset in pretraining files

class Hardness_Dataset(Dataset):
    def __init__(self, dataframe, image_root='Multi_Depth_Data/', transform=None, contact_number = 2):
        self.data = dataframe
        self.image_root = image_root
        self.transform = transform
        if contact_number == 2:
            self.contact_levels = [2,8]
        elif contact_number == 4: 
            self.contact_levels = [2,4,6,8]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        object_name = row['Object']
        hardness = row['Hardness_Level']
        if hardness == int(hardness):
            hardness = int(hardness)
        rank = row['Pose_Number']

        # Load Contact1 image (this will be used for computing the difference image)
        contact1_filename = f"{object_name}_{hardness}_Contact1_{rank}.png"
        contact1_path = os.path.join(self.image_root, contact1_filename)
        contact1_img = Image.open(contact1_path).convert("RGB")
        if self.transform:
            contact1_img = self.transform(contact1_img)

        # Collect difference images
        diff_imgs = []
        for contact in self.contact_levels:
            contact_filename = f"{object_name}_{hardness}_Contact{contact}_{rank}.png"
            contact_path = os.path.join(self.image_root, contact_filename)
            contact_img = Image.open(contact_path).convert("RGB")
            if self.transform:
                contact_img = self.transform(contact_img)

            # Subtract Contact1 from current image
            diff_img = contact_img - contact1_img
            diff_imgs.append(diff_img)

        # Stack to shape [Contact_Number, 3, H, W]
        stacked = torch.stack(diff_imgs, dim=0)

        target = torch.tensor(float(hardness), dtype=torch.float32)
        return stacked, target
    
# 4. Important step in the workflow is the conversion from ShoreA to Shore00. This is because I measured the objects for finetuning with a Shore A durometer
# However as we pretrained on shore00, it is best to also finetune on this scale. For basic materials which is used (blocks, bands, soft wiper), one can use
# the following table which was implemented in code: https://worldwidefoam.com/wp-content/uploads/2022/01/Shore-Hardness-Scales-20220128.pdf?swcfpc=1

def shore00_to_shoreA(value_00, shore00_values, shoreA_values):
    if shore00_values[0] > shore00_values[-1]:
        shore00_values, shoreA_values = shore00_values.flip(0), shoreA_values.flip(0)

    value_00 = value_00.clamp(shore00_values[0], shore00_values[-1])
    idx = torch.searchsorted(shore00_values, value_00).clamp(1, len(shore00_values) - 1)

    x0, x1 = shore00_values[idx - 1], shore00_values[idx]
    y0, y1 = shoreA_values[idx - 1], shoreA_values[idx]
    w = (value_00 - x0) / (x1 - x0)

    return y0 + w * (y1 - y0)

def shoreA_to_shore00(value_A, shoreA_values, shore00_values):
    if shoreA_values[0] > shoreA_values[-1]:
        shoreA_values, shore00_values = shoreA_values.flip(0), shore00_values.flip(0)

    value_A = value_A.clamp(shoreA_values[0], shoreA_values[-1])
    idx = torch.searchsorted(shoreA_values, value_A).clamp(1, len(shoreA_values) - 1)

    x0, x1 = shoreA_values[idx - 1], shoreA_values[idx]
    y0, y1 = shore00_values[idx - 1], shore00_values[idx]
    w = (value_A - x0) / (x1 - x0)

    return y0 + w * (y1 - y0)

# 5. Get dataframe of self collected data, this will be automatically made when collecting data
def get_collected_data(csv_path):
    df = pd.read_csv(csv_path)
    df = df.drop_duplicates(subset=['Object', 'Pose_Number', 'Contact'], keep='last').reset_index(drop=True)
    return df

# 6. Main
if __name__ == "__main__":
    # 6.1. Define path to dataframe and images
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_path = "/home/fv124/lico_share_dir"

    image_root = os.path.join(base_path, "Multi_Depth_Data")
    csv_path = os.path.join(image_root, "Data_Overview.csv")
    df_collected = get_collected_data(csv_path=csv_path)

    # 6.2 Define transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue = 0.01),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])
    
    # 6.3 Define shoreA and shore00 conversion levels
    shoreA_values = torch.tensor([80, 75, 70, 65, 60, 55, 50, 45, 40, 35, 30, 25, 20, 15, 10, 5], 
                                    dtype=torch.float32, device=device)
    shore00_values = torch.tensor([98, 97, 95, 94, 94, 91, 90, 88, 86, 83, 80, 76, 70, 62, 55, 45], 
                                    dtype=torch.float32, device=device)

    
    # 6.4 In case of leave-one-out code underneath will run, to go to immediate final model finetuning, this will be skipped
    if leave_one_out == True:
        unique_objects = df_collected['Object'].unique()
        results = []

        for obj in list(reversed(unique_objects))[:7]:
            print(f"\n===== Leave-One-Out: {obj} =====")

            # Extract the "_Block_XX.X" suffix, see paper about note on damaged sensor
            suffix = '_'.join(obj.split('_')[-2:])  # e.g., "_Block_22.5"

            val_mask = df_collected['Object'] == obj
            train_mask = ~df_collected['Object'].str.endswith(suffix)

            val_df = df_collected[val_mask].copy()
            train_df = df_collected[train_mask].copy()

            # Drop duplicates if necessary
            dedup_cols = ['Object', 'Hardness_Level', 'Pose_Number']
            train_df = train_df.drop_duplicates(subset=dedup_cols).reset_index(drop=True)
            val_df = val_df.drop_duplicates(subset=dedup_cols).reset_index(drop=True)

            train_ds = Hardness_Dataset(train_df, image_root=image_root, transform=val_transform)
            val_ds = Hardness_Dataset(val_df, image_root=image_root, transform=val_transform)
            
            train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=8)

            model = Hardness_CNN_LSTM().to(device)
            model.load_state_dict(torch.load(f"CNN{resnet_depth}_LSTM.pth"))
            optimizer = AdamW([
            {'params': model.cnn_backbone.parameters(), 'lr': 3e-5},
            {'params': model.lstm.parameters(), 'lr': 3e-5},
            {'params': model.fc1.parameters(), 'lr': 5e-5},
            {'params': model.fc2.parameters(), 'lr': 5e-5},
            {'params': model.fc3.parameters(), 'lr': 5e-5},
            {'params': model.out.parameters(), 'lr': 5e-5}
            ], weight_decay=1e-4)
            criterion = nn.MSELoss()
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=2)

            best_val_loss = float('inf')
            patience, patience_counter = 5, 0
            best_preds, best_actuals = [], []

            for epoch in range(15):
                model.train()
                running_loss = 0.0
                for images, labels in train_loader:
                    images, labels = images.to(device), labels.to(device).view(-1)
                    optimizer.zero_grad()
                    outputs = model(images).view(-1)
                    labels = shoreA_to_shore00(labels, shoreA_values, shore00_values) #this is key part, the labels are scaled to shore00 to finetune on same range as pretrained model
                    loss = criterion(outputs, labels)                 
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                train_loss = running_loss / len(train_loader)
                scheduler.step(train_loss)

                model.eval()
                val_loss = 0.0
                fold_preds, fold_actuals = [], []
                with torch.no_grad():
                    for images, labels in val_loader:
                        images = images.to(device)
                        labels = labels.to(device).view(-1)
                        outputs = model(images).view(-1)   
                        labels = shoreA_to_shore00(labels, shoreA_values, shore00_values)
                        loss = criterion(outputs, labels) 
                        val_loss += loss.item()
                        fold_preds.extend(outputs.cpu().tolist())
                        fold_actuals.extend(labels.cpu().tolist())
                val_loss /= len(val_loader)
                best_preds, best_actuals = fold_preds, fold_actuals

                print(f"[{obj}] Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
                sys.stdout.flush()

            # Save predictions
            pred_df = pd.DataFrame({
                'Object': [obj] * len(best_preds),
                'Actual': best_actuals,
                'Predicted': best_preds
            })
            pred_df.to_csv(f"CNN{resnet_depth}_LSTM_Finetuned_{obj.replace(' ', '_')}_results.csv", index=False)

            results.append({
                'Object': obj,
                'Val_Loss': best_val_loss
            })

        # Save LOOO summary
        results_df = pd.DataFrame(results)
        results_df.to_csv(f"CNN{resnet_depth}_LSTM_Finetuned_results.csv", index=False)

    # 6.5 This is final model trained on full dataset, which will be eventually used for deployment in the app
    print("\n===== Final Model Training on Full Dataset =====")

    # Drop duplicates if needed
    df = df_collected
    dedup_cols = ['Object', 'Hardness_Level', 'Pose_Number']
    full_df = df.drop_duplicates(subset=dedup_cols).reset_index(drop=True)

    full_ds = Hardness_Dataset(full_df, image_root=image_root, transform=train_transform)
    full_loader = DataLoader(full_ds, batch_size=8, shuffle=True)

    best_loss = float('inf')
    patience, patience_counter = 5, 0
    model = Hardness_CNN_LSTM().to(device)
    model.load_state_dict(torch.load(f"CNN{resnet_depth}_LSTM.pth"))
        # model.reset_regressor()
    optimizer = AdamW([
        {'params': model.cnn_backbone.parameters(), 'lr': 3e-5},
        {'params': model.lstm.parameters(), 'lr': 3e-5},
        {'params': model.fc1.parameters(), 'lr': 5e-5},
        {'params': model.fc2.parameters(), 'lr': 5e-5},
        {'params': model.fc3.parameters(), 'lr': 5e-5},
        {'params': model.out.parameters(), 'lr': 5e-5}
        ], weight_decay=1e-4)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=1)

    for epoch in range(15):
        model.train()
        running_loss = 0.0
        for images, labels in full_loader:
            images, labels = images.to(device), labels.to(device).view(-1)
            optimizer.zero_grad()
            outputs = model(images).view(-1)
            labels = shoreA_to_shore00(labels, shoreA_values, shore00_values)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch_loss = running_loss / len(full_loader)
        scheduler.step(epoch_loss)

        print(f"[Full Model] Epoch {epoch+1} | Loss: {epoch_loss:.4f}")
        sys.stdout.flush()

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience_counter = 0
            torch.save(model.state_dict(), f"CNN{resnet_depth}_LSTM_Finetuned.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping.")
                break

    





    



