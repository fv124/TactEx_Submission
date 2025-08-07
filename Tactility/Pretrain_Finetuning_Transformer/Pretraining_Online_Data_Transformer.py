# 0. Import the right packages
import os
import torch
import sys
import cv2
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
import torchvision.models as models
from PIL import Image
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW
import torch
import torch.nn as nn
import timm
import torchvision.models as models
from skimage.metrics import structural_similarity as ssim

# 1.Change crucial parameters to run right model
create_table = False
contact_number = 2
lstm_layers = 3
train_range = 'Full'
# + also change 6.1: path to data!

# 2. Recover code from CNN_LSTM version
from Pretrain_Finetuning_CNN_LSTM.Pretraining_Online_Data_CNNLSTM import variance_penalty, compute_rmse #for computing loss function
from Pretrain_Finetuning_CNN_LSTM.Pretraining_Online_Data_CNNLSTM import Online_Data, calculate_color_ssim, get_frames, get_online_data #for computing online dataframe

# 3. Define Transformer Hardness Estimator
class Hardness_Transformer(nn.Module):
    def __init__(self, pretrained = True):
        super(Hardness_Transformer, self).__init__()

        # 3.1 Use CNN backbone to extract features, this one works fine with transformer
        self.cnn = timm.create_model("v", pretrained=pretrained, features_only=True)
        self.cnn_out_dim = self.cnn.feature_info[-1]['num_chs']

        # 3.2 Transform CNN features to transformer input dimension
        self.project = nn.Linear(self.cnn_out_dim, 256)

        # 3.3 Transformer Encoder for (improved?) temporal modeling
        encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=4, dim_feedforward=512, dropout=0.1)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # 3.4 Regression head
        self.regressor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        B, T, C, H, W = x.size() #batch, time frames, channels, heigh, width
        x = x.view(B * T, C, H, W)

        feats = self.cnn(x)[-1]           # Shape: [B*T, C', H', W']
        feats = feats.mean(dim=[2, 3])    # Global Average Pooling → [B*T, C']
        feats = self.project(feats)       # [B*T, 256]
        feats = feats.view(B, T, -1)      # [B, T, 256]
        feats = feats.permute(1, 0, 2)    # [T, B, 256] for Transformer (this model wants temporal sequence as first dimension)

        temporal_feat = self.transformer(feats)  # [T, B, 256]
        pooled = temporal_feat.mean(dim=0)       # [B, 256]

        out = self.regressor(pooled)  # [B, 1]
        return out.squeeze(1)         # [B]

# 6. Main
if __name__ == "__main__":
    # 6.1 define path of online_data
    base_path = "/home/fv124/lico_share_dir"
    path2 = os.path.join(base_path, "Online_Data")

    # 6.2 Load (previously saved) data
    if create_table == True:
        df_online = get_online_data(path2)
        df_online.to_csv("df_online_higher_thrshold.csv", index=False)
    else:
        df_online = pd.read_csv("df_online_higher_thrshold.csv")
    df_online = df_online[df_online['Shape'] != 'CHOCO']
    df_online = df_online[df_online['Shape'] != 'BASIC']
    df_online = df_online.drop_duplicates()
    df_online = df_online.reset_index(drop=True)
    df_online = df_online[df_online['stamps'] != '']
    df_online = df_online[df_online['stamps'].notna()].reset_index(drop=True)

    if train_range == 'Full':
        pass
    elif train_range == 'Half':
        df_online = df_online[df_online['Hardness_Shore00'] >= 40].reset_index(drop=True)

    # 6.3 Transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        #transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue = 0.02),
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
    
    # 6.4 Define device, model, optimizers, criterion and scheduler
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Hardness_Transformer().to(device)
    optimizer = AdamW([
                {'params': model.cnn.parameters(), 'lr': 3e-5},
                {'params': model.transformer.parameters(), 'lr': 3e-5},
                {'params': model.regressor.parameters(), 'lr': 5e-5}
                ], weight_decay=1e-4)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=2)
    results = []

    # 6.5 Split data, define dataframes and dataloaders
    train_df, val_df = train_test_split(df_online, test_size=0.2, random_state=1)
    train_ds = Online_Data(train_df, video_root=path2, transform=train_transform)
    val_ds = Online_Data(val_df, video_root=path2, transform=val_transform)
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)

    # 6.6 Start Training
    print('Training Started')
    sys.stdout.flush()
    for epoch in range(80):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device).view(-1)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels) + 4*variance_penalty(outputs) #balance term between MSE loss and model collapse penalty
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
                images, labels = images.to(device), labels.to(device).view(-1)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                fold_preds.extend(outputs.cpu().tolist())
                fold_actuals.extend(labels.cpu().tolist())

        val_loss /= len(val_loader)
        scheduler.step(train_loss)
        best_preds, best_actuals = fold_preds, fold_actuals
        train_rmse = compute_rmse(fold_preds, fold_actuals)  # Or compute on training preds if available
        val_rmse = compute_rmse(fold_preds, fold_actuals)

        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val RMSE: {val_rmse:.4f}")
        sys.stdout.flush()

    # 6.7 Save Prediciton
    pred_df = pd.DataFrame({
            'Actual': best_actuals,
            'Predicted': best_preds
        })
    pred_df.to_csv(f"Predictions_Transformer.csv", index=False)
    torch.save(model.state_dict(), f"Transformer.pth")


