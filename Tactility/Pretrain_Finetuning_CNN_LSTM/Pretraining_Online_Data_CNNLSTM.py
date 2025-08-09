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
import torchvision.models as models
from skimage.metrics import structural_similarity as ssim

# 1. Change crucial parameters to run right model
create_table = False
contact_number = 2
resnet_depth = 50
lstm_layers = 3
train_range = 'Full'


# 2. Define function which will be called to create table out of directory with online videos of gelsight data, these can be downloaded via following link: 
# https://people.csail.mit.edu/yuan_wz/hardness-estimation.htm#:~:text=GelSight%20measures%20the%20gradient%20of,R%20squared%20higher%20than%200.98.

# 2.1 Here, we provide the path where the videos are saved (on own computer or on HPC) and a dataframe will be created consisting of the id number, hardness value, shape and timestamp of contact
# To identify the contact, it is dependent on two functions defined underneath (2.2) 
# !!! This function takes long to run, therefore we recommend to run it once and save it after this. This can be done be setting create_table parameter to False after first run!!!
def get_online_data(path):
    filenames = [file for file in os.listdir(path) if file.endswith(".avi")] #videos are define in .avi format
    df = pd.DataFrame(columns = ['Shape', 'Number', "Hardness_Shore00"])

    for file in filenames:
        shape, hardness, number = file.split('_')
        hardness = int(hardness)
        df.loc[len(df)] = [shape, number, hardness]

    df = get_frames(df, path) 
    return df

# 2.2. We will use SSIM, just as we used it for collecting data, but now to get the initial contact frame from video in function get_frames
def calculate_color_ssim(img1, img2):
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    if img1.shape != img2.shape:
        raise ValueError("Input images must have the same dimensions")
    
    score, diff = ssim(img1, img2, multichannel=True, full=True, win_size=3)
    return score, diff

def get_frames(df, path2):
    df['stamps'] = "" #we will add extra column to each row defining the stamp of inital contact
    for idx in range(len(df)): 
        row = df.iloc[idx]
        number = row['Number']
        hardness = row['Hardness_Shore00']
        if hardness == int(hardness):
                hardness = f"{int(hardness):02d}"
        shape = row['Shape']

        video_name = f"{shape}_{hardness}_{number}"
        video_path = os.path.join(path2, video_name)
        cap = cv2.VideoCapture(video_path)
        ret, frame1 = cap.read()

        if idx%500==0:
            print(f'{idx} passed')
            sys.stdout.flush()
        i = 0
        while i <= 25:
            ret, frame2 = cap.read()
            if not ret:
                break
            if calculate_color_ssim(frame1, frame2)[0] <= 0.90: #we use 0.90 as SSIM score to define contact, based on trial and error we are confident this works best
                df.loc[idx, 'stamps'] = i
                i = 30 #will stop while loop
            i += 1
        cap.release()
    return df 

# 3. Define Pytorch Dataset Class: this custom class is based on examples from pytorch code: https://docs.pytorch.org/tutorials/beginner/data_loading_tutorial.html
# class needs to overwrite two functions: __len__ and __get__item and should inherit from torch.utils.data.Dataset
class Online_Data(Dataset):
    def __init__(self, dataframe, video_root='Online_Data', transform=None, contact_number = contact_number):
        self.data = dataframe
        self.video_root = video_root
        self.transform = transform

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        row = self.data.iloc[index]
        number = row['Number']
        hardness = row['Hardness_Shore00']
        if hardness == int(hardness):
            hardness = f"{int(hardness):02d}"
        shape = row['Shape']
        stamp = row['stamps']
        
        video_name = f"{shape}_{hardness}_{number}"
        video_path = os.path.join(self.video_root, video_name)

        # Read specific frames
        frames = self.extract_frames(video_path, stamp)

        if self.transform:
            frames = [self.transform(Image.fromarray(f).convert("RGB")) for f in frames]

        ref_frame = frames[0]
        diff_imgs = [frame - ref_frame for frame in frames[1:]] #we will train on difference of contact images, not on pure contact images
        stacked = torch.stack(diff_imgs, dim=0)
        target = torch.tensor(int(hardness), dtype=torch.float32)

        return stacked, target
    
    def extract_frames(self, video_path, stamp):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []

        start_frame = stamp
        if contact_number == 2:
            all_frames = [start_frame, stamp+1, stamp+7]
            end_frame = stamp+7
        elif contact_number == 5:
            all_frames = [start_frame, stamp+1, stamp+3, stamp+5, stamp+7]
            end_frame = stamp+7

        i = 0
        while i < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if i in all_frames:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            elif i >= end_frame:
                break
            i += 1
        cap.release()

        return frames
    
# 4. Define LSTM Model 
class Hardness_CNN_LSTM(nn.Module):
    def __init__(self, lstm_hidden_size=512, lstm_layers=lstm_layers, resnet_depth = resnet_depth):
        super().__init__()

        # Define pretrained resnet model as start for backbone
        if resnet_depth == 50:
            resnet = models.resnet50(pretrained=True)
            self.lstm_input_size = 2048

        elif resnet_depth == 34:
            resnet = models.resnet34(pretrained=True)
            self.lstm_input_size = 512

        elif resnet_depth == 101:
            resnet = models.resnet101(pretrained=True)
            self.lstm_input_size = 2048

        # Remove last FC layer
        modules = list(resnet.children())[:-1]
        self.cnn_backbone = nn.Sequential(*modules)

        # Optionally unfreeze for finetuning
        for param in self.cnn_backbone.parameters():
            param.requires_grad = True  # set to False to freeze

        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=0.2
        )

        self.feature_dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(lstm_hidden_size, 256)
        self.bn1 = nn.LayerNorm(256) #I remarked that layernorm works better than batchnorm
        self.dropout1 = nn.Dropout(0.2)

        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.LayerNorm(128)
        self.dropout2 = nn.Dropout(0.1)

        self.fc3 = nn.Linear(128, 32)
        self.bn3 = nn.LayerNorm(32)

        self.out = nn.Linear(32, 1) 
        nn.init.kaiming_uniform_(self.out.weight, a=0.01)
        nn.init.zeros_(self.out.bias)

        self._init_weights()
        
    # Beginning of Reference: OpenAI ChatGPT (on 2025/07/22)
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Use He initialization, more suitable for ReLU activation
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.1)  # Small positive bias
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.kaiming_normal_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        nn.init.constant_(param.data, 0)
                        # Set forget gate bias to positive values
                        param.data[m.hidden_size:2*m.hidden_size] = 1.0
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    # End of Reference: OpenAI ChatGPT (on 2025/07/22)

    def forward(self, x):
        # x shape: [B, 2, 3, 224, 224]
        B, T, C, H, W = x.size()
        x = x.view(B * T, C, H, W)  # [B*2, 3, 224, 224]

        x = self.cnn_backbone(x)   # [B*2, 2048, 1, 1]
        x = x.view(B, T, -1)      
        x = self.feature_dropout(x)

        lstm_out, _ = self.lstm(x)   # [B, 2, 512]
        avg_output = lstm_out[:, -1, :] #[B, 512]

        x = F.relu(self.bn1(self.fc1(avg_output))) #[B, 256]
        x = self.dropout1(x)

        x = F.relu(self.bn2(self.fc2(x))) #[B, 128]
        x = self.dropout2(x)

        x = F.relu(self.bn3(self.fc3(x))) #[B, 32]

        x = self.out(x) #[B, 1]
        return x.squeeze(1)

# 5. Helper Functions
def variance_penalty(outputs):
    return torch.clamp(1.0 / (outputs.var(unbiased=False) + 1e-6), max=1000.0) # penalize low variance

def compute_rmse(preds, targets):
        preds = torch.tensor(preds)
        targets = torch.tensor(targets)
        mse = torch.mean((preds - targets) ** 2)
        rmse = torch.sqrt(mse)
        return rmse.item()


# 6. Main
if __name__ == "__main__":
    # 6.1 define path of online_data
    base_path = "/home/fv124/lico_share_dir"
    path2 = os.path.join(base_path, "Online_Data")

    # 6.2 Load (previously saved) data and manipulate
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
    model = Hardness_CNN_LSTM().to(device)
    optimizer = AdamW([
    {'params': model.cnn_backbone.parameters(), 'lr': 5e-5},
    {'params': model.lstm.parameters(), 'lr': 5e-5},
    {'params': model.fc1.parameters(), 'lr': 5e-5},
    {'params': model.fc2.parameters(), 'lr': 5e-5},
    {'params': model.fc3.parameters(), 'lr': 5e-5},
    {'params': model.out.parameters(), 'lr': 1e-3}  # higher lr for last layer
    ])
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
    pred_df.to_csv(f"Predictions_CNN{resnet_depth}_LSTM.csv", index=False)
    torch.save(model.state_dict(), f"Predictions_CNN{resnet_depth}_LSTM.csv.pth")
