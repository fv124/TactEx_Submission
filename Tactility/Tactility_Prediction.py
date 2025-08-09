from Tactility.Pretrain_Finetuning_CNN_LSTM.Pretraining_Online_Data_CNNLSTM import Hardness_CNN_LSTM
from Tactility.Pretrain_Finetuning_Transformer.Pretraining_Online_Data_Transformer import Hardness_Transformer
from Tactility.Pretrain_Finetuning_CNN_LSTM.Finetuning_Collected_Data_CNNLSTM import Hardness_Dataset
import torch
import pandas as pd
from torchvision import transforms
from torch.utils.data import DataLoader

def tactility_prediction(model_name='CNN_LSTM'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_name == 'CNN_LSTM':
        model = Hardness_CNN_LSTM(resnet_depth=50).to(device)
        model.load_state_dict(torch.load('Tactility/Pretrain_Finetuning_CNN_LSTM/CNN50_LSTM_Full_Finetuned.pth', map_location=torch.device('cpu')))
    elif model_name == 'Transformer':
        model = Hardness_Transformer().to(device)
        model.load_state_dict(torch.load('Tactility/Transformer_Half_Finetuned.pth', map_location=torch.device('cpu')))
    model.eval()

    val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    ])

    preds = []
    targets = []
    df = pd.read_csv('Tactility/Data/Data_Overview.csv')
    df = df.drop_duplicates(subset=['Object', 'Pose_Number', 'Hardness_Level'], keep='last').reset_index(drop=True)
    df.head()

    val_ds = Hardness_Dataset(df, transform=val_transform, image_root='Tactility/Data/')
    val_loader = DataLoader(val_ds, batch_size=8)

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images).squeeze()
            preds.extend(outputs.cpu().numpy())
            targets.extend(labels.cpu().numpy())

    return preds




