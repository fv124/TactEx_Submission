# TactEx_Project

<p align="center">
  <img src="https://github.com/user-attachments/assets/1f9b502d-818f-479b-9c1e-5a6c3b2c3a1c" width="600" height="350" alt="Overview Image">
</p>
This page contains the code to recreate the results reported from TactEx: A Multimodal Robotic Pipeline for Human-Like Touch and Hardness Estimation. The report can be consulted on this GitHub page as well (Main/Report_TactEx_Felix_Verstraete). This work was conducted at Imperial College London by members of the Embodied Intelligence Lab.

## Contents

- [Demo](#demo)
- [Calibration](#calibration)
- [Launching](#launching)
- [Tactility](#Tactility)
- [Servoing](#Servoing)
- [Language](#Language)

## Demo
https://github.com/user-attachments/assets/724e007f-3ee0-4272-ac70-260be6d979df

## Calibration
Before Launching the app, the user should calibrate the camera to the robot space and make sure the robot (this project made use of XArm UFactory 850) is connected. This takes about 5 minutes and is explained the [notebook](./Calibration/Calibrate.ipynb).

## Launching
For Launching the app, please make sure you have first calibrated the camera and inserted the API keys in the language models. You can set the visual servoing to "YOLO" or "GSAM" in the [app file](./Report_App.py). Install the requirement by running following command.

```python
pip install -r requirements.txt
```

Then launch the app by following command. A streamlit browser will open.

```python
streamlit run Report_App.py
```

## Tactility
Several models were used in this project for predicting the hardness. To investigate the pretraining and finetune models of VGG16-LSTM3 or ResNet50-LSTM3, please follow next [link](./Tactility/Pretrain_Finetuning_CNN_LSTM). To investigate the transformer model, please follow next [link](./Tactility/Pretrain_Finetuning_Transformer). On top of the files one can change the critical parameters, used in the ablation study results.

The code for tactility prediction or tactile sensing is available by running following commands:
```python
python Tactility/Tactility_Prediction.py
```
```python
python Tactility/Tactility_Sensing.py
```

The notebook for collecting the small custom dataset for finetuning is found in following [folder](./Tactility/Collect_Data/Collect_Data.ipynb).

## Servoing 
The visual servoing models can be found in following [folder](./Vision). You can either choose to run the YOLO model or the GSAM model. Their advantages and disadvantages are reported in the paper. Exampoes masks and scene images (both annotatied and raw) are also visible in the folder.

## Language
The NLP logic for the visual servoing can be found in this [file](./Language/NLP.py). Finally, the LLM and LLM as a judge are alos located in the folder Language. Please generate an API key via Groq to enable connection to the different language models. To run them eventually, use following commands:

```python
python Language/Tactile_LLM.py.py
```
```python
python Language/test_LLM_Evaluator.py
```

## References:
Full reference is given to all the papers, authors and websites mentioned on the last page of the [report](./Report_TactEx_Felix_Verstraete.pdf). Reference for code is given inline.
