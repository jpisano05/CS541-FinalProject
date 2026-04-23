import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, Dataset, random_split
from torchvision import datasets, models
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
from torch.optim import lr_scheduler
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchaudio
from tempfile import TemporaryDirectory
import os
import time
import cv2
from pathlib import Path
import librosa

#data path
dataPath = '/data'

#decides 1 in sampleRate frames are utilized
#must be sampled as the full frameage is way to big to process
sampleRate = 5
#resolution to resize video frames to
resizeResolution = 112

#hyperparameters
batchSize = 32
specLearningRate = 0.1
specMomentum = 0.9
specWeightDecay = 5e-4
specStepSize = 30
specGamma = 0.1
specEpochs = 10

#define dataset classes
class AVDataset(Dataset):
    #initialize
    def __init__(self, folder):
        self.files = list(Path(folder).glob("*.pt"))
    
    #get length
    def __len__(self):
        return len(self.files)

    #load file
    def __getitem__(self, idx):
        data = torch.load(self.files[idx])
        
        #don't load the raw audio since it's unneeded with the spectrogram
        return data["spec"], data["video"], data["label"]

#define model classes

#Residual block to avoid vanishing gradient, to be used with the Resnet 18
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1, bias = False)
        self.bn1 = nn.batchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = stride, bias = False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out

#ResNet18 for handling the spectrogram
class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        
        #in channels should equal the shape[0] of the spectrogram
        #128 in this case
        self.in_channels = 128
        self.conv1 = nn.Conv2d(3, 128, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace = True)
        
        self.layer1 = self._make_layer(ResBlock, 128, 2, stride = 1)
        self.layer2 = self._make_layer(ResBlock, 256, 2, stride = 2)
        self.layer3 = self._make_layer(ResBlock, 512, 2, stride = 2)
        self.layer4 = self._make_layer(ResBlock, 1024, 2, stride = 2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        #output a 256 size vector
        self.fc = nn.Linear(1024, 256)
    
    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
        

def makeSpectrogram(audioTensor, sr=25000, targetSr=16000):
    """Convert raw audio tensor to mel-spectrogram"""
    audio = audioTensor.numpy().squeeze()
    
    # Resample to 16kHz
    if sr != targetSr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=targetSr)
    
    # Take first 3 seconds, pad if shorter
    maxLen = targetSr * 3
    if len(audio) > maxLen:
        audio = audio[:maxLen]
    else:
        audio = np.pad(audio, (0, maxLen - len(audio)))
    
    # Make mel-spectrogram
    mel = librosa.feature.melspectrogram(
        y=audio, sr=targetSr, n_mels=128, fmax=8000
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    
    return torch.tensor(mel_db, dtype=torch.float32)

_faceCascade = None

#function to extract the most prominent face from a video frame, crop it, and resize it to a target size (default 112x112)
def extractStillFace(frame, targetSize=112):
    global _faceCascade

    if frame is None or frame.size == 0:
        raise ValueError("Frame is empty; cannot extract face.")

    if _faceCascade is None:
        cascadePath = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        _faceCascade = cv2.CascadeClassifier(cascadePath)
        if _faceCascade.empty():
            raise RuntimeError(f"Failed to load Haar cascade at: {cascadePath}")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    faces = _faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda faceBox: faceBox[2] * faceBox[3])
        face = frame[y:y+h, x:x+w]
    else:
        h, w = frame.shape[:2]
        minDim = min(h, w)
        startX = (w - minDim) // 2
        startY = (h - minDim) // 2
        face = frame[startY:startY+minDim, startX:startX+minDim]

    faceResized = cv2.resize(face, (targetSize, targetSize), interpolation=cv2.INTER_AREA)
    if faceResized.ndim == 2:
        faceResized = cv2.cvtColor(faceResized, cv2.COLOR_GRAY2BGR)

    return torch.tensor(faceResized, dtype=torch.float32).permute(2, 0, 1) / 255.0

#loads in all the data from a /data file and converts it to numpy
#startFrom lets you start from a specific speaker so not every file needs to be constantly remade
def setupData(startFrom = 1):
    #all the paths

    videoPath = Path('data/3625687')
    audioPath = Path('data/3625687/audio_25k/audio_25k')
    
    savePath = Path('processed_data')
    savePath.mkdir(exist_ok=True)

    numSpeakers = 34
    sampleId = 0

    for n in range(startFrom - 1, numSpeakers):
        #speaker 21 just doesn't exist? so skip it
        if n == 20:
            continue
        
        print("beginning speaker:", n + 1)

        speakerId = f"s{n+1}"

        speakerVideoPath = videoPath / speakerId / speakerId
        speakerAudioPath = audioPath / speakerId
        videos = os.listdir(speakerVideoPath)
        audios = os.listdir(speakerAudioPath)
        
        #for each video
        for v in videos:
            #make sure the video is actually a video
            if not v.lower().endswith(".mpg"):
                continue
            
            vPath = os.path.join(speakerVideoPath, v)
            cap = cv2.VideoCapture(vPath)
            
            #go through each frame and add them to a list
            frames = []
            frameCounter = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frameCounter % sampleRate == 0:
                    frame = cv2.resize(frame, (resizeResolution, resizeResolution))
                    frames.append(frame)
                frameCounter += 1
            
            cap.release()
            
            #stack all the frames into one numpy array then add them to the set
            if not frames:
                continue
            
            videoArr = np.stack(frames)
            
            #then get the matching audio
            try:
                aPath = os.path.join(speakerAudioPath, v[:-3] + "wav")
                print("Trying:", aPath, os.path.exists(aPath))
                
                waveform, sr = torchaudio.load(aPath)
                audioArr = waveform.numpy()
            #if no matching audio then toss out the video since they need to be corresponding for this to work
            except Exception as e:
                print("No matching audio")
                print(e)
                continue
            
            #tensor the 3 forms of data
            audioTensor = torch.tensor(audioArr)
            videoTensor = torch.tensor(videoArr)
            label = n+1
            
            #make a spectrogram from the audio
            spec = makeSpectrogram(audioTensor)
            
            #store the matching pieces together
            sample = {
                "audio": audioTensor,
                "video": videoTensor,
                "label": label,
                "spec": spec
            }

            #save to files for later
            torch.save(sample, savePath / Path("sample" + str(sampleId) + ".pt"))
            sampleId += 1

#load the data from a particular speaker
def loadSpeaker(speakerNum):
    audioTensor = torch.load(Path('data') / Path("s" + speakerNum + "audio.pt"))
    videoTensor = torch.load(Path('data') / Path("s" + speakerNum + "video.pt"))
    speakerTensor = torch.load(Path('data') / Path("s" + speakerNum + "labels.pt"))
    
    return audioTensor, videoTensor, speakerTensor

#collate function to properly pad the video
#spectrograms are already padded
#drops the audio for now since i dont think we need it, just the spectrogram
def collate_fn(batch):
    specs, videos, labels = zip(*batch)
    
    #video padding
    #get the largest video
    mostFrames = max(v.shape[0] for v in videos)
    
    paddedVideos = []
    #for each video in the batch
    for v in videos:
        extraFrames = mostFrames - v.shape[0]
        
        #add extra empty frames to match length of the longest
        padTensor = torch.zeros((extraFrames, *v.shape[1:]), dtype=v.dtype)
        v = torch.cat([v, padTensor], dim = 0)
        
        paddedVideos.append(v)
    
    return (
        torch.stack(specs),
        torch.stack(paddedVideos),
        torch.tensor(labels)
    )

#training loop
def train(model, device, trainLoader, testLoader, epochs):
    size = len(trainLoader.dataset)
    
    #train mode for batch norm
    model.train()
    
    #for batch, (X, y) in enumerate(trainLoader):
    #    X, y = X.to(device), y.to(device)
    #    pred = model(X)
    #    loss = lossFunc(pred, y)
    #    
    #    #backwards
    #    loss.backward()
    #    optimizer.step()
    #    optimizer.zero_grad()
    #    
    #    if batch % 100 == 0:
    #        #print loss at current step for debug purposes
    #        loss, current = loss.item(), batch * batchSize + len(X)
    #        print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

if __name__ == "__main__":
    #setup data only needs to be ran if the .pt files have not already been created
    #setupData()
    
    #setup dataloaders
    data = AVDataset("processed_data")
    
    #80/20 split
    trainSplit = int(0.8 * len(data))
    testSplit = len(data) - trainSplit
    
    trainData, testData = random_split(data, [trainSplit, testSplit])
    
    trainLoader = DataLoader(trainData, batch_size=batchSize, shuffle=True, num_workers=2, collate_fn=collate_fn, pin_memory=True)
    testLoader = DataLoader(testData, batch_size=batchSize, shuffle=False, num_workers=2, collate_fn=collate_fn, pin_memory=True)
    
    
    spec, vid, label = trainData[0]
    print(spec.shape)
    print(vid.shape)
    
    #setup model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    #define the spectrogram model
    specModel = ResNet18().to(device)
    
    #Setup criterion for the ResNet18
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(specModel.parameters(), lr = specLearningRate, momentum = specMomentum, weight_decay = specWeightDecay)
    scheduler = optim.lr_scheduler(optimizer, step_size = specStepSize, gamma = specGamma)
    
    pass