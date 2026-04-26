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
specLearningRate = 1e-3
specMomentum = 0.9
specWeightDecay = 5e-4
specStepSize = 30
specGamma = 0.1
specEpochs = 5
contrastiveTemperature = 0.07

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
        self.bn1 = nn.BatchNorm2d(out_channels)
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
        self.conv1 = nn.Conv2d(1, 128, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(128)
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
        
#Creates face-voice pairs for contrastive learning
class FaceVoicePairDataset(Dataset):

    def __init__(self, speakerList, numPairs=50000):
        self.pairs = []
        self.speakerData = {}
        
        # Load all speaker data
        for spk in speakerList:
            spkStr = str(spk)
            
            # Load video and spectrograms
            videoTensor = torch.load(
                Path('data') / Path("s" + spkStr + "video.pt")
            )
            specList = torch.load(
                Path('data') / Path("s" + spkStr + "spectrograms.pt")
            )
            
            self.speakerData[spk] = {
                'video': videoTensor,
                'specs': specList
            }
        
        speakerListCopy = list(speakerList)
        
        # Create positive pairs (same person, different clips)
        for _ in range(numPairs // 2):
            spk = random.choice(speakerListCopy)
            data = self.speakerData[spk]
            numClips = len(data['video'])
            if numClips < 2:
                continue
            
            # Pick two different clips from the same person
            i, j = random.sample(range(numClips), 2)
            self.pairs.append({
                'face_spk': spk,
                'face_idx': i,      # face from clip i
                'voice_spk': spk,
                'voice_idx': j,     # voice from clip j
                'label': 1          # same person
            })
        
        # Create negative pairs (different people)
        for _ in range(numPairs // 2):
            spk1, spk2 = random.sample(speakerListCopy, 2)
            data1 = self.speakerData[spk1]
            data2 = self.speakerData[spk2]
            
            i = random.randint(0, len(data1['video']) - 1)
            j = random.randint(0, len(data2['specs']) - 1)
            
            self.pairs.append({
                'face_spk': spk1,
                'face_idx': i,      # face from person 1
                'voice_spk': spk2,
                'voice_idx': j,     # voice from person 2
                'label': 0          # different people
            })
        
        random.shuffle(self.pairs)
        print(f"Created {len(self.pairs)} pairs "
              f"({numPairs // 2} positive, {numPairs // 2} negative)")
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        
        # Get face: grab middle frame from the video, crop the face
        video = self.speakerData[pair['face_spk']]['video'][pair['face_idx']]
        middleFrame = video[len(video) // 2].numpy()
        face = extractStillFace(middleFrame, targetSize=224)
        # extractStillFace already returns (3, 224, 224) normalized tensor
        
        # Get voice spectrogram
        spec = self.speakerData[pair['voice_spk']]['specs'][pair['voice_idx']]
        # Add channel dimension: (128, time) -> (1, 128, time)
        if spec.dim() == 2:
            spec = spec.unsqueeze(0)
        
        label = pair['label']
        
        return face, spec, label
    
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

def loadAudioArray(audioPath):
    """Load WAV audio without TorchCodec; returns (channels, samples) float32 array."""
    audio, sr = librosa.load(audioPath, sr=None, mono=False)
    if audio.ndim == 1:
        audio = np.expand_dims(audio, axis=0)
    return audio.astype(np.float32), sr

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
    audioPathOptions = [Path('data/3625687/audio_25k'), Path('data/3625687/audio_25k/audio_25k')]
    audioPath = next((p for p in audioPathOptions if p.exists()), audioPathOptions[0])
    if not videoPath.exists():
        raise FileNotFoundError(f"Video dataset path not found: {videoPath}")
    if not audioPath.exists():
        raise FileNotFoundError(f"Audio dataset path not found. Tried: {audioPathOptions}")
    
    savePath = Path('data')
    savePath.mkdir(exist_ok=True)

    numSpeakers = 34
    sampleId = 0

    for n in range(startFrom - 1, numSpeakers):
        #speaker 21 just doesn't exist? so skip it
        if n == 20:
            continue
        
        print("beginning speaker:", n + 1)

        speakerId = f"s{n+1}"

        speakerVideoPath = videoPath / speakerId
        if not speakerVideoPath.exists():
            speakerVideoPath = videoPath / speakerId / speakerId
        speakerAudioPath = audioPath / speakerId
        if not speakerVideoPath.exists():
            print(f"Skipping {speakerId}: missing video path {speakerVideoPath}")
            continue
        if not speakerAudioPath.exists():
            print(f"Skipping {speakerId}: missing audio path {speakerAudioPath}")
            continue

        videos = os.listdir(speakerVideoPath)
        
        #for each video
        for v in videos:
            #make sure the video is actually a video
            if not v.lower().endswith(".mpg") or v.startswith("._"):
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
                    face = extractStillFace(frame, targetSize=resizeResolution)
                    frame = (face.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
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
                if not os.path.exists(aPath):
                    raise FileNotFoundError(aPath)

                audioArr, sr = loadAudioArray(aPath)
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
    
    #adjust spec size
    specs = [spec.unsqueeze(0) for spec in specs]
    
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

#custom contrastive loss function because pytorch doesn't have one by default
def contrastiveLoss(specEmbed, faceEmbed, temperature=contrastiveTemperature):
    #normalize both
    specEmbed = F.normalize(specEmbed, dim=1)
    faceEmbed = F.normalize(faceEmbed, dim=1)
    
    #get the logits
    logits = specEmbed @ faceEmbed.T
    logits = logits / temperature
    
    labels = torch.arange(specEmbed.size(0), device=specEmbed.device)
    
    #compare
    lossS2F = F.cross_entropy(logits, labels)
    lossF2S = F.cross_entropy(logits.T, labels)
    
    return (lossS2F + lossF2S) / 2
    

#training loop for training a spectrogram into 256 and a face into 256
#this does not work with the videos yet, a seperate function will be made for that
def train(specModel, faceModel, trainLoader, testLoader, optimizer, device):
    #normalize so that the input matches what the pretrained resnet 50 actually wants
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1,3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1,3,1,1)
    
    #scaler for speedup
    scaler = torch.amp.GradScaler()
    
    for epoch in range(specEpochs):
        #set both models to train
        specModel.train()
        faceModel.train()
        trainLoss = 0.0
        batches = 0
        
        #training set
        for specs, videos, labels in trainLoader:
            specs = specs.to(device)
            videos = videos.to(device)
            
            #forward prop
            #first frame of each batch to get a face still
            faces = videos[:, 0].to(device)
            faces = faces.permute(0, 3, 1, 2)
            faces = F.interpolate(faces, size=(224, 224))

            faces = (faces - mean) / std
            
            optimizer.zero_grad()
            
            #wrap in a scaler
            with torch.amp.autocast(device_type="cuda"):
                specEmbed = specModel(specs)
                faceEmbed = faceModel(faces)
            
                #calculate loss
                loss = contrastiveLoss(specEmbed, faceEmbed)
            
            trainLoss += loss.item()
            batches += 1
            
            if batches % 10 == 0:
                print(f"Epoch {epoch} | Batch {batches} | Loss: {loss.item()}")
            
            #backward prop
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        trainLoss /= batches
        
        scheduler.step()
        
        #test set
        specModel.eval()
        faceModel.eval()
        testLoss = 0.0
        batches = 0
        
        with torch.inference_mode():
            for specs, videos, labels in testLoader:
                specs = specs.to(device)
                videos = videos.to(device)
                
                #forward prop
                specEmbed = specModel(specs)
                
                faceEmbeds = []
                #first frame of each batch to get a face still
                faces = videos[:, 0].to(device)
                faces = faces.permute(0, 3, 1, 2)
                faces = F.interpolate(faces, size=(224, 224))

                faces = (faces - mean) / std

                faceEmbed = faceModel(faces)
                
                #calculate loss
                loss = contrastiveLoss(specEmbed, faceEmbed)
                testLoss += loss.item()
                batches += 1
        
        #print the average loss over training and test set to get an idea of progress
        testLoss /= batches
        print(f"Epoch {epoch} | Train Loss: {trainLoss} | Test Loss: {testLoss}")
            

if __name__ == "__main__":
    #setup data only needs to be ran if the .pt files have not already been created
    #setupData(1)
    
    #setup dataloaders
    data = AVDataset("processed_data")
    
    #80/20 split
    trainSplit = int(0.8 * len(data))
    testSplit = len(data) - trainSplit
    
    trainData, testData = random_split(data, [trainSplit, testSplit])
    
    trainLoader = DataLoader(trainData, batch_size=batchSize, shuffle=True, num_workers=8, persistent_workers=True, prefetch_factor=4, collate_fn=collate_fn, pin_memory=True)
    testLoader = DataLoader(testData, batch_size=batchSize, shuffle=False, num_workers=8, persistent_workers=True, prefetch_factor=4, collate_fn=collate_fn, pin_memory=True)
    
    
    spec, vid, label = trainData[0]
    print(spec.shape)
    print(vid.shape)
    
    #setup model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    #define the spectrogram model
    specModel = ResNet18().to(device)
    
    #define the face model
    weights = models.ResNet50_Weights.DEFAULT
    faceModel = models.resnet50(weights=weights)
    #get a preprocess to give the resnet50 the shape it wants
    preprocess = weights.transforms()
    #alter output layer to fit the 256 we want
    faceModel.fc = nn.Linear(faceModel.fc.in_features, 256)
    
    #freeze part of the face model since its pretrained
    for param in faceModel.parameters():
        param.requires_grad = False
    for param in faceModel.fc.parameters():
        param.requires_grad = True
    
    faceModel = faceModel.to(device)
    
    #Setup criterion
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(list(specModel.parameters()) + list(faceModel.parameters()), lr = specLearningRate, momentum = specMomentum, weight_decay = specWeightDecay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = specStepSize, gamma = specGamma)
    
    
    
    train(specModel, faceModel, trainLoader, testLoader, optimizer, device)
    
    #save weights to be loaded back in later
    torch.save(specModel.state_dict(), "specModel.pth")
    torch.save(faceModel.state_dict(), "faceModel.pth")
    pass

    # #Testing the pair dataset on 3 speakers
    # trainSpeakers = [1, 2, 3]
    # dataset = FaceVoicePairDataset(trainSpeakers, numPairs=100)
    
    # loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # for faces, specs, labels in loader:
    #     print(f"Faces shape: {faces.shape}")
    #     print(f"Specs shape: {specs.shape}")
    #     print(f"Labels: {labels}")
    #     break
