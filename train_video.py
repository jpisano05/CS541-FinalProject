import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, Dataset, random_split
from torchvision import datasets, models
from torchvision.models.video import R3D_18_Weights, r3d_18
from sklearn.metrics import roc_curve
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

#number of frames to user per vidoe
numFrames = 16

#hyperparameters
batchSize = 32 #smaller batch since video data uses more memory
specLearningRate = 1e-3
specMomentum = 0.9
specWeightDecay = 5e-4
specStepSize = 30
specGamma = 0.1
specEpochs = 15
contrastiveTemperature = 0.1

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
 
#Convert raw audio tensor to mel-spectrogram
def makeSpectrogram(audioTensor, sr=25000, targetSr=16000):
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

# collate function to properly pad the video
# spectrograms are already padded
# drops the audio for now since i dont think we need it, just the spectrogram
# what changed: collate function now pads video clips to numFrames and stacks them into a batch
def collate_fn(batch):
    specs, videos, labels = zip(*batch)
    
    specs = [spec.unsqueeze(0) for spec in specs]
    
    paddedVideos = []
    for v in videos:
        # Take first numFrames, or pad if shorter
        if v.shape[0] >= numFrames:
            v = v[:numFrames]
        else:
            extraFrames = numFrames - v.shape[0]
            padTensor = torch.zeros((extraFrames, *v.shape[1:]), dtype=v.dtype)
            v = torch.cat([v, padTensor], dim=0)
        
        paddedVideos.append(v)
    
    return (
        torch.stack(specs),
        torch.stack(paddedVideos),
        torch.tensor(labels)
    )
 
def contrastiveLoss(specEmbed, faceEmbed, temperature=contrastiveTemperature):
    specEmbed = F.normalize(specEmbed, dim=1, eps=1e-8)
    faceEmbed = F.normalize(faceEmbed, dim=1, eps=1e-8)
    
    logits = specEmbed @ faceEmbed.T
    logits = logits / temperature
    
    labels = torch.arange(specEmbed.size(0), device=specEmbed.device)
    
    lossS2F = F.cross_entropy(logits, labels)
    lossF2S = F.cross_entropy(logits.T, labels)
    
    return (lossS2F + lossF2S) / 2

# Convert video batch from (batch, frames, H, W, channels) 
# helper function to prepare video clips for R3D
# to what R3D expects: (batch, channels, frames, H, W) at 112x112
def prepareVideoClips(videos, device):
    """
   
    """
    # videos is (batch, frames, H, W, 3)
    clips = videos.to(device).float()
    
    # Rearrange: (batch, frames, H, W, C) -> (batch, C, frames, H, W)
    clips = clips.permute(0, 4, 1, 2, 3)
    
    # R3D expects 112x112, resize if needed
    B, C, T, H, W = clips.shape
    if H != 112 or W != 112:
        # Reshape to (B*T, C, H, W) to resize, then back
        clips = clips.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        clips = F.interpolate(clips, size=(112, 112))
        clips = clips.reshape(B, T, C, 112, 112).permute(0, 2, 1, 3, 4)
    
    # Normalize with ImageNet stats
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1, 1)
    
    # Normalize to 0-1 first if needed (frames are 0-255 uint8)
    if clips.max() > 1.0:
        clips = clips / 255.0
    
    clips = (clips - mean) / std
    
    return clips
 
# training loop for training a spectrogram into 256 and a face into 256
# this does not work with the videos yet, a seperate function will be made for that
# what changed: training loop uses video clips instead of single frames
def train(specModel, videoModel, trainLoader, testLoader, optimizer, device):
    scaler = torch.amp.GradScaler()
    
    for epoch in range(specEpochs):
        specModel.train()
        videoModel.train()
        trainLoss = 0.0
        batches = 0
        
        for specs, videos, labels in trainLoader:
            specs = specs.to(device)
            
            # NEW - prepare video clips for R3D (all frames, not just first)
            clips = prepareVideoClips(videos, device)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast(device_type="cuda"):
                specEmbed = specModel(specs)
                videoEmbed = videoModel(clips)
            
                loss = contrastiveLoss(specEmbed, videoEmbed)
            
            trainLoss += loss.item()
            batches += 1
            
            if batches % 10 == 0:
                print(f"Epoch {epoch} | Batch {batches} | Loss: {loss.item()}")
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        trainLoss /= batches
        
        scheduler.step()
        
        specModel.eval()
        videoModel.eval()
        testLoss = 0.0
        batches = 0
        
        with torch.inference_mode():
            for specs, videos, labels in testLoader:
                specs = specs.to(device)
                
                # NEW - use video clips
                clips = prepareVideoClips(videos, device)
                
                specEmbed = specModel(specs)
                videoEmbed = videoModel(clips)
                
                loss = contrastiveLoss(specEmbed, videoEmbed)
                testLoss += loss.item()
                batches += 1
        
        testLoss /= batches
        print(f"Epoch {epoch} | Train Loss: {trainLoss} | Test Loss: {testLoss}")
        
        
# Evaluate the trained model - verification and retrieval accuracy
# what changed: evaluate now uses video clips
def evaluate(specModel, videoModel, testLoader, device):
    specModel.eval()
    videoModel.eval()
 
    allSpecEmbeds = []
    allVideoEmbeds = []
    allLabels = []
 
    with torch.inference_mode():
        for specs, videos, labels in testLoader:
            specs = specs.to(device)
            
            # NEW - use video clips instead of single frame
            clips = prepareVideoClips(videos, device)
 
            specEmbed = F.normalize(specModel(specs), dim=1, eps=1e-8)
            videoEmbed = F.normalize(videoModel(clips), dim=1, eps=1e-8)
 
            allSpecEmbeds.append(specEmbed.cpu())
            allVideoEmbeds.append(videoEmbed.cpu())
            allLabels.append(labels)
 
    allSpecEmbeds = torch.cat(allSpecEmbeds)
    allVideoEmbeds = torch.cat(allVideoEmbeds)
    allLabels = torch.cat(allLabels)
 
    total = len(allLabels)
    uniqueSpeakers = allLabels.unique()
 
    print(f"\n{'='*50}")
    print("EVALUATION RESULTS (VIDEO ENCODER)")
    print(f"{'='*50}")
    print(f"Total test samples: {total}")
    print(f"Unique speakers: {len(uniqueSpeakers)}")
 
    # Verification
    similarity = allVideoEmbeds @ allSpecEmbeds.T
    predictions = similarity.argmax(dim=1)
 
    correct = (allLabels == allLabels[predictions]).sum().item()
    verifyAcc = correct / total * 100
    randomBaseline = 1 / len(uniqueSpeakers) * 100
 
    print("\n--- Verification ---")
    print(f"Accuracy: {verifyAcc:.2f}%")
    print(f"Random guess baseline: {randomBaseline:.2f}%")
 
    # EER
    labels_matrix = (allLabels.unsqueeze(1) == allLabels.unsqueeze(0)).float()
    mask = ~torch.eye(total, dtype=bool)
    scores = similarity[mask]
    targets = labels_matrix[mask]
 
    scores_np = scores.numpy()
    targets_np = targets.numpy()
 
    fpr, tpr, thresholds = roc_curve(targets_np, scores_np)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = fpr[eer_idx] * 100
 
    print("\n--- Verification (EER) ---")
    print(f"EER: {eer:.2f}%")
 
    # Retrieval
    print("\n--- Face-to-Voice Retrieval (Video) ---")
 
    for gallerySize in [10, 50, 100]:
        if gallerySize > total:
            continue
 
        top1Correct = 0
        top5Correct = 0
        trials = min(200, total)
        validTrials = 0
 
        for _ in range(trials):
            queryIdx = torch.randint(0, total, (1,)).item()
            queryVideo = allVideoEmbeds[queryIdx]
            queryLabel = allLabels[queryIdx]
 
            matchingIdxs = ((allLabels == queryLabel) & (torch.arange(total) != queryIdx)).nonzero().squeeze()
 
            if matchingIdxs.numel() == 0:
                continue
 
            matchIdx = matchingIdxs[torch.randint(0, len(matchingIdxs), (1,))].item()
 
            nonMatchIdxs = (allLabels != queryLabel).nonzero().squeeze()
 
            if len(nonMatchIdxs) < gallerySize - 1:
                continue
 
            randIdxs = nonMatchIdxs[torch.randperm(len(nonMatchIdxs))[:gallerySize - 1]]
            galleryIdxs = torch.cat([torch.tensor([matchIdx]), randIdxs])
 
            gallerySpecs = allSpecEmbeds[galleryIdxs]
            galleryLabels = allLabels[galleryIdxs]
 
            sims = F.cosine_similarity(queryVideo.unsqueeze(0), gallerySpecs)
            ranked = sims.argsort(descending=True)
 
            if galleryLabels[ranked[0]] == queryLabel:
                top1Correct += 1
 
            topk = min(5, gallerySize)
            if queryLabel in galleryLabels[ranked[:topk]]:
                top5Correct += 1
 
            validTrials += 1
 
        if validTrials == 0:
            continue
 
        print(f"\nGallery size {gallerySize}:")
        print(f"  Top-1 accuracy: {top1Correct/validTrials*100:.2f}%")
        print(f"  Top-5 accuracy: {top5Correct/validTrials*100:.2f}%")
        print(f"  Random baseline: {1/gallerySize*100:.2f}%")
 
    print(f"\n{'='*50}")
    return verifyAcc

if __name__ == "__main__":
    #setup dataloaders (same data as still face version)
    dataFolder = "data"
    data = AVDataset(dataFolder)
    if len(data) == 0:
        print("No preprocessed .pt samples found in data/. Running setupData(1)...")
        setupData(1)
        data = AVDataset(dataFolder)

    if len(data) < 2:
        raise RuntimeError(
            f"Need at least 2 preprocessed samples in {dataFolder}/, found {len(data)}."
        )
    
    #80/20 split
    trainSplit = int(0.8 * len(data))
    testSplit = len(data) - trainSplit
    if trainSplit == 0:
        trainSplit = 1
        testSplit = len(data) - 1
    if testSplit == 0:
        testSplit = 1
        trainSplit = len(data) - 1
    
    trainData, testData = random_split(data, [trainSplit, testSplit])
    
    trainLoader = DataLoader(trainData, batch_size=batchSize, shuffle=True, num_workers=min(4, os.cpu_count()), persistent_workers=True, prefetch_factor=2, collate_fn=collate_fn, pin_memory=True)
    testLoader = DataLoader(testData, batch_size=batchSize, shuffle=False, num_workers=min(4, os.cpu_count()), persistent_workers=True, prefetch_factor=2, collate_fn=collate_fn, pin_memory=True)
    
    spec, vid, label = trainData[0]
    print(f"Spec shape: {spec.shape}")
    print(f"Video shape: {vid.shape}")
    print(f"Using {numFrames} frames for video encoder")
    
    #setup model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True
    
    # Voice encoder: same ResNet18 as still face version
    # Load the TRAINED weights from the still face experiment
    specModel = ResNet18().to(device)
    specWeights = torch.load("trainedWeights/specModel.pth", map_location=device)
    specModel.load_state_dict(specWeights)
    print("Loaded trained spectrogram weights from still face experiment")
    
    # NEW - Video encoder: R3D-18 (pretrained on Kinetics-400)
    # This replaces the ResNet-50 still face encoder
    weights = R3D_18_Weights.DEFAULT
    videoModel = r3d_18(weights=weights)
    
    # Replace final layer to output 256-dim vector (same as face model)
    videoModel.fc = nn.Linear(videoModel.fc.in_features, 256)
    
    # Freeze pretrained layers, only train the final projection
    for param in videoModel.parameters():
        param.requires_grad = False
    for param in videoModel.fc.parameters():
        param.requires_grad = True
    
    videoModel = videoModel.to(device)
    
    # Setup optimizer — include both models
    # specModel params are included but frozen since we loaded trained weights
    optimizer = optim.SGD(
        list(filter(lambda p: p.requires_grad, specModel.parameters())) + 
        list(filter(lambda p: p.requires_grad, videoModel.parameters())), 
        lr=specLearningRate, momentum=specMomentum, weight_decay=specWeightDecay
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=specStepSize, gamma=specGamma)
    
    # Train or load
    doTrain = 1
    if doTrain:
        train(specModel, videoModel, trainLoader, testLoader, optimizer, device)
        
        torch.save(specModel.state_dict(), "specModel_video.pth")
        torch.save(videoModel.state_dict(), "videoModel.pth")
    else:
        videoWeights = torch.load("trainedWeights/videoModel.pth", map_location=device)
        videoModel.load_state_dict(videoWeights)
        videoModel.to(device)
    
    # Run evaluation
    evaluate(specModel, videoModel, testLoader, device)
    
 
