import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torchvision import datasets, models
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
from torch.optim import lr_scheduler
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

if __name__ == "__main__":
    #setup data only needs to be ran if the .pt files have not already been created
    setupData()
    
    #setup dataloaders
    data = AVDataset("processed_data")
    loader = DataLoader(data, batch_size=batchSize, shuffle=True, num_workers=2, collate_fn=collate_fn, pin_memory=True)
    
    #setup model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    pass
