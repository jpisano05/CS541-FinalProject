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
def setupData(startFrom = 0):
    #all the paths

    videoPath = Path('data/3625687')
    audioPath = Path('data/3625687/audio_25k/audio_25k')

    numSpeakers = 34

    for n in range(startFrom - 1, numSpeakers):
        #speaker 21 just doesn't exist? so skip it
        if n == 20:
            continue
        
        print("beginning speaker:", n + 1)

        audioSet = []
        videoSet = []
        speakerSet = []

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
            if frames:
                arr = np.stack(frames)
                videoSet.append(arr)
            
            #then get the matching audio
            try:
                aPath = os.path.join(speakerAudioPath, v[:-3] + "wav")
                print("Trying:", aPath, os.path.exists(aPath))
                
                waveform, sr = torchaudio.load(aPath)
                arr = waveform.numpy()
                audioSet.append(arr)
            #if no matching audio then toss out the video since they need to be corresponding for this to work
            except Exception as e:
                print("No matching audio")
                print(e)
                videoSet.pop()
                continue
            
            #then append the speaker number to speaker set as a label
            speakerSet.append(n+1)
    
        #tensor the 3 sets
        audioTensor = [torch.tensor(a) for a in audioSet]
        videoTensor = [torch.tensor(v) for v in videoSet]
        speakerTensor = torch.tensor(speakerSet)
        
        #save to files for later
        torch.save(audioTensor, Path('data') / Path(speakerId + "audio.pt"))
        torch.save(videoTensor, Path('data') / Path(speakerId + "video.pt"))
        torch.save(speakerTensor, Path('data') / Path(speakerId + "labels.pt"))

        # save spectrograms
        specList = []
        for a in audioSet:
            spec = makeSpectrogram(torch.tensor(a))
            specList.append(spec)
        torch.save(specList, Path('data') / Path(speakerId + "spectrograms.pt"))

#load the data from a particular speaker
def loadSpeaker(speakerNum):
    audioTensor = torch.load(Path('data') / Path("s" + speakerNum + "audio.pt"))
    videoTensor = torch.load(Path('data') / Path("s" + speakerNum + "video.pt"))
    speakerTensor = torch.load(Path('data') / Path("s" + speakerNum + "labels.pt"))
    
    return audioTensor, videoTensor, speakerTensor


if __name__ == "__main__":
    setupData(22)
    pass