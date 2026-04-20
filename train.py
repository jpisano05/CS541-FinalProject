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

#data path
dataPath = '/data'

#decides 1 in sampleRate frames are utilized
#must be sampled as the full frameage is way to big to process
sampleRate = 5
#resolution to resize video frames to
resizeResolution = 112

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

#load the data from a particular speaker
def loadSpeaker(speakerNum):
    audioTensor = torch.load(Path('data') / Path("s" + speakerNum + "audio.pt"))
    videoTensor = torch.load(Path('data') / Path("s" + speakerNum + "video.pt"))
    speakerTensor = torch.load(Path('data') / Path("s" + speakerNum + "labels.pt"))
    
    return audioTensor, videoTensor, speakerTensor


if __name__ == "__main__":
    setupData(22)
    pass