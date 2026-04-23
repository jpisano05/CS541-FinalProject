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

def loadAudioArray(audioPath):
    """Load WAV audio without TorchCodec; returns (channels, samples) float32 array and sample rate."""
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

#loads in all the data and converts it to numpy
#startFrom lets you start from a specific speaker so not every file needs to be constantly remade
def setupData(startFrom = 0):
    # Prefer local gridcorpus layout, then fall back to legacy path layout.
    videoPath = Path('gridcorpus/video') if Path('gridcorpus/video').exists() else Path('data/3625687')
    audioPath = Path('gridcorpus/audio_25k') if Path('gridcorpus/audio_25k').exists() else Path('data/3625687/audio_25k/audio_25k')
    outputPath = Path('data')
    outputPath.mkdir(parents=True, exist_ok=True)

    if not videoPath.exists():
        raise FileNotFoundError(f"Video dataset path not found: {videoPath}")
    if not audioPath.exists():
        raise FileNotFoundError(
            f"Audio dataset path not found: {audioPath}. "
            "Extract gridcorpus/audio_25k.zip so files exist under gridcorpus/audio_25k/s*/."
        )

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

            arr = np.stack(frames)
            videoSet.append(arr)
            
            #then get the matching audio
            try:
                aPath = os.path.join(speakerAudioPath, v[:-3] + "wav")
                print("Trying:", aPath, os.path.exists(aPath))
                if not os.path.exists(aPath):
                    raise FileNotFoundError(aPath)

                arr, sr = loadAudioArray(aPath)
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
        torch.save(audioTensor, outputPath / Path(speakerId + "audio.pt"))
        torch.save(videoTensor, outputPath / Path(speakerId + "video.pt"))
        torch.save(speakerTensor, outputPath / Path(speakerId + "labels.pt"))

        # save spectrograms
        specList = []
        for a in audioSet:
            spec = makeSpectrogram(torch.tensor(a))
            specList.append(spec)
        torch.save(specList, outputPath / Path(speakerId + "spectrograms.pt"))

#load the data from a particular speaker
def loadSpeaker(speakerNum):
    audioTensor = torch.load(Path('data') / Path("s" + speakerNum + "audio.pt"))
    videoTensor = torch.load(Path('data') / Path("s" + speakerNum + "video.pt"))
    speakerTensor = torch.load(Path('data') / Path("s" + speakerNum + "labels.pt"))
    
    return audioTensor, videoTensor, speakerTensor


if __name__ == "__main__":
    setupData(22)
    pass
