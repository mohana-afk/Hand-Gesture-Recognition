# Hand Gesture Recognition System

This project is a real-time hand gesture recognition system built using OpenCV, MediaPipe, and Machine Learning.

## Features
- Detects hand landmarks using MediaPipe
- Trains a machine learning model (Random Forest)
- Recognizes gestures in real-time using webcam

## Tech Stack
- Python
- OpenCV
- MediaPipe
- Scikit-learn

## How It Works
1. Hand landmarks are captured using MediaPipe
2. Data is stored and used to train a model
3. Model predicts gestures from live webcam feed

## Setup
1. Install dependencies:
pip install -r requirements.txt

2. Run:
python prediction.py
