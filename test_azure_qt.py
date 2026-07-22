"""
Minimal Qt + Azure Speech recognition test.
Tests whether Azure can hear speech while a Qt event loop is running.

Usage:
    python test_azure_qt.py

Speak after '🎤 Session started'. Results print to console.
App closes automatically after 20 seconds.
"""

import os
import sys
import pathlib

# Load key directly from .env
SPEECH_KEY = None
SPEECH_REGION = "centralus"
env_path = pathlib.Path(__file__).parent / ".env"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if '=' not in line or line.startswith('#'):
                continue
            k, v = line.split('=', 1)
            k = k.strip(); v = v.strip().strip('"').strip("'")
            if k in ('AZURE_SPEECH_KEY', 'SPEECH_KEY') and not SPEECH_KEY:
                SPEECH_KEY = v
            if k in ('AZURE_REGION', 'SPEECH_REGION') and SPEECH_REGION == 'centralus':
                SPEECH_REGION = v

if not SPEECH_KEY:
    print("ERROR: No Azure key found in .env")
    sys.exit(1)

print(f"Key (first 10): {SPEECH_KEY[:10]}...")
print(f"Region: {SPEECH_REGION}")

from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget
from PyQt5.QtCore import QTimer
import azure.cognitiveservices.speech as speechsdk

app = QApplication(sys.argv)

# Simple window
win = QWidget()
win.setWindowTitle("Azure Qt Test")
layout = QVBoxLayout()
status = QLabel("Starting Azure...")
results_label = QLabel("Heard: (nothing yet)")
layout.addWidget(status)
layout.addWidget(results_label)
win.setLayout(layout)
win.resize(400, 150)
win.show()

# Azure setup
speech_config = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
speech_config.speech_recognition_language = "en-US"
audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

results = []

def on_recognized(evt):
    if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
        text = evt.result.text
        print(f"✅ RECOGNIZED: '{text}'")
        results.append(text)
        results_label.setText(f"Heard: {text}")
    elif evt.result.reason == speechsdk.ResultReason.NoMatch:
        print(f"❌ No match")

def on_recognizing(evt):
    if evt.result.reason == speechsdk.ResultReason.RecognizingSpeech:
        print(f"   ... hearing: '{evt.result.text}'")
        status.setText(f"Hearing: {evt.result.text}")

def on_canceled(evt):
    print(f"⛔ CANCELED: {evt.reason}")
    if evt.reason == speechsdk.CancellationReason.Error:
        print(f"   Error: {evt.error_details}")

def on_session_started(evt):
    print("🎤 Session started — speak now!")
    status.setText("🎤 Listening... speak now!")

recognizer.recognized.connect(on_recognized)
recognizer.recognizing.connect(on_recognizing)
recognizer.canceled.connect(on_canceled)
recognizer.session_started.connect(on_session_started)

# Start async (same as Dugal v10)
recognizer.start_continuous_recognition_async()
print("Recognition started (async). Qt event loop running.")

# Auto-close after 20 seconds
def finish():
    recognizer.stop_continuous_recognition_async()
    if results:
        print(f"\n✅ SUCCESS — Qt + Azure heard {len(results)} utterance(s):")
        for r in results:
            print(f"   '{r}'")
    else:
        print("\n❌ NOTHING recognized in 20 seconds under Qt event loop.")
        print("The problem IS the Qt/Azure interaction.")
    app.quit()

QTimer.singleShot(20000, finish)
sys.exit(app.exec_())
