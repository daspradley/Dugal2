"""
Standalone Azure Speech recognition test.
Run this directly to confirm Azure can hear you independently of Dugal.

Usage:
    python test_azure_mic.py

It will listen for 15 seconds and print anything it hears.
"""

import os
import time

# Load credentials from the same .env file Dugal uses
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import azure.cognitiveservices.speech as speechsdk

# --- Credentials ---
# If these aren't in your .env, paste them directly here temporarily
SPEECH_KEY    = os.environ.get("AZURE_SPEECH_KEY") or os.environ.get("SPEECH_KEY", "")
SPEECH_REGION = os.environ.get("AZURE_SPEECH_REGION") or os.environ.get("SPEECH_REGION", "centralus")

if not SPEECH_KEY:
    # Try reading from the .env file directly as a fallback
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("AZURE_SPEECH_KEY=") or line.startswith("SPEECH_KEY="):
                    SPEECH_KEY = line.split("=", 1)[1].strip().strip('"').strip("'")
                if line.startswith("AZURE_SPEECH_REGION=") or line.startswith("SPEECH_REGION="):
                    SPEECH_REGION = line.split("=", 1)[1].strip().strip('"').strip("'")

if not SPEECH_KEY:
    print("ERROR: Could not find Azure speech key.")
    print("Edit this file and set SPEECH_KEY directly, or make sure your .env has AZURE_SPEECH_KEY=...")
    exit(1)

print(f"Using region: {SPEECH_REGION}")
print(f"Key (first 10): {SPEECH_KEY[:10]}...")
print()

# --- Configure ---
speech_config = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
speech_config.speech_recognition_language = "en-US"
audio_config  = speechsdk.audio.AudioConfig(use_default_microphone=True)
recognizer    = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

results = []

def recognized_cb(evt):
    if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
        text = evt.result.text
        print(f"✅ RECOGNIZED: '{text}'")
        results.append(text)
    elif evt.result.reason == speechsdk.ResultReason.NoMatch:
        print(f"❌ No match: {evt.result.no_match_details}")

def canceled_cb(evt):
    print(f"⛔ CANCELED: {evt.reason}")
    if evt.reason == speechsdk.CancellationReason.Error:
        print(f"   Error details: {evt.error_details}")

def session_started_cb(evt):
    print("🎤 Session started — speak now!")

def recognizing_cb(evt):
    print(f"   ... hearing: '{evt.result.text}'")

recognizer.recognized.connect(recognized_cb)
recognizer.canceled.connect(canceled_cb)
recognizer.session_started.connect(session_started_cb)
recognizer.recognizing.connect(recognizing_cb)

print("Starting Azure Speech recognition for 15 seconds...")
print("Say something clearly after '🎤 Session started'")
print("-" * 50)

recognizer.start_continuous_recognition()
time.sleep(15)
recognizer.stop_continuous_recognition()

print("-" * 50)
if results:
    print(f"✅ SUCCESS — Azure heard {len(results)} utterance(s):")
    for r in results:
        print(f"   '{r}'")
else:
    print("❌ NOTHING RECOGNIZED in 15 seconds.")
    print()
    print("Possible causes:")
    print("  1. Microphone input level too low — check Windows Sound settings -> Recording -> Properties -> Levels")
    print("  2. Azure key or region wrong")
    print("  3. Network issue blocking Azure connection")
    print("  4. Microphone not actually the default device for this audio stack")
