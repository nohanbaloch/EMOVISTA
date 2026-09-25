import os
import json
import sys
import queue
import sounddevice as sd
from vosk import Model, KaldiRecognizer

class VoskSTT:
    def __init__(self, model_path: str, sample_rate: int = 16000):
        model_path = os.path.abspath(model_path)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Vosk model path not found: {model_path}")

        self.model = Model(model_path)
        self.recognizer = KaldiRecognizer(self.model, sample_rate)
        self.sample_rate = sample_rate
        self.q = queue.Queue()

    def callback(self, indata, frames, time, status):
        """This is called (from a separate thread) for each audio block."""
        if status:
            print(status, file=sys.stderr)
        self.q.put(bytes(indata))

    def record_and_transcribe(self):
        """
        Opens the microphone, records audio, feeds to Vosk, and returns text.
        Stops when a final result is found or after a brief silence/timeout logic implies done.
        For simplicity, returns the first final result.
        """
        print("Listening for speech...", file=sys.stderr)
        
        # Open stream
        with sd.RawInputStream(samplerate=self.sample_rate, blocksize=8000, device=None, dtype='int16',
                               channels=1, callback=self.callback):
            
            # Loop until we get a full result
            while True:
                data = self.q.get()
                if self.recognizer.AcceptWaveform(data):
                    result = json.loads(self.recognizer.Result())
                    text = result.get("text", "")
                    if text: # Return as soon as we have text
                        print(f"Recognized: {text}", file=sys.stderr)
                        return text
                else:
                    # Partial result logic if we wanted to stream
                    pass
                
    def accept_audio(self, pcm_bytes: bytes):
        return self.recognizer.AcceptWaveform(pcm_bytes)

    def get_result(self):
        result = json.loads(self.recognizer.Result())
        return result.get("text", "")

    def get_partial(self):
        result = json.loads(self.recognizer.PartialResult())
        return result.get("partial", "")
