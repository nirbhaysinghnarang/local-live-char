import argparse
import os
import time
import threading
import signal
from queue import Queue
from enum import Enum
from typing import Optional

import numpy as np
from silero_vad import VADIterator, load_silero_vad
from sounddevice import InputStream, stop, play
from langchain_ollama import OllamaLLM
from moonshine_onnx import MoonshineOnnxModel, load_tokenizer
from pvorca import create

# Audio settings
SAMPLING_RATE = 16000
CHUNK_SIZE = 512
LOOKBACK_CHUNKS = 5
MAX_SPEECH_SECS = 15

class BotState(Enum):
    LISTENING = "LISTENING"
    SPEAKING = "SPEAKING"
    PROCESSING = "PROCESSING"

class Transcriber:
    def __init__(self, model_name, rate=16000):
        if rate != 16000:
            raise ValueError("Moonshine supports sampling rate 16000 Hz.")
        self.model = MoonshineOnnxModel(model_name=model_name)
        self.rate = rate
        self.tokenizer = load_tokenizer()
        self.inference_secs = 0
        self.number_inferences = 0
        self.speech_secs = 0
        self.__call__(np.zeros(int(rate), dtype=np.float32))  # Warmup

    def __call__(self, speech):
        """Returns string containing Moonshine transcription of speech."""
        self.number_inferences += 1
        self.speech_secs += len(speech) / self.rate
        start_time = time.time()
        
        tokens = self.model.generate(speech[np.newaxis, :].astype(np.float32))
        text = self.tokenizer.decode_batch(tokens)[0]
        
        self.inference_secs += time.time() - start_time
        return text

class TTSManager:
    def __init__(self, access_key: str):
        try:
            self.orca = create(access_key=access_key)
            self.stream_lock = threading.Lock()
            self.current_stream = None
        except Exception as e:
            print(f"Failed to initialize Orca: {e}")
            raise

    def start_stream(self):
        """Start a new Orca stream"""
        with self.stream_lock:
            if self.current_stream is not None:
                self.stop_stream()
            self.current_stream = self.orca.stream_open()
    
    def process_stream(self, text: str) -> Optional[np.ndarray]:
        """Process a chunk of text and return audio if available"""
        with self.stream_lock:
            if self.current_stream is not None:
                try:
                    pcm = self.current_stream.synthesize(text)
                    if pcm is not None:
                        return np.array(pcm, dtype=np.float32) / 32768.0
                except Exception as e:
                    print(f"Failed to process stream: {e}")
            return None

    def flush_stream(self) -> Optional[np.ndarray]:
        """Flush the remaining text in the stream"""
        with self.stream_lock:
            if self.current_stream is not None:
                try:
                    pcm = self.current_stream.flush()
                    if pcm is not None:
                        return np.array(pcm, dtype=np.float32) / 32768.0
                except Exception as e:
                    print(f"Failed to flush stream: {e}")
            return None
    
    def stop_stream(self):
        """Stop the current stream"""
        with self.stream_lock:
            if self.current_stream is not None:
                try:
                    self.current_stream.close()
                except Exception as e:
                    print(f"Failed to close stream: {e}")
                finally:
                    self.current_stream = None

    def cleanup(self):
        """Clean up resources"""
        self.stop_stream()
        self.orca.delete()

class VoiceChatBot:
    def __init__(self, moonshine_model="moonshine/base", ollama_model="llama2", pv_access_key=None):
        # Initialize components
        self.transcriber = Transcriber(model_name=moonshine_model, rate=SAMPLING_RATE)
        self.llm = OllamaLLM(model=ollama_model)
        self.tts = TTSManager("TG+GQUIL26lJLxolZPqmYYRhYZC26pxKB+HN6cxUVLf/bQn4OlfP/g==") 
        
        # Initialize VAD
        self.vad_model = load_silero_vad(onnx=True)
        self.vad_iterator = VADIterator(
            model=self.vad_model,
            sampling_rate=SAMPLING_RATE,
            threshold=0.5,
            min_silence_duration_ms=300,
        )
        
        # State management
        self.state = BotState.LISTENING
        self.should_stop = False
        
        # Audio processing
        self.audio_queue = Queue()
        self.speech_buffer = np.empty(0, dtype=np.float32)
        
        # Setup audio stream
        self.stream = InputStream(
            samplerate=SAMPLING_RATE,
            channels=1,
            blocksize=CHUNK_SIZE,
            dtype=np.float32,
            callback=self._audio_callback
        )

    def _set_state(self, new_state: BotState):
        old_state = self.state
        self.state = new_state
        print(f"\nState: {old_state.value} -> {new_state.value}")

    def _audio_callback(self, data, frames, time, status):
        if status:
            print(f"Status: {status}")
        self.audio_queue.put((data.copy().flatten(), status))

    def _soft_reset_vad(self):
        self.vad_iterator.triggered = False
        self.vad_iterator.temp_end = 0
        self.vad_iterator.current_sample = 0

    def _process_speech(self) -> Optional[str]:
        if len(self.speech_buffer) > 0:
            text = self.transcriber(self.speech_buffer)
            self.speech_buffer = np.empty(0, dtype=np.float32)
            return text
        return None

    def _handle_llm_response(self, text: str):
        print("\nAI: ", end="", flush=True)
        
        try:
            if self.tts:
                self.tts.start_stream()
            
            current_chunk = ""
            for chunk in self.llm.stream(text):
                print(chunk, end="", flush=True)
                current_chunk += chunk
                       
            print("\n")
        except Exception as e:
            print(f"\nError in LLM response: {e}")
        finally:
            if self.tts:
                self.tts.stop_stream()
            self._set_state(BotState.LISTENING)

    def _process_audio_loop(self):
        lookback_size = LOOKBACK_CHUNKS * CHUNK_SIZE

        while not self.should_stop:
            try:
                chunk, status = self.audio_queue.get(timeout=1.0)
                
                # Update speech buffer
                self.speech_buffer = np.concatenate((self.speech_buffer, chunk))
                if self.state != BotState.SPEAKING:
                    self.speech_buffer = self.speech_buffer[-lookback_size:]

                # Process VAD
                speech_dict = self.vad_iterator(chunk)
                
                if speech_dict:
                    if "start" in speech_dict and self.state == BotState.LISTENING:
                        self._set_state(BotState.SPEAKING)
                        
                    if "end" in speech_dict and self.state == BotState.SPEAKING:
                        self._set_state(BotState.PROCESSING)
                        text = self._process_speech()
                        if text:
                            print(f"\nYou said: {text}")
                            self._handle_llm_response(text)

                elif self.state == BotState.SPEAKING:
                    if (len(self.speech_buffer) / SAMPLING_RATE) > MAX_SPEECH_SECS:
                        self._set_state(BotState.PROCESSING)
                        text = self._process_speech()
                        if text:
                            print(f"\nYou said: {text}")
                            self._handle_llm_response(text)
                        self._soft_reset_vad()

            except Queue.Empty:
                continue
            except Exception as e:
                print(f"\nError in audio processing: {e}")

    def start(self):
        print("\nStarting Voice ChatBot. Press Ctrl+C to exit.")
        print("Current state: LISTENING")
        
        def interrupt_handler(signum, frame):
            self.should_stop = True
            
        signal.signal(signal.SIGINT, interrupt_handler)
        
        self.stream.start()
        process_thread = threading.Thread(target=self._process_audio_loop)
        process_thread.start()
        
        try:
            process_thread.join()
        finally:
            self.stream.stop()
            self.stream.close()
            if self.tts:
                self.tts.cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Voice-controlled Chatbot")
    parser.add_argument(
        "--moonshine-model",
        default="moonshine/base",
        choices=["moonshine/base", "moonshine/tiny"],
        help="Moonshine model to use for speech recognition"
    )
    parser.add_argument(
        "--ollama-model",
        default="llama3.2:1b",
        help="Ollama model to use for chat responses"
    )
    parser.add_argument(
        "--pv-access-key",
        help="Picovoice access key for text-to-speech",
        default=None
    )
    
    args = parser.parse_args()
    
    chatbot = VoiceChatBot(
        moonshine_model=args.moonshine_model,
        ollama_model=args.ollama_model,
        pv_access_key=args.pv_access_key
    )
    chatbot.start()
