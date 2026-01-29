import asyncio
import time
import base64
import io
import string
import random
import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import matplotlib.pyplot as plt
from elevenlabs import ElevenLabs

import os, asyncio
import time
import logging
import traceback
from elevenlabs import ElevenLabs, VoiceSettings
import re  # Add this import for regex
# aiohttp
import aiohttp
from typing import AsyncGenerator
import functools

def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.4f} seconds")
        return result
    return wrapper

class ElevenLabsTTS:
    """
    A module for interacting with ElevenLabs API to generate text-to-speech audio streams. 
    """
    def __init__(self, api_key: str, voice_id: str):
        """
        Initialize the ElevenLabs client with the given API key.
        """
        if not api_key:
            raise ValueError("API key for ElevenLabs is required.")
        self.client = ElevenLabs(api_key=api_key)
        self.voice_id = voice_id
        self.session = None

    async def ensure_session(self):
        """Ensure an aiohttp session exists or create a new one"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session

    @timer
    def split_text_into_sentences(self, text: str) -> list[str]:
        """
        Split the input text into sentences using regex.
        :param text: The text to split.
        :return: A list of sentences.
        """
        return re.split(r'(?<=[.!?]) +', text)

    def _is_text_meaningful(self, text: str) -> bool:
        """
        Check if the text contains meaningful content worth synthesizing.
        Returns False for empty, whitespace-only, or punctuation-only text.
        """
        if not text or not text.strip():
            return False
            
        # Check if text contains only quote characters, punctuation, or whitespace
        import string
        non_meaningful_chars = string.punctuation + string.whitespace
        return not all(c in non_meaningful_chars for c in text)
    
    @timer
    def text_to_speech_stream(self, text: str, voice_id=None, settings=None):
        """
        Generate a text-to-speech audio stream. 
        """
        voice_id = voice_id or self.voice_id
        
        # Check if the text is meaningful before sending to API
        if not self._is_text_meaningful(text):
            logging.info(f"Skipping TTS generation for non-meaningful text: {repr(text)}")
            return []
            
        logging.info(f"Generating TTS stream for text (length: {len(text)})")
        try:
            settings = settings or VoiceSettings(
                stability=0.4,      # Slightly increased for telephony
                similarity_boost=0.75,  # Balanced for telephony
                style=0.45,         # Slightly reduced for clarity
                use_speaker_boost=True
            )

            audio_chunks = self.client.generate(         # generator object
                text=text,
                voice=voice_id,
                # voice_settings=settings,
                model="eleven_multilingual_v2",
                optimize_streaming_latency=4,  # Balanced setting for telephony (0-4)
                stream=True,
                output_format="ulaw_8000",  # Best format for Twilio (8kHz µ-law)
            )
                
            logging.debug(f"Generated audio data.")
            return audio_chunks
        except Exception as e:
            traceback.print_exc()
            logging.error(f"Error generating TTS stream: {str(e)}")
            raise

    async def text_to_speech_stream_async(self, text: str, voice_id=None, settings=None) -> AsyncGenerator[bytes, None]:
        """
        Async version that doesn't block the event loop
        """
        start_time = time.time()
        voice_id = voice_id or self.voice_id
        
        if not self._is_text_meaningful(text):
            logging.info(f"Skipping async TTS generation for non-meaningful text: {repr(text)}")
            return
            
        logging.info(f"Generating async TTS stream for text (length: {len(text)})")
        
        try:
            # Run the blocking TTS call in a thread pool to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            audio_chunks = await loop.run_in_executor(
                None, 
                self.text_to_speech_stream, 
                text, voice_id, settings
            )
            
            # Process chunks asynchronously
            for chunk in audio_chunks:
                await asyncio.sleep(0)  # Yield control back to event loop
                if chunk:
                    yield chunk
            
            end_time = time.time()
            logging.info(f"Async TTS stream completed in {end_time - start_time:.2f} seconds")
                    
        except Exception as e:
            traceback.print_exc()
            logging.error(f"Error generating async TTS stream: {str(e)}")
            raise
    
    @timer
    def text_to_speech_stream_for_voice_test(self, text: str, voice_id="4fI6S5BDphoC8rLuBs42", settings=None):
        """
        Generate a text-to-speech audio stream. 
        """
        # Check if the text is meaningful before sending to API
        if not self._is_text_meaningful(text):
            logging.info(f"Skipping voice test TTS generation for non-meaningful text: {repr(text)}")
            return []
            
        logging.info("Generating text-to-speech stream.")
        try:
            settings = settings or VoiceSettings(
                stability=0.3,  # More natural variation
                similarity_boost=0.8,  # Consistent voice tone
                style=0.5,  # Expressiveness
                use_speaker_boost=True
            )

            audio_chunks = self.client.generate(         # generator object
                text=text,
                voice=voice_id or self.voice_id,
                voice_settings=settings,
                model="eleven_multilingual_v2",
                optimize_streaming_latency=0,
                stream=True,
                output_format="mp3_44100_128",  # mp3_22050_32', 'mp3_44100_32', 'mp3_44100_64', 'mp3_44100_96', 'mp3_44100_128', 'mp3_44100_192', 'pcm_16000', 'pcm_22050', 'pcm_24000', 'pcm_44100', 'ulaw_8000']
            )
                
            logging.debug(f"Generated audio data.")
        except Exception as e:
            traceback.print_exc()
            logging.error(f"Error generating TTS stream: {str(e)}")
            raise

        return audio_chunks  # adjust import if needed

# === CONFIG ===
ELEVENLABS_API_KEY = "sk_9c7875201156a4eead6d5dccdfabbd5235281f4c3d98295b"
VOICE_ID = "4fI6S5BDphoC8rLuBs42"
TEST_COUNT = 4

# === FastAPI app ===
app = FastAPI()
tts_instances = [ElevenLabsTTS(ELEVENLABS_API_KEY, VOICE_ID) for _ in range(TEST_COUNT)]
results = []

random_inputs = [
    "Hello, I'm Alexander from Volume?",
    "Do you want to book an appointment?",
    "That's Funny!",
    "Volume provides services for Lead Generation?",
    "Let's book an appointment",
    "Sorry, the slot is not available for booking appointment",
    "The call has been ended.",
    "Thanks for asking about Volume.",
    "Let's come to the point back",
    "Volume provides cold call services"
    ]

@app.get("/run")
async def run_test():
    global results
    results = []
    tasks = []
    for idx in range(TEST_COUNT):
        tasks.append(send_tts_request(idx))
    await asyncio.gather(*tasks)
    return {"done": True, "results_count": len(results)}

async def send_tts_request(idx):
    tts = tts_instances[idx]
    text = random_inputs[(idx%10)]

    start_time = time.perf_counter()
    print(f"Request {idx}: SENDING at {start_time:.4f}")
    
    stream_start_time = None
    end_time = None
    chunk_count = 0
    chunk_size_total = 0

    try:
        async for chunk in tts.text_to_speech_stream_async(text):
            if chunk:
                chunk_count += 1
                chunk_size_total += len(chunk)
                if stream_start_time is None:
                    stream_start_time = time.perf_counter()
                    print(f"Request {idx}: First Chunk arrived at {stream_start_time:.4f}")
    except Exception as e:
        status = f"ERROR: {str(e)}"
    else:
        status = "200 OK"

    end_time = time.perf_counter()
    stream_start_latency = (stream_start_time - start_time) if stream_start_time else None
    total_latency = end_time - start_time

    results.append({
        "id": idx,
        "input": text,
        "chunks": chunk_count,
        "total_bytes": chunk_size_total,
        "start_time": f"{start_time:.4f}",
        "stream_start_time": f"{stream_start_time:.4f}" if stream_start_time else "N/A",
        "stream_start_latency": f"{stream_start_latency:.4f}" if stream_start_latency else "N/A",
        "end_time": f"{end_time:.4f}",
        "total_latency": f"{total_latency:.4f}",
        "status": status
    })

    print(f"Request {idx}: COMPLETED at {end_time:.4f} (took {end_time - start_time:.2f}s)")

@app.get("/", response_class=HTMLResponse)
async def view_results():
    global results
    results = []
    await run_test()
    # print(results)

    results = sorted(results, key=lambda x: int(x["id"]))

    # print("\n\n", results)

    stream_latencies = [float(r['stream_start_latency']) if r['stream_start_latency'] != "N/A" else 5.0 for r in results]
    total_latencies = [float(r['total_latency']) for r in results]

    html = """
    <html><head><title>ElevenLabs TTS Test</title></head><body>
    <h1>ElevenLabs TTS Streaming - {} Requests</h1>
    <table border="1" cellpadding="5">
        <tr>
            <th>ID</th><th>Input</th><th>Chunks</th><th>Total Bytes</th>
            <th>Start Time</th><th>Stream Start</th><th>Stream Latency</th>
            <th>End Time</th><th>Total Latency</th><th>Status</th>
        </tr>
    """.format(TEST_COUNT)

    for r in sorted(results, key=lambda x: x['id']):
        html += f"""
        <tr>
            <td>{r['id']}</td>
            <td>{r['input']}</td>
            <td>{r['chunks']}</td>
            <td>{r['total_bytes']}</td>
            <td>{r['start_time']}</td>
            <td>{r['stream_start_time']}</td>
            <td>{r['stream_start_latency']}</td>
            <td>{r['end_time']}</td>
            <td>{r['total_latency']}</td>
            <td>{r['status']}</td>
        </tr>
        """

    html += "</table>"

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    req_ids = range(len(total_latencies))

    ax1.plot(req_ids, total_latencies, 'b-o', alpha=0.7, label="Total Latency")
    ax1.plot(req_ids, stream_latencies, 'r-s', alpha=0.7, label="First Chunk Latency")
    ax1.set_title("Latency Trends")
    ax1.set_xlabel("Request ID")
    ax1.set_ylabel("Seconds")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.hist(total_latencies, bins=15, alpha=0.7, label="Total")
    ax2.hist(stream_latencies, bins=15, alpha=0.7, label="Stream Start")
    ax2.set_title("Latency Distribution")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3.boxplot([total_latencies, stream_latencies], labels=['Total', 'Stream Start'])
    ax3.set_title("Boxplot Comparison")
    ax3.grid(True, alpha=0.3)

    metrics = [
        ['Metric', 'Total', 'Stream Start'],
        ['Mean', f"{np.mean(total_latencies):.3f}", f"{np.mean(stream_latencies):.3f}"],
        ['Median', f"{np.median(total_latencies):.3f}", f"{np.median(stream_latencies):.3f}"],
        ['Std Dev', f"{np.std(total_latencies):.3f}", f"{np.std(stream_latencies):.3f}"],
        ['Min', f"{np.min(total_latencies):.3f}", f"{np.min(stream_latencies):.3f}"],
        ['Max', f"{np.max(total_latencies):.3f}", f"{np.max(stream_latencies):.3f}"],
    ]
    ax4.axis('off')
    table = ax4.table(cellText=metrics[1:], colLabels=metrics[0], cellLoc='center', loc='center')
    ax4.set_title("Summary Stats")

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)

    html += f"<h2>Latency Analysis</h2><img src='data:image/png;base64,{img_b64}'/>"
    html += "</body></html>"

    return HTMLResponse(content=html)

if __name__ == "__main__":
    uvicorn.run("test11labs:app", host="127.0.0.1", port=8000, reload=True)
