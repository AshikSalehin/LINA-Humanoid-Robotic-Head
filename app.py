import os
import base64
import logging
import json
import time
import threading
import requests
from deepgram import LiveTranscriptionEvents
from flask import Flask, request, jsonify, render_template
from flask_sock import Sock
from elevenlabs import ElevenLabs
from deepgram import DeepgramClient, LiveOptions, PrerecordedOptions
from flask_cors import CORS
import uuid
from threading import Event
import asyncio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Output to console
        logging.FileHandler('app.log')  # Also save to file
    ]
)

# Add a test log message to verify logging is working
logging.info("Application started - logging system initialized")

app = Flask(__name__, static_folder='static')
sock = Sock(app)
CORS(app)

# Configuration
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY", "csk-y6fwrtkpvp6fwp9yr8nv48d8x349pjr6hnwvv8kr854jtjkf")
CEREBRAS_BASE_URL = "https://api.cerebras.ai/v1/"
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "your-elevenlabs-key")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "4fI6S5BDphoC8rLuBs42")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "your-deepgram-key")

# Store active WebSocket connections
active_ws_connections = {}

# System prompt (unchanged)
SYSTEM_PROMPT = f"""
<identity>
You are "LINA," the university's front desk information assistant. If asked if you're AI, confirm professionally and return to assisting visitors.
</identity>

<primary_objective>
Provide accurate building navigation and staff location information to university visitors.
</primary_objective>

<capabilities>
- Give building locations and room numbers
- Provide directions to university facilities
- Share staff office locations and departments
- Explain campus navigation using public maps
</capabilities>

<limitations>
- Cannot access private schedules or student records
- Cannot make appointments or deliver messages
- Cannot retrieve internal system information
</limitations>

<script>
1. "Welcome to the University of Frontier Technology, Bangladesh! How may I help you today?"
2. [Listen for staff name, department, or location request]
3. Provide specific directions: "Dr. [Name]'s office is in [Building], Room [Number]. Take [Elevator/Stairs] to floor [X]."
4. "Would you like directions to another location?"
</script>

<demo_navigation>
Building: Academic Tower (5 floors, each with 5 rooms)
Floor 1: Rooms 101-105
- 101: Admissions Office
- 102: Registrar
- 103: Financial Aid
- 104: Campus Security
- 105: Lost & Found

Floor 2: Rooms 201-205
- 201: Dr. Sarah Chen (Mathematics)
- 202: Prof. Michael Torres (Physics)
- 203: Dr. Aisha Patel (Chemistry)
- 204: Prof. James Wilson (Biology)
- 205: Department Lounge

Floor 3: Rooms 301-305
- 301: Dr. Emily Roberts (Literature)
- 302: Prof. David Kim (History)
- 303: Dr. Maria Garcia (Philosophy)
- 304: Prof. Robert Johnson (Psychology)
- 305: Conference Room

Floor 4: Rooms 401-405
- 401: Dr. Lisa Anderson (Computer Science)
- 402: Prof. Thomas Brown (Engineering)
- 403: Dr. Jennifer Lee (Statistics)
- 404: Prof. William Davis (Robotics)
- 405: Research Lab

Floor 5: Rooms 501-505
- 501: Dean's Office
- 502: Provost Office
- 503: Faculty Senate
- 504: Board Room
- 505: University Archives
</demo_navigation>

<rules>
- Use only public information
- Keep responses under 3 sentences
- If unsure: "Please visit the physical information desk at [Location]"
- Never speculate about staff availability
- Clearly pronounce building names (e.g., "Academic Tower")
- For staff requests: "Professor [Name] is in [Building], Room [Number]. Take [Elevator/Stairs] to floor [X]."
</rules>
"""
  # Keep your existing system prompt here

class CerebrasLLM:
    def __init__(self, api_key, SYSTEM_PROMPT):
        self.api_key = api_key
        self.SYSTEM_PROMPT = SYSTEM_PROMPT
        self.base_url = CEREBRAS_BASE_URL
        self.conversation_history = []
        self.reset_conversation()
    
    def reset_conversation(self):
        self.conversation_history = [
            {"role": "system", "content": self.SYSTEM_PROMPT}
        ]
    
    def get_headers(self):
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
    
    def get_available_models(self):
        try:
            response = requests.get(f"{self.base_url}models", 
                                   headers=self.get_headers(),
                                   timeout=10)
            return [model["id"] for model in response.json().get("data", [])]
        except Exception as e:
            logging.error(f"Model fetch error: {str(e)}")
            return []
    
    def stream_response(self, user_input):
        """Stream the LLM response in chunks to reduce latency"""
        self.conversation_history.append({"role": "user", "content": user_input})
        
        try:
            models = self.get_available_models()
            if not models:
                yield "Error: No models available"
                return
            
            selected_model = models[1] if len(models) > 1 else models[0]
            
            response = requests.post(
                f"{self.base_url}chat/completions",
                headers=self.get_headers(),
                json={
                    "model": selected_model,
                    "messages": self.conversation_history,
                    "temperature": 0.7,
                    "max_tokens": 40960,
                    "stream": True
                },
                stream=True,
                timeout=30
            )
            
            full_response = ""
            for chunk in response.iter_lines():
                if chunk:
                    chunk = chunk.decode().strip()
                    if chunk.startswith("data: "):
                        chunk = chunk[6:]
                    if chunk and chunk != "[DONE]":
                        try:
                            data = json.loads(chunk)
                            content = data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                            if content:
                                full_response += content
                                yield content
                        except json.JSONDecodeError:
                            continue
            
            self.conversation_history.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            logging.error(f"LLM Error: {str(e)}")
            yield "I'm having trouble responding. Please try again."

class ElevenLabsTTS:
    MODEL = "eleven_flash_v2_5"
    LATENCY_OPTIMIZATION = 4
    OUTPUT_FORMAT = "mp3_22050_32"
    
    def __init__(self, api_key, voice_id):
        self.client = ElevenLabs(api_key=api_key)
        self.voice_id = voice_id
    
    def text_to_speech(self, text):
        try:
            audio = self.client.generate(
                text=text,
                voice=self.voice_id,
                model=self.MODEL,
                optimize_streaming_latency=self.LATENCY_OPTIMIZATION,
                output_format=self.OUTPUT_FORMAT
            )
            return audio if isinstance(audio, bytes) else b''.join(audio)
        except Exception as e:
            logging.error(f"TTS Error: {str(e)}")
            return None

class DeepgramStreamingSTT:
    def __init__(self, api_key):
        self.api_key = api_key
        self.client = DeepgramClient(api_key)
        self.active_connections = {}

    # Add proper send_audio method
    def send_audio(self, session_id, audio_data):
        if session_id in self.active_connections:
            try:
                self.active_connections[session_id].send(audio_data)
                return True
            except Exception as e:
                logging.error(f"Failed to send audio for session {session_id}: {str(e)}")
                return False
        else:
            logging.warning(f"Cannot send audio - no active connection for session {session_id}")
            return False
    
    def close(self, session_id):
        if session_id in self.active_connections:
            try:
                self.active_connections[session_id].finish()
                del self.active_connections[session_id]
                return True
            except Exception as e:
                logging.error(f"Failed to close connection: {str(e)}")
                return False
        return False
    
    def create_streaming_client(self, session_id, callback):
        """Create a streaming client for the given session ID"""
        try:
            options = LiveOptions(
                model="nova-2",
                language="en-US",
                # encoding="opus",  # Add this line
                sample_rate=16000,  # Match typical WebM/Opus sample rate
                smart_format=True,
                interim_results=True,
                punctuate=True,
                endpointing=300,
                utterance_end_ms=1000
            )
            
            # Create websocket connection
            live_transcription = self.client.listen.websocket.v("1")
            
            # Define handlers using the correct event types
            def on_message(_, result):
                try:
                    if result.is_final:
                        transcript = result.channel.alternatives[0].transcript
                        logging.info(f"Final Transcript for {session_id}: {transcript}")
                        if transcript.strip():
                            callback(transcript, True)
                    else:
                        transcript = result.channel.alternatives[0].transcript
                        if transcript.strip():
                            callback(transcript, False)
                except Exception as e:
                    logging.error(f"Error in Deepgram on_message: {str(e)}")
            
            def on_error(_, error):
                logging.error(f"Deepgram error for {session_id}: {error}")
                
            def on_close(_, close_event):
                logging.info(f"Deepgram connection closed for {session_id} with event: {close_event}")
                if session_id in self.active_connections:
                    del self.active_connections[session_id]

            # Attach handlers using the correct event enum
            live_transcription.on(LiveTranscriptionEvents.Transcript, on_message)
            live_transcription.on(LiveTranscriptionEvents.Error, on_error)
            live_transcription.on(LiveTranscriptionEvents.Close, on_close)
            
            # Start connection with options
            live_transcription.start(options)
            logging.info(f"Started Deepgram connection for {session_id}")
            
            # Store the connection
            self.active_connections[session_id] = live_transcription
            return live_transcription
            
        except Exception as e:
            logging.error(f"Failed to create streaming client for {session_id}: {str(e)}")
            return None

session_llms = {}

task_events = {}

tts = ElevenLabsTTS(ELEVENLABS_API_KEY, ELEVENLABS_VOICE_ID)

# Add global STT instance
stt = DeepgramStreamingSTT(DEEPGRAM_API_KEY)

def process_ai_response(session_id, transcript):
    try:
        # Cancel any ongoing task for this session
        if session_id in task_events:
            task_events[session_id].set()  # Signal cancellation
            task_events[session_id] = Event()  # Create new event for new task
        else:
            task_events[session_id] = Event()

        # Use the session-specific LLM
        llm = session_llms.get(session_id)
        if not llm:
            llm = CerebrasLLM(CEREBRAS_API_KEY, SYSTEM_PROMPT)
            session_llms[session_id] = llm

        # Start streaming LLM response and process it in real-time
        threading.Thread(
            target=stream_and_process_response,
            args=(session_id, transcript, task_events[session_id], llm)  # Pass llm
        ).start()
        
    except Exception as e:
        logging.error(f"AI response error: {str(e)}")
        send_error_to_client(session_id, str(e))

async def stream_and_process_response(session_id, transcript, cancel_event, llm):
    try:
        buffer = ""
        chunk_index = 0

        async for token in llm.stream_response(transcript):
            if cancel_event.is_set():
                return

            buffer += token
            # Send every 8 words or on punctuation
            if len(buffer.split()) >= 2 or any(p in token for p in ".!?"):
                text_chunk = buffer.strip()
                buffer = ""
                if text_chunk:
                    asyncio.create_task(process_audio_chunk(session_id, text_chunk, chunk_index, cancel_event))
                    chunk_index += 1

        # Send any remaining buffer
        if buffer:
            asyncio.create_task(process_audio_chunk(session_id, buffer.strip(), chunk_index, cancel_event))

    except Exception as e:
        logging.error(f"Streaming response error: {str(e)}")
        if not cancel_event.is_set():
            await send_error_to_client(session_id, str(e))

def process_audio_chunk(session_id, text, chunk_index, cancel_event):
    """Process a single audio chunk and send to client"""
    try:
        if not text.strip() or cancel_event.is_set():
            return
            
        # Generate audio for text
        audio_data = tts.text_to_speech(text)
        if not audio_data or cancel_event.is_set():
            return
            
        # Send audio chunk to client immediately without normalization
        if session_id in active_ws_connections:
            try:
                ws = active_ws_connections[session_id]
                chunk_response = {
                    "type": "assistant_audio_chunk",
                    "data": {
                        "audio": base64.b64encode(audio_data).decode('utf-8'),
                        "chunk_index": chunk_index,
                        "total_chunks": 1,
                        "is_last": True if chunk_index == 0 else False
                    }
                }
                ws.send(json.dumps(chunk_response))
                logging.info(f"Sent audio chunk {chunk_index} with text: {text}")
            except Exception as e:
                logging.error(f"Failed to send audio chunk {chunk_index}: {str(e)}")
    except Exception as e:
        logging.error(f"Error processing audio chunk: {str(e)}")

# Add helper function for error handling
def send_error_to_client(session_id, error_message):
    if session_id in active_ws_connections:
        try:
            ws = active_ws_connections[session_id]
            error_response = {
                "type": "error",
                "data": {"message": error_message}
            }
            ws.send(json.dumps(error_response))
        except Exception as e:
            logging.error(f"Failed to send error message: {str(e)}")

@sock.route('/ws')
def websocket_endpoint(ws):
    session_id = str(uuid.uuid4())
    active_ws_connections[session_id] = ws
    
    # Create a new LLM for this session
    session_llms[session_id] = CerebrasLLM(CEREBRAS_API_KEY, SYSTEM_PROMPT)

    logging.info(f"Client connected: {session_id}")
    
    # Send connected message
    try:
        ws.send(json.dumps({
            "type": "status",
            "data": {"message": "Connected to server"}
        }))
        
        # Define the transcription callback for this session
        def transcription_callback(transcript, is_final):
            try:
                if session_id in active_ws_connections:
                    try:
                        ws.send(json.dumps({
                            "type": "transcript",
                            "data": {
                                "text": transcript,
                                "is_final": is_final
                            }
                        }))
                        
                        if is_final and transcript.strip():
                            logging.info(f"Processing final transcript: {transcript}")
                            process_ai_response(session_id, transcript)
                    except Exception as e:
                        logging.error(f"Failed to send transcript to client: {str(e)}")
                else:
                    logging.warning(f"Cannot send transcript - session not found: {session_id}")
                    
            except Exception as e:
                logging.error(f"Error in transcription callback: {str(e)}")
        
        # Process incoming messages
        while True:
            try:
                # Handle incoming messages
                message = ws.receive()
                if message is None:
                    logging.info(f"Client {session_id} disconnected (receive returned None)")
                    break
                    
                logging.debug(f"Received message from {session_id}, length: {len(message)}")
                
                data = json.loads(message)
                message_type = data.get('type')
                logging.debug(f"Message type: {message_type}")
                
                if message_type == 'initial_message':
                    logging.info(f"Received initial message from client {session_id}: {data.get('data', {}).get('message')}")
                    platform = data.get('data', {}).get('message')
                    if platform == "Bubble":
                        new_SYSTEM_PROMPT = data.get('data', {}).get('prompt')
                        session_llms[session_id] = CerebrasLLM(CEREBRAS_API_KEY, new_SYSTEM_PROMPT)
                    else:
                        session_llms[session_id] = CerebrasLLM(CEREBRAS_API_KEY, SYSTEM_PROMPT)

                    ws.send(json.dumps({
                        "type": "status",
                        "data": {"message": "Initial message received"}
                    }))
                
                elif message_type == 'start_stream':
                    logging.info(f"Starting stream for {session_id}")
                    
                    connection = stt.create_streaming_client(session_id, transcription_callback)
                    
                    if connection:
                        ws.send(json.dumps({
                            "type": "status",
                            "data": {"message": "Streaming started"}
                        }))
                        logging.info(f"Stream started for {session_id}")
                        
                        # Create initial greeting
                        if session_id in task_events:
                            task_events[session_id].set()  # Cancel any ongoing task
                        
                        task_events[session_id] = Event()
                        
                        # Process initial greeting using streaming approach
                        greeting_text = "Hello There"
                        threading.Thread(
                            target=process_audio_chunk,
                            args=(session_id, greeting_text, 0, task_events[session_id])
                        ).start()
                    else:
                        ws.send(json.dumps({
                            "type": "error",
                            "data": {"message": "Failed to start streaming"}
                        }))
                
                elif message_type == 'audio_data':
                    try:
                        audio_data = base64.b64decode(data.get('data', ''))
                        if session_id not in stt.active_connections:
                            # Stream not started yet, send a more helpful error
                            ws.send(json.dumps({
                                "type": "error",
                                "data": {"message": "Stream not initialized - send start_stream first"}
                            }))
                            logging.warning(f"Received audio data before stream initialization for {session_id}")
                        elif not stt.send_audio(session_id, audio_data):
                            ws.send(json.dumps({
                                "type": "error",
                                "data": {"message": "Failed to process audio"}
                            }))
                    except Exception as e:
                        logging.error(f"Error processing audio data: {str(e)}")
                        ws.send(json.dumps({
                            "type": "error",
                            "data": {"message": "Invalid audio format"}
                        }))
                
                elif message_type == 'stop_stream':
                    logging.info(f"Stopping stream for {session_id}")
                    if stt.close(session_id):
                        ws.send(json.dumps({
                            "type": "status",
                            "data": {"message": "Streaming stopped"}
                        }))
                    else:
                        ws.send(json.dumps({
                            "type": "error",
                            "data": {"message": "Failed to stop streaming"}
                        }))
                
                elif message_type == 'reset':
                    logging.info(f"Resetting conversation for {session_id}")
                    if session_id in session_llms:
                        session_llms[session_id].reset_conversation()
                    ws.send(json.dumps({
                        "type": "status",
                        "data": {"message": "Conversation reset"}
                    }))
                else:
                    logging.warning(f"Unknown message type: {message_type}")
                    
            except json.JSONDecodeError as e:
                logging.error(f"Invalid JSON received from {session_id}: {str(e)}")
                ws.send(json.dumps({
                    "type": "error",
                    "data": {"message": "Invalid JSON format"}
                }))
                
            except Exception as e:
                logging.error(f"Error processing message: {str(e)}")
                try:
                    ws.send(json.dumps({
                        "type": "error",
                        "data": {"message": f"Server error: {str(e)}"}
                    }))
                except:
                    logging.error("Failed to send error message - websocket likely closed")
                break
                
    except Exception as e:
        logging.error(f"WebSocket error: {str(e)}")
    finally:
        if session_id in active_ws_connections:
            del active_ws_connections[session_id]
        if session_id in session_llms:
            del session_llms[session_id]
        stt.close(session_id)
        logging.info(f"Cleaned up session {session_id}")

# Regular Flask routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/reset', methods=['POST'])
def handle_reset():
    session_id = request.json.get('session_id')
    if session_id and session_id in session_llms:
        session_llms[session_id].reset_conversation()
        return jsonify({"status": "Conversation reset"})
    return jsonify({"status": "Session not found"}), 404

if __name__ == '__main__':
    try:
        # Run on localhost without SSL
        print("Running on http://localhost:8116")
        app.run(host='127.0.0.1', port=5000, debug=True)
    except Exception as e:
        print(f"Error starting server: {e}")
