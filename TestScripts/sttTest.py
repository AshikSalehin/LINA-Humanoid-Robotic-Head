import asyncio
import os
import time
import traceback
import io
import base64
import numpy as np
import matplotlib.pyplot as plt
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from dotenv import load_dotenv
from pydub import AudioSegment
import uvicorn
import aiohttp
import json

load_dotenv()

API_KEY = os.getenv("DEEPGRAM_API_KEY")
MODEL = "nova-2"
LANGUAGE = "en-US"
SAMPLE_RATE = 8000
ENCODING = "linear16"
TEST_COUNT = 5  # Start with 5, increase later
MP3_FILE_PATH = "conv.mp3"  # Set your test file

results = []

# Use lifespan for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print(f"🚀 Deepgram STT Tester Starting...")
    print(f"🔧 Testing {TEST_COUNT} parallel requests")
    print(f"🎯 Model: {MODEL}, Language: {LANGUAGE}")
    print(f"🎵 Audio file: {MP3_FILE_PATH}")
    print(f"🔑 API Key: {'✅ Set' if API_KEY else '❌ NOT SET'}")
    
    if not API_KEY:
        print("\n⚠️  WARNING: API key not found!")
        print("   Create a .env file with: DEEPGRAM_API_KEY=your_key_here")
    
    print("\n🌐 Available endpoints:")
    print("  /           - Run full test and view results")
    print("  /single-test - Test single request")
    print("  /health     - Health check")
    print("  /test-audio - Test audio file loading")
    print("\n⏸️  Press Ctrl+C to stop")
    
    yield
    # Shutdown
    print("👋 Shutting down...")

app = FastAPI(lifespan=lifespan)

async def transcribe_audio_direct(audio_bytes, idx=0):
    """Direct HTTP API call to Deepgram"""
    start = time.perf_counter()
    
    try:
        headers = {
            'Authorization': f'Token {API_KEY}',
            'Content-Type': 'audio/wav'
        }
        
        params = {
            'model': MODEL,
            'language': LANGUAGE,
            'sample_rate': str(SAMPLE_RATE),
            'encoding': ENCODING,
            'punctuate': 'true',
            'smart_format': 'true',
            'utterances': 'true'
        }
        
        print(f"Client {idx}: Sending request to Deepgram...")
        
        timeout = aiohttp.ClientTimeout(total=30)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                'https://api.deepgram.com/v1/listen',
                headers=headers,
                data=audio_bytes,
                params=params
            ) as resp:
                
                first_result_time = time.perf_counter()
                
                if resp.status == 200:
                    response_data = await resp.json()
                    transcript = extract_transcript(response_data)
                    
                    if transcript:
                        status = "OK"
                    else:
                        transcript = "No transcript in response"
                        status = "ERROR"
                else:
                    transcript = f"HTTP Error {resp.status}: {await resp.text()}"
                    status = "ERROR"
    
    except asyncio.TimeoutError:
        transcript = "Timeout error"
        status = "ERROR"
        first_result_time = time.perf_counter()
    except Exception as e:
        transcript = f"Exception: {str(e)[:100]}"
        status = "ERROR"
        first_result_time = time.perf_counter()
    
    end = time.perf_counter()
    latency = first_result_time - start if 'first_result_time' in locals() else None
    total_time = end - start
    
    print(f"Client {idx}: Completed in {total_time:.3f}s - {status}")
    
    return {
        "id": idx,
        "words": transcript[:80] + "..." if len(transcript) > 80 else transcript,
        "start_time": f"{start:.4f}",
        "first_result_time": f"{first_result_time:.4f}" if latency else "N/A",
        "stream_start_latency": f"{latency:.4f}" if latency else "N/A",
        "end_time": f"{end:.4f}",
        "total_latency": f"{total_time:.4f}",
        "status": status
    }

def extract_transcript(response):
    """Extract transcript from Deepgram response"""
    if not response:
        return "Empty response"
    
    # Try to get transcript from response
    try:
        if isinstance(response, dict):
            if 'results' in response and 'channels' in response['results']:
                channels = response['results']['channels']
                if channels and len(channels) > 0:
                    channel = channels[0]
                    if 'alternatives' in channel and channel['alternatives'] and len(channel['alternatives']) > 0:
                        return channel['alternatives'][0].get('transcript', 'No transcript')
            elif 'transcript' in response:
                return response['transcript']
    except Exception as e:
        return f"Error extracting transcript: {str(e)[:50]}"
    
    return "Could not extract transcript"

@app.get("/")
async def index():
    global results
    results = []
    
    # Load and prepare audio
    try:
        print(f"\n=== Loading audio file: {MP3_FILE_PATH} ===")
        if not os.path.exists(MP3_FILE_PATH):
            error_msg = f"Audio file not found: {MP3_FILE_PATH}"
            print(f"❌ {error_msg}")
            return HTMLResponse(content=generate_error_html(error_msg))
        
        audio = AudioSegment.from_mp3(MP3_FILE_PATH)
        audio = audio.set_frame_rate(SAMPLE_RATE).set_channels(1)
        audio_10s = audio[:20 * 1000]  # 20 seconds
        
        # Export as WAV format
        buffer = io.BytesIO()
        audio_10s.export(buffer, format="wav")
        audio_data = buffer.getvalue()
        
        print(f"✅ Audio loaded successfully:")
        print(f"   Duration: {len(audio_10s)/1000:.1f}s")
        print(f"   Data size: {len(audio_data)} bytes")
        print(f"   Sample rate: {SAMPLE_RATE}Hz")
        
    except Exception as e:
        error_msg = f"Error loading audio: {str(e)}"
        print(f"❌ {error_msg}")
        return HTMLResponse(content=generate_error_html(error_msg))
    
    # Run tests in parallel
    print(f"\n=== Starting {TEST_COUNT} parallel STT tests ===")
    
    # Create tasks for all tests
    tasks = []
    for idx in range(TEST_COUNT):
        task = asyncio.create_task(transcribe_audio_direct(audio_data, idx))
        tasks.append(task)
    
    # Wait for all tasks to complete
    try:
        completed = await asyncio.gather(*tasks, return_exceptions=True)
    except Exception as e:
        print(f"Error in gather: {e}")
        completed = [Exception(f"Gather error: {e}")] * TEST_COUNT
    
    # Process results
    print(f"\n=== Processing Results ===")
    for idx, result in enumerate(completed):
        if isinstance(result, Exception):
            print(f"Task {idx} failed with exception: {result}")
            results.append({
                "id": idx,
                "words": f"ERROR: {str(result)[:80]}",
                "start_time": "0.0000",
                "first_result_time": "N/A",
                "stream_start_latency": "N/A",
                "end_time": "0.0000",
                "total_latency": "0.0000",
                "status": "ERROR"
            })
        else:
            results.append(result)
            status_icon = "✅" if result['status'] == 'OK' else "❌"
            print(f"Task {idx}: {status_icon} {result['status']} - {result['words'][:50]}...")
    
    # Sort and analyze results
    sorted_results = sorted(results, key=lambda x: x['id'])
    stream_latencies = []
    total_latencies = []
    
    successful = 0
    for r in sorted_results:
        if r['status'] == 'OK':
            successful += 1
            if r['stream_start_latency'] != "N/A":
                try:
                    stream_latencies.append(float(r['stream_start_latency']))
                except (ValueError, TypeError):
                    pass
            try:
                total_latencies.append(float(r['total_latency']))
            except (ValueError, TypeError):
                pass
    
    print(f"\n=== Summary ===")
    print(f"Total tests: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {len(results)-successful}")
    print(f"Success rate: {successful/len(results)*100:.1f}%")
    
    if successful > 0:
        print(f"Average latency: {np.mean(total_latencies):.3f}s")
        print(f"Min latency: {np.min(total_latencies):.3f}s")
        print(f"Max latency: {np.max(total_latencies):.3f}s")
    
    # Generate HTML report
    html = generate_html_report(sorted_results, stream_latencies, total_latencies, successful)
    return HTMLResponse(content=html)

def generate_error_html(error_msg):
    """Generate error HTML page"""
    return f"""
    <html>
    <head>
        <title>Error - Deepgram STT Test</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
            .error-box {{ background-color: #ffebee; padding: 30px; border-radius: 10px; border-left: 6px solid #f44336; }}
            .success-box {{ background-color: #e8f5e8; padding: 30px; border-radius: 10px; border-left: 6px solid #4caf50; }}
            .info-box {{ background-color: #e3f2fd; padding: 30px; border-radius: 10px; border-left: 6px solid #2196f3; }}
            pre {{ background: #f5f5f5; padding: 15px; border-radius: 5px; overflow-x: auto; }}
            a {{ color: #2196f3; text-decoration: none; }}
            a:hover {{ text-decoration: underline; }}
        </style>
    </head>
    <body>
        <h1>❌ Deepgram STT Test Error</h1>
        
        <div class="error-box">
            <h2>Error Details</h2>
            <p><strong>{error_msg}</strong></p>
        </div>
        
        <div class="info-box">
            <h2>🔧 Troubleshooting Steps</h2>
            <ol>
                <li><strong>Check .env file:</strong> Make sure it contains: <code>DEEPGRAM_API_KEY=your_key_here</code></li>
                <li><strong>Verify audio file:</strong> Make sure <code>{MP3_FILE_PATH}</code> exists in current directory</li>
                <li><strong>Check API key:</strong> Verify your Deepgram API key is valid and has credits</li>
                <li><strong>Check internet connection:</strong> Ensure you can reach api.deepgram.com</li>
            </ol>
            
            <h3>Quick Tests:</h3>
            <p>
                <a href="/health" style="background-color: #4CAF50; color: white; padding: 10px 15px; text-decoration: none; border-radius: 5px; margin-right: 10px;">
                    Health Check
                </a>
                <a href="/test-audio" style="background-color: #2196F3; color: white; padding: 10px 15px; text-decoration: none; border-radius: 5px; margin-right: 10px;">
                    Test Audio File
                </a>
                <a href="/single-test" style="background-color: #FF9800; color: white; padding: 10px 15px; text-decoration: none; border-radius: 5px;">
                    Single Test
                </a>
            </p>
        </div>
        
        <div style="margin-top: 30px; color: #666; font-size: 12px;">
            <p>Current directory: {os.getcwd()}</p>
            <p>Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
    </body>
    </html>
    """

def generate_html_report(sorted_results, stream_latencies, total_latencies, successful):
    """Generate HTML report with charts"""
    success_rate = successful/len(sorted_results)*100 if sorted_results else 0
    
    # Start building HTML
    html_parts = []
    
    # HTML Header
    html_parts.append(f"""
    <html>
    <head>
        <title>Deepgram STT Results</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px; margin-bottom: 30px; }}
            .card {{ background: white; padding: 25px; border-radius: 10px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); margin-bottom: 25px; }}
            .success {{ border-left: 6px solid #4CAF50; }}
            .error {{ border-left: 6px solid #f44336; }}
            .warning {{ border-left: 6px solid #ff9800; }}
            .info {{ border-left: 6px solid #2196F3; }}
            table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            th {{ background-color: #f8f9fa; font-weight: bold; }}
            tr:hover {{ background-color: #f5f5f5; }}
            .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px; margin: 20px 0; }}
            .metric {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); text-align: center; }}
            .metric-value {{ font-size: 32px; font-weight: bold; color: #333; margin-bottom: 5px; }}
            .metric-label {{ font-size: 14px; color: #666; }}
            .success-bg {{ background-color: #e8f5e8; }}
            .error-bg {{ background-color: #ffebee; }}
            .warning-bg {{ background-color: #fff3e0; }}
            pre {{ background: #f8f9fa; padding: 15px; border-radius: 5px; overflow-x: auto; font-family: 'Courier New', monospace; }}
            .btn {{ display: inline-block; padding: 10px 20px; background: #2196F3; color: white; text-decoration: none; border-radius: 5px; margin: 5px; }}
            .btn:hover {{ background: #0b7dda; }}
            .btn-success {{ background: #4CAF50; }}
            .btn-warning {{ background: #FF9800; }}
        </style>
    </head>
    <body>
    <div class="container">
        <div class="header">
            <h1>🎤 Deepgram Speech-to-Text Test Results</h1>
            <p>Parallel STT Performance Analysis | SDK v5.3.0 | Direct HTTP API</p>
        </div>
    """)
    
    # Metrics Section
    html_parts.append("""
        <div class="card info">
            <h2>📊 Performance Metrics</h2>
            <div class="metric-grid">
    """)
    
    # Add metrics
    metrics = [
        (f"{TEST_COUNT}", "Total Tests", ""),
        (f"{successful}", "Successful", "success-bg" if successful > 0 else "error-bg"),
        (f"{len(sorted_results)-successful}", "Failed", "error-bg" if len(sorted_results)-successful > 0 else ""),
        (f"{success_rate:.1f}%", "Success Rate", "success-bg" if success_rate > 50 else "warning-bg"),
    ]
    
    if successful > 0 and total_latencies:
        metrics.extend([
            (f"{np.mean(total_latencies):.3f}s", "Avg Latency", ""),
            (f"{np.min(total_latencies):.3f}s", "Min Latency", "success-bg"),
            (f"{np.max(total_latencies):.3f}s", "Max Latency", "warning-bg"),
        ])
    
    for value, label, css_class in metrics:
        html_parts.append(f"""
            <div class="metric {css_class}">
                <div class="metric-value">{value}</div>
                <div class="metric-label">{label}</div>
            </div>
        """)
    
    html_parts.append("""
            </div>
        </div>
    """)
    
    # Configuration Card
    html_parts.append(f"""
        <div class="card">
            <h2>⚙️ Test Configuration</h2>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px;">
                <div>
                    <strong>Model:</strong> {MODEL}<br>
                    <strong>Language:</strong> {LANGUAGE}<br>
                    <strong>Sample Rate:</strong> {SAMPLE_RATE}Hz
                </div>
                <div>
                    <strong>Encoding:</strong> {ENCODING}<br>
                    <strong>Audio File:</strong> {MP3_FILE_PATH}<br>
                    <strong>Test Time:</strong> {time.strftime('%Y-%m-%d %H:%M:%S')}
                </div>
            </div>
        </div>
    """)
    
    # Results Table
    html_parts.append("""
        <div class="card">
            <h2>📋 Detailed Results</h2>
            <table>
                <tr>
                    <th>ID</th>
                    <th>Status</th>
                    <th>Transcript Preview</th>
                    <th>First Latency</th>
                    <th>Total Latency</th>
                    <th>Response Time</th>
                </tr>
    """)
    
    for r in sorted_results:
        status_class = "success" if r['status'] == 'OK' else "error"
        status_icon = "✅" if r['status'] == 'OK' else "❌"
        latency_display = r['stream_start_latency'] if r['stream_start_latency'] != "N/A" else "N/A"
        
        html_parts.append(f"""
            <tr>
                <td><strong>{r['id']}</strong></td>
                <td>{status_icon} {r['status']}</td>
                <td title="{r['words']}">
                    <div style="max-width: 300px; overflow: hidden; text-overflow: ellipsis;">
                        {r['words']}
                    </div>
                </td>
                <td>{latency_display}</td>
                <td>{r['total_latency']}s</td>
                <td>{r['end_time']}</td>
            </tr>
        """)
    
    html_parts.append("""
            </table>
        </div>
    """)
    
    # Charts if we have successful results
    if successful > 1 and total_latencies:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        ax1, ax2, ax3, ax4 = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]
        
        # Plot 1: Latency Trends
        successful_ids = [i for i, r in enumerate(sorted_results) if r['status'] == 'OK']
        ax1.plot(successful_ids, total_latencies, 'b-o', alpha=0.7, label="Total Latency", markersize=8)
        if stream_latencies:
            ax1.plot(successful_ids[:len(stream_latencies)], stream_latencies, 'r-s', alpha=0.7, label="First Result", markersize=8)
        ax1.set_title("Latency Trends", fontsize=14, fontweight='bold')
        ax1.set_xlabel("Request ID")
        ax1.set_ylabel("Seconds")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Histogram
        ax2.hist(total_latencies, bins=min(10, len(total_latencies)), alpha=0.6, label="Total", color='blue', edgecolor='black')
        if stream_latencies:
            ax2.hist(stream_latencies, bins=min(10, len(stream_latencies)), alpha=0.6, label="First Result", color='red', edgecolor='black')
        ax2.set_title("Latency Distribution", fontsize=14, fontweight='bold')
        ax2.set_xlabel("Seconds")
        ax2.set_ylabel("Frequency")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Boxplot
        box_data = [total_latencies]
        labels = ['Total']
        if stream_latencies:
            box_data.append(stream_latencies)
            labels.append('First Result')
        
        bp = ax3.boxplot(box_data, labels=labels, patch_artist=True)
        colors = ['lightblue', 'lightcoral']
        for patch, color in zip(bp['boxes'], colors[:len(box_data)]):
            patch.set_facecolor(color)
        ax3.set_title("Latency Comparison", fontsize=14, fontweight='bold')
        ax3.set_ylabel("Seconds")
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Statistics Table
        ax4.axis('off')
        metrics_data = [['Metric', 'Value']]
        
        if total_latencies:
            metrics_data.append(['Count', f"{len(total_latencies)}"])
            metrics_data.append(['Mean', f"{np.mean(total_latencies):.3f}s"])
            metrics_data.append(['Median', f"{np.median(total_latencies):.3f}s"])
            metrics_data.append(['Std Dev', f"{np.std(total_latencies):.3f}s"])
            metrics_data.append(['Min', f"{np.min(total_latencies):.3f}s"])
            metrics_data.append(['Max', f"{np.max(total_latencies):.3f}s"])
        
        table = ax4.table(cellText=metrics_data[1:], colLabels=metrics_data[0], 
                         loc='center', cellLoc='center', colWidths=[0.4, 0.6])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        ax4.set_title("Statistical Summary", fontsize=14, fontweight='bold', y=1.05)
        
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=150, bbox_inches='tight')
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode()
        plt.close(fig)
        
        html_parts.append(f"""
        <div class="card">
            <h2>📈 Performance Analysis</h2>
            <img src='data:image/png;base64,{img_b64}' alt='Performance Charts' style="width: 100%;">
        </div>
        """)
    
    # Action Buttons
    html_parts.append("""
        <div class="card">
            <h2>🔧 Actions</h2>
            <p>
                <a href="/" class="btn">Run Test Again</a>
                <a href="/single-test" class="btn btn-warning">Single Request Test</a>
                <a href="/health" class="btn">Health Check</a>
                <a href="/test-audio" class="btn">Test Audio</a>
            </p>
        </div>
    """)
    
    # System Info
    html_parts.append(f"""
        <div class="card">
            <h2>💻 System Information</h2>
            <pre>
Python Version: {os.sys.version.split()[0]}
Current Directory: {os.getcwd()}
Audio File: {'✅ Found' if os.path.exists(MP3_FILE_PATH) else '❌ Missing'}
API Key: {'✅ Configured' if API_KEY else '❌ Not configured'}
Test Count: {TEST_COUNT}
Successful Tests: {successful}
            </pre>
        </div>
    """)
    
    # Footer
    html_parts.append(f"""
        <div style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #666; font-size: 12px; text-align: center;">
            <p>Deepgram STT Tester | Direct HTTP API | Generated at {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>Using model: {MODEL} | Language: {LANGUAGE}</p>
        </div>
    """)
    
    html_parts.append("""
        </div>
    </body>
    </html>
    """)
    
    return "".join(html_parts)

@app.get("/single-test")
async def single_test():
    """Test single request"""
    try:
        # Load audio
        audio = AudioSegment.from_mp3(MP3_FILE_PATH).set_frame_rate(SAMPLE_RATE).set_channels(1)
        audio_10s = audio[:10 * 1000]  # 10 seconds for quick test
        
        # Export as WAV
        buffer = io.BytesIO()
        audio_10s.export(buffer, format="wav")
        audio_data = buffer.getvalue()
        
        # Make single request
        result = await transcribe_audio_direct(audio_data, 0)
        
        html = f"""
        <html>
        <head>
            <title>Single Test Result</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
                .card {{ background: white; padding: 30px; border-radius: 10px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); max-width: 800px; margin: 0 auto; }}
                .success {{ border-left: 6px solid #4CAF50; }}
                .error {{ border-left: 6px solid #f44336; }}
                pre {{ background: #f8f9fa; padding: 15px; border-radius: 5px; overflow-x: auto; }}
                .btn {{ display: inline-block; padding: 10px 20px; background: #2196F3; color: white; text-decoration: none; border-radius: 5px; margin: 10px 5px; }}
            </style>
        </head>
        <body>
            <div class="card {'success' if result['status'] == 'OK' else 'error'}">
                <h1>{'✅ Success' if result['status'] == 'OK' else '❌ Error'}</h1>
                <h2>Single Test Result</h2>
                
                <p><strong>Status:</strong> {result['status']}</p>
                <p><strong>Time Taken:</strong> {result['total_latency']}</p>
                <p><strong>First Result Time:</strong> {result['stream_start_latency']}</p>
                
                <h3>Transcript:</h3>
                <pre>{result['words']}</pre>
                
                <h3>Full Response:</h3>
                <pre>{json.dumps(result, indent=2)}</pre>
                
                <p>
                    <a href="/" class="btn">Back to Main Test</a>
                    <a href="/health" class="btn">Health Check</a>
                </p>
            </div>
        </body>
        </html>
        """
        
        return HTMLResponse(content=html)
        
    except Exception as e:
        return HTMLResponse(content=generate_error_html(str(e)))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "api_key_configured": bool(API_KEY),
        "audio_file_exists": os.path.exists(MP3_FILE_PATH),
        "test_count": TEST_COUNT,
        "model": MODEL,
        "language": LANGUAGE
    }

@app.get("/test-audio")
async def test_audio():
    """Test audio loading"""
    try:
        if not os.path.exists(MP3_FILE_PATH):
            return {
                "status": "error",
                "message": f"File not found: {MP3_FILE_PATH}",
                "current_directory": os.getcwd(),
                "files": [f for f in os.listdir('.') if f.endswith('.mp3')]
            }
        
        audio = AudioSegment.from_mp3(MP3_FILE_PATH)
        return {
            "status": "success",
            "file": MP3_FILE_PATH,
            "duration_seconds": len(audio) / 1000,
            "channels": audio.channels,
            "sample_rate": audio.frame_rate,
            "file_size_bytes": os.path.getsize(MP3_FILE_PATH)
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

if __name__ == "__main__":
    uvicorn.run("sttTest:app", host="127.0.0.1", port=8000, reload=True)