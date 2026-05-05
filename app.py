from flask import Flask, request, send_file, jsonify, Response, stream_with_context
from flask_cors import CORS
import insightface
from insightface.app import FaceAnalysis
import cv2
import numpy as np
import os
import tempfile
import uuid
import concurrent.futures
import json
import time
import threading
import subprocess
import shutil

app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = None
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

print("Loading face analysis model...")
face_analyser = FaceAnalysis(name='buffalo_l')
face_analyser.prepare(ctx_id=-1) # -1 = CPU, 0 = GPU

print("Loading inswapper model...")
swapper = insightface.model_zoo.get_model('inswapper_128.onnx', download=False, download_zip=False)

print("Models loaded!")

UPLOAD_FOLDER = tempfile.gettempdir()
THREADS = 4
MAX_PROCESS_DIM = 540
VIDEO_PROCESS_FAST_DIM = 540
VIDEO_PROCESS_QUALITY_DIM = 720
VIDEO_FRAME_SKIP_FAST = [(2400, 4), (1800, 3), (1200, 2), (900, 1), (600, 1), (400, 1), (200, 1)]
VIDEO_FRAME_SKIP_QUALITY = [(2400, 2), (1800, 2), (1200, 1), (900, 1), (600, 1), (400, 1), (200, 1)]
FRAME_SKIP_THRESHOLDS = [(2400, 4), (1800, 3), (1200, 2), (900, 1), (600, 1), (400, 1), (200, 1)]

# Shared progress store
job_progress = {}
job_lock = threading.Lock()


def update_progress(job_id, pct, title, sub, stage, done=False):
with job_lock:
job_progress[job_id] = {
'pct': pct,
'title': title,
'sub': sub,
'stage': stage,
'done': done,
}


def cleanup_job(job_id, delay=30):
def _cleanup():
time.sleep(delay)
with job_lock:
job_progress.pop(job_id, None)
t = threading.Thread(target=_cleanup, daemon=True)
t.start()


def load_image_or_video_frame(file_storage):
if file_storage.mimetype.startswith('image/'):
img_bytes = np.frombuffer(file_storage.read(), np.uint8)
return cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

if file_storage.mimetype.startswith('video/'):
temp_id = uuid.uuid4().hex
temp_path = os.path.join(UPLOAD_FOLDER, f'source_{temp_id}.mp4')
file_storage.save(temp_path)
cap = cv2.VideoCapture(temp_path)
if not cap.isOpened():
return None
ret, frame = cap.read()
cap.release()
return frame if ret else None

return None


def resize_frame_for_speed(frame, max_dim=MAX_PROCESS_DIM):
h, w = frame.shape[:2]
if max(h, w) <= max_dim:
return frame, 1.0
scale = max_dim / max(h, w)
resized = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
return resized, scale


def get_video_processing_settings(frame_count, quality_mode='quality'):
if quality_mode == 'fast':
thresholds = VIDEO_FRAME_SKIP_FAST
max_dim = VIDEO_PROCESS_FAST_DIM
frame_skip = 1
for threshold, skip in thresholds:
if frame_count >= threshold:
frame_skip = skip
break
if frame_count > 1200:
process_dim = 360
elif frame_count > 800:
process_dim = 480
else:
process_dim = max_dim
else:
frame_skip = 1
process_dim = VIDEO_PROCESS_QUALITY_DIM

return frame_skip, process_dim


def ensure_minimum_output_fps(frames, input_fps, output_fps):
if output_fps <= input_fps or len(frames) == 0:
return frames
output_len = int(round(len(frames) * output_fps / input_fps))
if output_len <= len(frames):
return frames
ratio = len(frames) / output_len
duplicated = []
for i in range(output_len):
idx = min(int(i * ratio), len(frames) - 1)
duplicated.append(frames[idx])
return duplicated


def mux_audio(source_video_path, input_audio_path, output_path):
ffmpeg_path = shutil.which('ffmpeg') or shutil.which('ffmpeg.exe')
if not ffmpeg_path:
print('Audio muxing failed: ffmpeg not found in PATH')
return False

ffmpeg_cmd = [
ffmpeg_path,
'-y',
'-i', source_video_path,
'-i', input_audio_path,
'-c:v', 'copy',
'-c:a', 'aac',
'-map', '0:v:0',
'-map', '1:a:0?',
'-shortest',
output_path
]
try:
result = subprocess.run(
ffmpeg_cmd,
check=True,
stdout=subprocess.PIPE,
stderr=subprocess.PIPE,
text=True,
)
print('Audio muxing succeeded')
return True
except subprocess.CalledProcessError as e:
print('Audio muxing failed:')
print('stdout:', e.stdout)
print('stderr:', e.stderr)
return False
except Exception as e:
print(f'Audio muxing failed: {e}')
return False


@app.route('/')
def index():
return app.send_static_file('index.html')


# ── SSE progress endpoint ──────────────────────────────────────────────────────
@app.route('/swap/progress/<job_id>')
def progress_stream(job_id):
def generate():
sent_done = False
for _ in range(300):
with job_lock:
state = job_progress.get(job_id)

if state:
yield f"data: {json.dumps(state)}\n\n"
if state.get('done'):
sent_done = True
break
else:
yield f"data: {json.dumps({'pct':0,'title':'Waiting...','sub':'Initializing','stage':0,'done':False})}\n\n"

time.sleep(0.5)

if not sent_done:
yield f"data: {json.dumps({'pct':100,'title':'Done','sub':'','stage':4,'done':True})}\n\n"

return Response(
stream_with_context(generate()),
mimetype='text/event-stream',
headers={
'Cache-Control': 'no-cache',
'X-Accel-Buffering': 'no',
}
)


# ── Image swap ────────────────────────────────────────────────────────────────
@app.route('/swap/image', methods=['POST'])
def swap_image():
try:
if 'source' not in request.files or 'target' not in request.files:
return jsonify({'error': 'Both source and target images are required'}), 400

quality = request.form.get('quality', 'quality')
src_img = load_image_or_video_frame(request.files['source'])
tgt_bytes = np.frombuffer(request.files['target'].read(), np.uint8)
tgt_img = cv2.imdecode(tgt_bytes, cv2.IMREAD_COLOR)

if src_img is None or tgt_img is None:
return jsonify({'error': 'Could not decode images'}), 400

src_faces = face_analyser.get(src_img)
tgt_faces = face_analyser.get(tgt_img)

if len(src_faces) == 0:
return jsonify({'error': 'No face detected in source image'}), 400
if len(tgt_faces) == 0:
return jsonify({'error': 'No face detected in target image'}), 400

result = tgt_img.copy()
for face in tgt_faces:
result = swapper.get(result, face, src_faces[0], paste_back=True)

out_path = os.path.join(UPLOAD_FOLDER, f'result_{uuid.uuid4().hex}.jpg')
jpeg_quality = 85 if quality == 'fast' else 95
cv2.imwrite(out_path, result, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
return send_file(out_path, mimetype='image/jpeg')

except Exception as e:
print(f"Image swap error: {e}")
return jsonify({'error': str(e)}), 500


# ── Video swap ────────────────────────────────────────────────────────────────
@app.route('/swap/video', methods=['POST'])
def swap_video():
job_id = uuid.uuid4().hex

try:
if 'source' not in request.files or 'target' not in request.files:
return jsonify({'error': 'Both source face image and target video are required'}), 400

update_progress(job_id, 5, 'Uploading video...', 'Reading source face', 0)

src_img = load_image_or_video_frame(request.files['source'])
if src_img is None:
return jsonify({'error': 'Could not decode source media'}), 400

quality = request.form.get('quality', 'quality')
src_faces = face_analyser.get(src_img)

if len(src_faces) == 0:
return jsonify({'error': 'No face detected in source image'}), 400

update_progress(job_id, 15, 'Analyzing target...', 'Opening video file', 1)

video_id = uuid.uuid4().hex
video_in_path = os.path.join(UPLOAD_FOLDER, f'input_{video_id}.mp4')
video_out_path = os.path.join(UPLOAD_FOLDER, f'output_{video_id}.mp4')

request.files['target'].save(video_in_path)

cap = cv2.VideoCapture(video_in_path)
if not cap.isOpened():
return jsonify({'error': 'Could not open video file'}), 400

fps = cap.get(cv2.CAP_PROP_FPS)
if not fps or fps == 0:
fps = 30
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Read all frames
frames_buffer = []
while True:
ret, frame = cap.read()
if not ret:
break
frames_buffer.append(frame)
cap.release()

frame_skip, process_dim = get_video_processing_settings(len(frames_buffer), quality)
process_indices = [i for i in range(len(frames_buffer)) if i % frame_skip == 0]
estimated_count = len(process_indices)
update_progress(job_id, 25, 'Processing frames...', f'0 / {estimated_count} processed', 2)
print(f"Processing {len(frames_buffer)} frames with skip={frame_skip} and dim={process_dim} ({estimated_count} processed)...")

processed_count = [0]

def process_frame(args):
idx, frame = args
try:
proc_frame, scale = resize_frame_for_speed(frame, process_dim)
tgt_faces = face_analyser.get(proc_frame)
result = proc_frame.copy()
for face in tgt_faces:
result = swapper.get(result, face, src_faces[0], paste_back=True)
if scale != 1.0:
result = cv2.resize(result, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
except Exception as e:
print(f"Frame {idx} error: {e}")
result = frame

processed_count[0] += 1
done_n = processed_count[0]
pct = 25 + int((done_n / estimated_count) * 55)
update_progress(job_id, pct, 'Processing frames...', f'{done_n} / {estimated_count} processed', 2)
return (idx, result)

args = [(idx, frames_buffer[idx]) for idx in process_indices]
with concurrent.futures.ThreadPoolExecutor(max_workers=min(THREADS, estimated_count)) as executor:
results = list(executor.map(process_frame, args))

results.sort(key=lambda x: x[0])
result_map = {idx: frame for idx, frame in results}
processed_frames = []
last_frame = None
for i in range(len(frames_buffer)):
if i in result_map:
last_frame = result_map[i]
if last_frame is None:
last_frame = frames_buffer[i]
processed_frames.append(last_frame)

output_fps = float(fps)
processed_frames = ensure_minimum_output_fps(processed_frames, fps, output_fps)

output_width = width
output_height = height

update_progress(job_id, 82, 'Writing video...', 'Saving processed frames', 3)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(video_out_path, fourcc, output_fps, (output_width, output_height))
for frame in processed_frames:
if frame.shape[1] != output_width or frame.shape[0] != output_height:
frame = cv2.resize(frame, (output_width, output_height), interpolation=cv2.INTER_LINEAR)
out.write(frame)
out.release()

final_path = video_out_path
audio_path = video_out_path + '.audio.mp4'
if mux_audio(video_out_path, video_in_path, audio_path):
if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
final_path = audio_path
else:
print('Audio muxing succeeded but final audio output file is missing or empty:', audio_path)
else:
print('Audio muxing failed, returning video without audio')

update_progress(job_id, 90, 'Finalizing video...', 'Saving processed frames', 3)
print(f"Done! {len(results)} frames processed.")

update_progress(job_id, 100, 'Done!', 'Download ready', 4, done=True)
cleanup_job(job_id)

response = send_file(final_path, mimetype='video/mp4')
response.headers['X-Job-Id'] = job_id
return response

except Exception as e:
print(f"Video swap error: {e}")
update_progress(job_id, 0, 'Error', str(e), 0, done=True)
return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
