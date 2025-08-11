"""
Enhanced face recognition with motion detection, duplicate prevention, 
distance & pose validation for enterprise use.


In multiple case scenario it is taking the face with that is 
farther rather than detecting the one that is within limit solve it before review.
"""
import cv2
import numpy as np
import json
import os
import time
import random
from insightface.app import FaceAnalysis
from tqdm import tqdm
from scipy.spatial.distance import cosine
import faiss

# Configuration
KNOWN_FACES_JSON = "known_faces.json"
THRESHOLD = 15.0
SIMILARITY_THRESHOLD = 0.65
DUPLICATE_THRESHOLD = 0.75
EMBEDDINGS_PER_ANGLE = 3
MOTION_THRESHOLD = 2500
STABILITY_TIME = 3.0
MOTION_HISTORY_SIZE = 10
REGISTRATION_DELAY = 4.0

# Distance & Pose Validation
MIN_FACE_SIZE = 80  # 50cm distance
MAX_FACE_SIZE = 400  # 10cm distance
MAX_YAW_ANGLE = 30  # Looking towards camera
MAX_PITCH_ANGLE = 25

ANGLE_INSTRUCTIONS = [
    ("Look Straight", 0), ("Turn Left", 20), ("Turn Right", -20),
    ("Look Up", -10), ("Look Down", 10)
]

# Initialize InsightFace
print("🔧 Initializing Face Recognition Model...")
app = FaceAnalysis(name='buffalo_sc')
app.prepare(ctx_id=0, det_size=(640, 640))
print("✅ Face Recognition Model Ready!")

faiss_index = None
person_id_mapping = []

class MotionDetector:
    def __init__(self):
        self.motion_threshold = MOTION_THRESHOLD
        self.history_size = MOTION_HISTORY_SIZE
        self.previous_face_region = None
        self.motion_history = []
        self.last_bbox = None
        
    def detect_face_motion(self, frame, face_bbox):
        try:
            x1, y1, x2, y2 = [max(0, min(frame.shape[1] if i%2==0 else frame.shape[0], coord)) 
                             for i, coord in enumerate(face_bbox)]
            
            if x2 <= x1 or y2 <= y1:
                return True, "Invalid face region", 0
            
            current_gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
            
            # Calculate bbox motion
            bbox_motion = 0
            if self.last_bbox is not None:
                prev_center = ((self.last_bbox[0] + self.last_bbox[2]) / 2, 
                              (self.last_bbox[1] + self.last_bbox[3]) / 2)
                curr_center = ((x1 + x2) / 2, (y1 + y2) / 2)
                bbox_motion = np.sqrt((curr_center[0] - prev_center[0])**2 + 
                                    (curr_center[1] - prev_center[1])**2)
            
            self.last_bbox = (x1, y1, x2, y2)
            
            # Calculate pixel motion
            pixel_motion = 0
            if self.previous_face_region is not None:
                if current_gray.shape != self.previous_face_region.shape:
                    current_gray = cv2.resize(current_gray, self.previous_face_region.shape[::-1])
                
                diff = cv2.absdiff(current_gray, self.previous_face_region)
                _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
                pixel_motion = np.sum(thresh > 0)
            
            self.previous_face_region = current_gray.copy()
            
            total_motion = pixel_motion + (bbox_motion * 100)
            self.motion_history.append(total_motion)
            if len(self.motion_history) > self.history_size:
                self.motion_history.pop(0)
            
            avg_motion = np.mean(self.motion_history) if self.motion_history else total_motion
            is_moving = avg_motion > self.motion_threshold
            
            motion_info = f"Motion: {avg_motion:.0f} ({'MOVING' if is_moving else 'STABLE'})"
            if bbox_motion > 0:
                motion_info += f" | Head: {bbox_motion:.1f}px"
            
            return is_moving, motion_info, avg_motion
            
        except Exception as e:
            return True, f"Motion error: {e}", 0
    
    def reset(self):
        self.previous_face_region = None
        self.motion_history = []
        self.last_bbox = None

class FaceValidator:
    @staticmethod
    def estimate_distance(face_width):
        """Estimate distance from face width (calibrated for 50cm max)"""
        return (14 * 900) / face_width if face_width > 0 else float('inf')
    
    @staticmethod
    def calculate_pose_angles(face):
        """Calculate yaw and pitch from face keypoints"""
        if not hasattr(face, 'kps') or face.kps is None or len(face.kps) < 3:
            return None, None
        
        left_eye, right_eye, nose = face.kps[0], face.kps[1], face.kps[2]
        eye_center_x = (left_eye[0] + right_eye[0]) / 2
        nose_offset = nose[0] - eye_center_x
        eye_distance = np.linalg.norm(right_eye - left_eye)
        
        yaw = (nose_offset / eye_distance) * 45 if eye_distance > 0 else 0
        pitch = ((nose[1] - eye_center_x) / eye_distance) * 30 if eye_distance > 0 else 0
        
        return yaw, pitch
    
    @classmethod
    def validate_face(cls, face, frame_shape):
        """Validate face - ONLY ACCEPT FACES WITHIN 50CM"""
        bbox = face.bbox.astype(int)
        face_width = bbox[2] - bbox[0]
        
        distance = cls.estimate_distance(face_width)
        
        # STRICT: Only accept faces within 50cm range
        if distance <= 50:
            is_valid = True
            validation_info = f"Valid (dist: {distance:.0f}cm)"
        else:
            is_valid = False
            validation_info = f"Out of range: {distance:.0f}cm > 50cm"
        
        return is_valid, validation_info, distance

class UnknownFaceTracker:
    def __init__(self):
        self.motion_detector = MotionDetector()
        self.reset()
        
    def reset(self):
        self.first_detection_time = None
        self.last_embedding = None
        self.consistent_detections = 0
        self.stability_start_time = None
        self.is_ready_for_registration = False
        self.motion_detector.reset()
    
    def is_same_face(self, embedding1, embedding2, threshold=0.8):
        try:
            return (1 - cosine(embedding1, embedding2)) > threshold
        except:
            return False
    
    def track_unknown_face(self, face_embedding, frame, face_bbox, current_time):
        is_moving, motion_info, _ = self.motion_detector.detect_face_motion(frame, face_bbox)
        
        if self.first_detection_time is None:
            self.first_detection_time = current_time
            self.last_embedding = face_embedding.copy()
            self.consistent_detections = 1
            self.stability_start_time = None
            return False, f"🔍 New unknown face detected | {motion_info}", 0
        
        if not self.is_same_face(face_embedding, self.last_embedding):
            self.reset()
            self.first_detection_time = current_time
            self.last_embedding = face_embedding.copy()
            self.consistent_detections = 1
            return False, f"🔍 New unknown face detected | {motion_info}", 0
        
        self.consistent_detections += 1
        self.last_embedding = face_embedding.copy()
        
        if is_moving:
            self.stability_start_time = None
            time_since_first = current_time - self.first_detection_time
            return False, f"🚫 Face moving - waiting for stability | {motion_info} | Detected: {time_since_first:.1f}s", 0
        else:
            if self.stability_start_time is None:
                self.stability_start_time = current_time
            
            stability_duration = current_time - self.stability_start_time
            
            if stability_duration >= STABILITY_TIME:
                total_time = current_time - self.first_detection_time
                
                if total_time >= REGISTRATION_DELAY and self.consistent_detections >= 15:
                    return True, f"✅ Face stable - starting registration | {motion_info}", 0
                else:
                    time_remaining = max(0, REGISTRATION_DELAY - total_time)
                    return False, f"⏱️ Face stable - waiting {time_remaining:.1f}s | {motion_info}", time_remaining
            else:
                stability_remaining = STABILITY_TIME - stability_duration
                return False, f"⏸️ Stabilizing... {stability_remaining:.1f}s | {motion_info}", stability_remaining

def generate_random_id():
    return str(random.randint(100000000, 999999999))

def build_faiss_index(known_faces):
    global faiss_index, person_id_mapping
    
    embeddings_list = []
    person_id_mapping = []
    
    for person_id, embeddings in known_faces.items():
        for embedding in embeddings:
            try:
                if isinstance(embedding, list):
                    embedding = np.array(embedding)
                
                embedding = embedding.astype('float32')
                
                if embedding.ndim == 1 and len(embedding) > 0:
                    embeddings_list.append(embedding)
                    person_id_mapping.append(person_id)
            except Exception:
                continue
    
    if embeddings_list:
        try:
            embeddings_array = np.array(embeddings_list)
            embedding_dim = embeddings_array.shape[1]
            
            faiss_index = faiss.IndexFlatIP(embedding_dim)
            faiss.normalize_L2(embeddings_array)
            faiss_index.add(embeddings_array)
            
            print(f"🚀 FAISS index built: {len(embeddings_list)} embeddings from {len(known_faces)} people")
        except Exception as e:
            print(f"❌ FAISS error: {e}")
            faiss_index = None
            person_id_mapping = []

def load_known_faces():
    if not os.path.exists(KNOWN_FACES_JSON):
        return {}
    
    try:
        with open(KNOWN_FACES_JSON, 'r') as f:
            data = json.load(f)
        
        validated_data = {}
        for person_id, embeddings in data.items():
            valid_embeddings = [np.array(emb) for emb in embeddings 
                              if np.array(emb).ndim == 1 and len(emb) > 0]
            if valid_embeddings:
                validated_data[person_id] = valid_embeddings
        
        print(f"📚 Loaded {len(validated_data)} people")
        return validated_data
    except Exception as e:
        print(f"❌ Load error: {e}")
        return {}

def save_known_faces(data):
    try:
        serializable_data = {
            person_id: [emb.tolist() if isinstance(emb, np.ndarray) else emb 
                       for emb in embeddings]
            for person_id, embeddings in data.items()
        }
        
        with open(KNOWN_FACES_JSON, 'w') as f:
            json.dump(serializable_data, f, indent=2)
        
        print(f"💾 Saved {len(serializable_data)} people")
    except Exception as e:
        print(f"❌ Save error: {e}")

def check_for_duplicate_during_registration(face_embedding, known_faces, threshold=DUPLICATE_THRESHOLD):
    global faiss_index, person_id_mapping
    
    # FAISS search
    if faiss_index is not None and len(person_id_mapping) > 0:
        try:
            query = face_embedding.astype('float32').reshape(1, -1)
            faiss.normalize_L2(query)
            similarities, indices = faiss_index.search(query, min(3, faiss_index.ntotal))
            
            for similarity, idx in zip(similarities[0], indices[0]):
                if similarity > threshold and 0 <= idx < len(person_id_mapping):
                    matched_id = person_id_mapping[idx]
                    print(f"🔍 DUPLICATE DETECTED: {matched_id} (similarity: {similarity:.4f})")
                    return True, matched_id, float(similarity)
        except Exception:
            pass
    
    # Manual fallback
    best_match_id, best_similarity = None, 0
    for person_id, embeddings in known_faces.items():
        for stored_embedding in embeddings:
            try:
                similarity = 1 - cosine(face_embedding, np.array(stored_embedding))
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match_id = person_id
            except Exception:
                continue
    
    if best_similarity > threshold:
        print(f"🔍 DUPLICATE DETECTED: {best_match_id} (similarity: {best_similarity:.4f})")
        return True, best_match_id, best_similarity
    
    return False, None, best_similarity

def capture_embeddings_for_person(person_id, known_faces):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return False

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    cap.set(cv2.CAP_PROP_FPS, 15)

    embeddings = []
    print(f"📸 Starting capture for Person ID: {person_id}")

    for instruction, expected_yaw in ANGLE_INSTRUCTIONS:
        print(f"[INFO] Please: {instruction}")
        time.sleep(2)
        captured = 0
        pbar = tqdm(total=EMBEDDINGS_PER_ANGLE, desc=f"Capturing: {instruction}")

        # Wait for pose
        turned = False
        hold_start = None
        while not turned:
            ret, frame = cap.read()
            if not ret:
                continue

            faces = app.get(frame)
            if not faces:
                continue

            face = faces[0]
            yaw, _ = FaceValidator.calculate_pose_angles(face)
            if yaw is not None and abs(yaw - expected_yaw) < THRESHOLD:
                if hold_start is None:
                    hold_start = time.time()
                elif time.time() - hold_start > 1.0:
                    turned = True
            else:
                hold_start = None

            # Show preview
            bbox = face.bbox.astype(int)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 255), 2)
            cv2.putText(frame, f"Please: {instruction}", (bbox[0], bbox[1] - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow("Registration", frame)
            cv2.waitKey(1)

        # Capture embeddings
        while captured < EMBEDDINGS_PER_ANGLE:
            ret, frame = cap.read()
            if not ret:
                continue

            faces = app.get(frame)
            if not faces:
                continue

            face = faces[0]
            yaw, _ = FaceValidator.calculate_pose_angles(face)
            
            if yaw is not None and abs(yaw - expected_yaw) < THRESHOLD:
                # Check for duplicates
                is_duplicate, matched_id, similarity = check_for_duplicate_during_registration(
                    face.embedding, known_faces, DUPLICATE_THRESHOLD)
                
                if is_duplicate:
                    pbar.close()
                    cap.release()
                    cv2.destroyAllWindows()
                    
                    # Show duplicate message
                    duplicate_frame = np.zeros((300, 800, 3), dtype=np.uint8)
                    cv2.putText(duplicate_frame, "DUPLICATE DETECTED!", (50, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                    cv2.putText(duplicate_frame, f"Existing ID: {matched_id}", (50, 100), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    cv2.putText(duplicate_frame, "Press any key to continue...", (50, 200), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)
                    cv2.imshow("Duplicate Detection", duplicate_frame)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                    return False
                
                embeddings.append(face.embedding)
                captured += 1
                pbar.update(1)
                
                # Visual feedback
                bbox = face.bbox.astype(int)
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 5)
                cv2.putText(frame, "✅ CAPTURED!", (bbox[0], bbox[1]-50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow("Registration", frame)
                cv2.waitKey(200)

        pbar.close()

    cap.release()
    cv2.destroyAllWindows()

    known_faces[person_id] = embeddings
    save_known_faces(known_faces)
    print(f"✅ SUCCESS: {len(embeddings)} embeddings saved for {person_id}")
    return True

def find_matching_person_fast(face_embedding, similarity_threshold=0.65):
    global faiss_index, person_id_mapping
    
    if faiss_index is None or len(person_id_mapping) == 0:
        return None, 0
    
    try:
        query = face_embedding.astype('float32').reshape(1, -1)
        faiss.normalize_L2(query)
        similarities, indices = faiss_index.search(query, min(5, faiss_index.ntotal))
        
        if len(similarities[0]) > 0 and similarities[0][0] > similarity_threshold:
            best_idx = indices[0][0]
            if 0 <= best_idx < len(person_id_mapping):
                return person_id_mapping[best_idx], float(similarities[0][0])
        
        return None, 0
    except Exception:
        return None, 0

def draw_face_info(frame, face, person_id, similarity, status="recognized", motion_info="", validation_info=""):
    bbox = face.bbox.astype(int)
    x1, y1, x2, y2 = bbox
    
    # Color coding
    colors = {
        "recognized": (0, 255, 0),
        "unknown": (0, 0, 255),
        "unknown_moving": (0, 165, 255),
        "unknown_stable": (255, 255, 0),
        "invalid": (128, 128, 128)
    }
    color = colors.get(status, (255, 255, 0))
    
    # Labels
    if status == "recognized":
        label = f"ID: {person_id}"
        confidence_text = f"Conf: {similarity:.3f}"
    elif status == "invalid":
        label = "INVALID FACE"
        confidence_text = validation_info
    else:
        label = "UNKNOWN"
        confidence_text = {"unknown": "New Person", "unknown_moving": "Wait for stability", 
                          "unknown_stable": "Ready for registration"}.get(status, "")
    
    # Draw
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
    
    info_lines = [label, confidence_text, motion_info, validation_info]
    info_lines = [line for line in info_lines if line]
    
    label_bg_height = 25 * len(info_lines) + 10
    cv2.rectangle(frame, (x1, y1-label_bg_height), (x2, y1), color, -1)
    
    for i, line in enumerate(info_lines):
        line_y = y1 - label_bg_height + 20 + i*20
        font_size = 0.6 if i == 0 else 0.4
        cv2.putText(frame, line, (x1+5, line_y), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), 2)
    
    # Confidence bar
    if status == "recognized" and similarity > 0:
        bar_width = int((similarity if similarity <= 1 else 1) * (x2-x1))
        cv2.rectangle(frame, (x1, y2+5), (x1 + bar_width, y2+15), color, -1)
        cv2.rectangle(frame, (x1, y2+5), (x2, y2+15), (128, 128, 128), 1)
    
    return frame

def log_event(event_type, person_id, similarity=None):
    log_entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "event": event_type,
        "person_id": person_id
    }
    if similarity is not None:
        log_entry["similarity"] = similarity
    
    log_file = "face_recognition_log.json"
    
    try:
        logs = []
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                logs = json.load(f)
        
        logs.append(log_entry)
        if len(logs) > 1000:
            logs = logs[-1000:]
        
        with open(log_file, 'w') as f:
            json.dump(logs, f, indent=2)
    except Exception:
        pass

def main():
    print("=" * 80)
    print("🏢 OFFLINE ENTERPRISE FACE RECOGNITION SYSTEM")
    print("🎯 Motion Detection + Distance/Pose Validation + Duplicate Prevention")
    print("🛑 Press 'q' or Ctrl+C to stop")
    print("=" * 80)
    
    known_faces = load_known_faces()
    
    if known_faces:
        build_faiss_index(known_faces)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Camera not accessible.")
        return
        
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)  # Set consistent FPS
    
    unknown_tracker = UnknownFaceTracker()
    
    # Statistics
    stats = {"recognitions": 0, "registrations": 0, "duplicates": 0, "motion_prevented": 0, "invalid_faces": 0}
    
    last_detected_id = None
    last_detection_time = 0
    frame_count = 0
    
    # ANTI-FLICKER VARIABLES
    last_face_info = None  # Store last valid face info
    no_face_count = 0      # Count frames without faces
    FACE_PERSISTENCE = 5   # Frames to keep showing last face info
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            frame_count += 1
            
            # REMOVE FRAME SKIPPING - Process every frame for smooth display
            faces = app.get(frame)
            
            # Draw stats (always visible)
            info_lines = [
                f"Face Recognition + Motion + Distance Validation (50cm max)",
                f"Known People: {len(known_faces)} | Recognitions: {stats['recognitions']}",
                f"Registrations: {stats['registrations']} | Duplicates Prevented: {stats['duplicates']}",
                f"Motion Prevented: {stats['motion_prevented']} | Out of Range: {stats['invalid_faces']}"
            ]
            
            for i, line in enumerate(info_lines):
                font_size = 0.7 if i == 0 else 0.5
                cv2.putText(frame, line, (10, 30 + i*25), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), 2)
            
            # PROCESS FACES WITH ANTI-FLICKER LOGIC
            if faces:
                face = faces[0]
                current_time = time.time()
                
                # Validate face first
                is_valid, validation_info, distance = FaceValidator.validate_face(face, frame.shape)
                
                if not is_valid:
                    # Face out of range - count invalid faces but don't reset immediately
                    stats["invalid_faces"] += 1 if frame_count % 30 == 0 else 0
                    no_face_count += 1
                    
                    # Only reset tracker after several invalid frames
                    if no_face_count > FACE_PERSISTENCE:
                        unknown_tracker.reset()
                        last_face_info = None
                    
                    # Show persistent "out of range" message
                    cv2.putText(frame, f"Face detected but out of range: {distance:.0f}cm > 50cm", 
                               (10, frame.shape[0]-60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                    cv2.putText(frame, "Move closer to camera for detection", 
                               (10, frame.shape[0]-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                    
                else:
                    # Valid face within 50cm - reset counters
                    no_face_count = 0
                    
                    # Proceed with recognition (only process every 3rd frame for performance)
                    if frame_count % 3 == 0:
                        person_id, similarity = find_matching_person_fast(face.embedding, SIMILARITY_THRESHOLD)
                        
                        if person_id:
                            # Known person
                            last_face_info = {
                                "type": "recognized",
                                "person_id": person_id,
                                "similarity": similarity,
                                "validation_info": validation_info,
                                "bbox": face.bbox
                            }
                            unknown_tracker.reset()
                            
                            if person_id != last_detected_id or (current_time - last_detection_time) > 2:
                                print(f"[RECOGNIZED] ID: {person_id} (Similarity: {similarity:.4f}, Distance: {distance:.0f}cm)")
                                stats["recognitions"] += 1
                                last_detected_id = person_id
                                last_detection_time = current_time
                                log_event("recognition", person_id, similarity)
                        else:
                            # Unknown person within 50cm
                            should_register, status_message, _ = unknown_tracker.track_unknown_face(
                                face.embedding, frame, face.bbox.astype(int), current_time)
                            
                            # Determine visual status
                            if "MOVING" in status_message:
                                visual_status = "unknown_moving"
                                stats["motion_prevented"] += 1 if frame_count % 30 == 0 else 0
                            elif "stable" in status_message.lower() and should_register:
                                visual_status = "unknown_stable"
                            else:
                                visual_status = "unknown"
                            
                            motion_info = status_message.split('|')[1].strip() if '|' in status_message else ""
                            
                            last_face_info = {
                                "type": visual_status,
                                "person_id": None,
                                "similarity": 0,
                                "validation_info": validation_info,
                                "motion_info": motion_info,
                                "bbox": face.bbox,
                                "status_message": status_message
                            }
                            
                            if should_register:
                                print("🆕 [STABLE] Starting registration...")
                                
                                person_id = generate_random_id()
                                while person_id in known_faces:
                                    person_id = generate_random_id()
                                
                                # Show registration message
                                cv2.putText(frame, "STARTING REGISTRATION...", (10, frame.shape[0]-80), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                                cv2.imshow("Face Recognition System", frame)
                                cv2.waitKey(2000)
                                
                                cap.release()
                                cv2.destroyAllWindows()
                                
                                # Registration
                                known_faces = load_known_faces()
                                registration_successful = capture_embeddings_for_person(person_id, known_faces)
                                
                                if registration_successful:
                                    build_faiss_index(load_known_faces())
                                    print(f"[SUCCESS] Person {person_id} registered!")
                                    stats["registrations"] += 1
                                    log_event("registration", person_id)
                                    last_detected_id = person_id
                                else:
                                    print(f" [CANCELLED] Registration cancelled due to duplicate")
                                    stats["duplicates"] += 1
                                
                                unknown_tracker.reset()
                                last_face_info = None
                                
                                # Restart camera
                                cap = cv2.VideoCapture(0)
                                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                                cap.set(cv2.CAP_PROP_FPS, 30)
                                
                                last_detection_time = time.time()
                                continue
                
                # Rebuild FAISS periodically
                if frame_count % 500 == 0:
                    known_faces = load_known_faces()
                    if known_faces and len(known_faces) > len(person_id_mapping) // 3:
                        build_faiss_index(known_faces)
            else:
                # No faces detected
                no_face_count += 1
                
                # Reset after no faces for a while
                if no_face_count > FACE_PERSISTENCE * 2:
                    unknown_tracker.reset()
                    last_face_info = None
                    cv2.putText(frame, "No faces detected", (10, frame.shape[0]-30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            # DRAW PERSISTENT FACE INFO (anti-flicker)
            if last_face_info:
                # Create a temporary face object for drawing
                class TempFace:
                    def __init__(self, bbox):
                        self.bbox = bbox
                
                temp_face = TempFace(last_face_info["bbox"])
                
                if last_face_info["type"] == "recognized":
                    frame = draw_face_info(frame, temp_face, last_face_info["person_id"], 
                                         last_face_info["similarity"], "recognized", "", 
                                         last_face_info["validation_info"])
                else:
                    frame = draw_face_info(frame, temp_face, None, 0, last_face_info["type"], 
                                         last_face_info.get("motion_info", ""), 
                                         last_face_info["validation_info"])
                    
                    # Show status messages for unknown faces
                    if "status_message" in last_face_info:
                        status_lines = last_face_info["status_message"].split('|')
                        for i, line in enumerate(status_lines):
                            cv2.putText(frame, line.strip(), (10, frame.shape[0] - 90 + i*20), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Show frame with smooth, non-flickering display
            cv2.imshow("Face Recognition System", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Session Summary:")
        for key, value in stats.items():
            print(f"   - {key.replace('_', ' ').title()}: {value}")
        print(f"   - Known People: {len(load_known_faces())}")

if __name__ == '__main__':
    main()
