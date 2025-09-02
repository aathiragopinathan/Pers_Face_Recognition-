"""
Face Recognition System - Enhanced Frontend Communication
Two-way communication with detailed registration guidance
"""
import cv2
import time
import requests
import threading
from face_analyzer import FaceAnalyzer
from face_validator import FaceValidator, find_closest_valid_face
from unknown_tracker import UnknownFaceTracker
from data_manager import load_known_faces, log_event, generate_random_id
from faiss_manager import build_faiss_index, find_matching_person_fast
from capture_manager import capture_embeddings_for_person
from config import *

# =====  FRONTEND COMMUNICATION FUNCTIONS =====

def notify_frontend(event_type, data):
    """Send notification to frontend via HTTP"""
    try:
        # TODO: Update the URL construction when you provide your frontend URL
        url = f"{FRONTEND_BASE_URL}/{event_type}"
        response = requests.post(url, json=data, timeout=2)
        print(f"🌐 [FRONTEND] Sent {event_type}: Status {response.status_code}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ [FRONTEND] Communication error: {e}")
        return False

def notify_person_detected(person_id, confidence, is_unknown, message, distance):
    """Notify frontend that a person is detected"""
    try:
        # FIXED: Use proper endpoint construction
        url = f"http://localhost:8001/person_detected"
        response = requests.post(url, json={
            "person_id": person_id,
            "confidence": confidence,
            "is_unknown": is_unknown,
            "message": message,
            "distance": distance
        }, timeout=2)
        print(f"🌐 [LOCAL_API] Person detected sent: Status {response.status_code}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ [LOCAL_API] Person detection error: {e}")
        return False

def notify_person_left_area():
    """Notify frontend that person left detection area"""
    try:
        # FIXED: Use local FastAPI server
        url = f"http://localhost:8001/person_left_area"
        response = requests.post(url, json={}, timeout=2)
        print(f"🌐 [LOCAL_API] Person left area sent: Status {response.status_code}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ [LOCAL_API] Person left area error: {e}")
        return False

def notify_registration_start():
    """Notify frontend that registration process is starting"""
    try:
        # FIXED: Use local FastAPI server
        url = f"http://localhost:8001/registration_start"
        response = requests.post(url, json={}, timeout=2)
        print(f"🌐 [LOCAL_API] Registration start sent: Status {response.status_code}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ [LOCAL_API] Registration start error: {e}")
        return False

def notify_registration_angle_instruction(angle_name, instruction, angle_number, total_angles):
    """Notify frontend about current registration angle instruction"""
    try:
        # FIXED: Use local FastAPI server
        url = f"http://localhost:8001/registration_angle_instruction"
        response = requests.post(url, json={
            "angle_name": angle_name,
            "instruction": instruction,
            "angle_number": angle_number,
            "total_angles": total_angles
        }, timeout=2)
        print(f"🌐 [LOCAL_API] Angle instruction sent: {angle_name} - Status {response.status_code}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ [LOCAL_API] Angle instruction error: {e}")
        return False

def notify_registration_angle_complete(angle_name):
    """Notify frontend that current angle capture is complete"""
    try:
        # FIXED: Use local FastAPI server
        url = f"http://localhost:8001/registration_angle_complete"
        response = requests.post(url, json={
            "angle_name": angle_name
        }, timeout=2)
        print(f"🌐 [LOCAL_API] Angle complete sent: {angle_name} - Status {response.status_code}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ [LOCAL_API] Angle complete error: {e}")
        return False

def notify_registration_complete(person_id, success=True, message=None):
    """Notify frontend that registration is complete"""
    try:
        # FIXED: Use local FastAPI server
        url = f"http://localhost:8001/registration_complete"
        response = requests.post(url, json={
            "person_id": person_id,
            "success": success,
            "message": message or f"Registration completed for {person_id}!"
        }, timeout=2)
        print(f"🌐 [LOCAL_API] Registration complete sent: Status {response.status_code}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ [LOCAL_API] Registration complete error: {e}")
        return False

def notify_registration_failed(message="Registration failed", reason="Unknown"):
    """Notify frontend that registration failed"""
    try:
        # FIXED: Use local FastAPI server
        url = f"http://localhost:8001/registration_failed"
        response = requests.post(url, json={
            "message": message,
            "reason": reason
        }, timeout=2)
        print(f"🌐 [LOCAL_API] Registration failed sent: Status {response.status_code}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ [LOCAL_API] Registration failed error: {e}")
        return False

def check_user_registration_decision():
    """Check if user made registration decision via local FastAPI"""
    try:
        # FIXED: Use local FastAPI server
        url = f"http://localhost:8001/get_user_response"
        response = requests.get(url, timeout=2)
        if response.status_code == 200:
            data = response.json()
            return data.get("has_response", False), data.get("register", False)
    except Exception:
        pass
    return False, False

# ===== ENHANCED CAPTURE MANAGER WITH FRONTEND NOTIFICATIONS =====

def enhanced_capture_embeddings_for_person(person_id, known_faces):
    """Enhanced capture with frontend notifications for each angle"""
    print(f"📸 Starting enhanced registration for {person_id}")
    
    # Notify frontend that registration is starting
    notify_registration_start()
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot access camera for registration")
        notify_registration_failed("Camera access failed", "Cannot open camera")
        return False
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    
    face_analyzer = FaceAnalyzer()
    collected_embeddings = []
    total_angles = len(ANGLE_INSTRUCTIONS)
    
    try:
        for angle_index, (angle_name, target_angle) in enumerate(ANGLE_INSTRUCTIONS, 1):
            print(f"\n📸 Capturing angle {angle_index}/{total_angles}: {angle_name}")
            
            # Notify frontend about current angle instruction
            instruction = f"Please {angle_name.lower()}"
            notify_registration_angle_instruction(
                angle_name=angle_name,
                instruction=instruction,
                angle_number=angle_index,
                total_angles=total_angles
            )
            
            # Give user time to position themselves
            time.sleep(2)
            
            angle_embeddings = []
            attempts = 0
            max_attempts = 50  # 5 seconds at 10 FPS
            
            while len(angle_embeddings) < EMBEDDINGS_PER_ANGLE and attempts < max_attempts:
                ret, frame = cap.read()
                if not ret:
                    attempts += 1
                    continue
                
                faces = face_analyzer.get_faces(frame)
                
                if faces:
                    face = find_closest_valid_face(faces, frame.shape)
                    if face is not None:
                        # Check if user left detection area
                        is_valid, validation_info, distance = FaceValidator.validate_face(face, frame.shape)
                        
                        if not is_valid:
                            print("⚠️  Person moved out of detection area during registration")
                            notify_person_left_area()
                            # Wait for person to return
                            while True:
                                ret, frame = cap.read()
                                if ret:
                                    faces = face_analyzer.get_faces(frame)
                                    if faces:
                                        face = find_closest_valid_face(faces, frame.shape)
                                        if face is not None:
                                            is_valid, _, _ = FaceValidator.validate_face(face, frame.shape)
                                            if is_valid:
                                                print("✅ Person returned to detection area")
                                                # Re-notify current angle instruction
                                                notify_registration_angle_instruction(
                                                    angle_name=angle_name,
                                                    instruction=instruction,
                                                    angle_number=angle_index,
                                                    total_angles=total_angles
                                                )
                                                break
                                time.sleep(0.1)
                        
                        # Validate pose angle
                        pose_valid = True  # Simplified for this example
                        
                        if pose_valid:
                            angle_embeddings.append(face.embedding.copy())
                            print(f"✅ Captured embedding {len(angle_embeddings)}/{EMBEDDINGS_PER_ANGLE} for {angle_name}")
                
                attempts += 1
                time.sleep(0.1)
            
            if len(angle_embeddings) < EMBEDDINGS_PER_ANGLE:
                print(f"❌ Failed to capture enough embeddings for {angle_name}")
                notify_registration_failed(f"Failed to capture {angle_name} angle", "Insufficient embeddings")
                cap.release()
                return False
            
            collected_embeddings.extend(angle_embeddings)
            
            # Notify frontend that this angle is complete
            notify_registration_angle_complete(angle_name)
            
            print(f"✅ {angle_name} angle captured successfully")
            time.sleep(1)  # Brief pause between angles
        
        # Check for duplicates
        print(f"🔍 Checking for duplicates among {len(collected_embeddings)} embeddings...")
        
        for existing_person_id, existing_embeddings in known_faces.items():
            for new_embedding in collected_embeddings:
                for existing_embedding in existing_embeddings:
                    similarity = 1 - FaceValidator.cosine_distance(new_embedding, existing_embedding)
                    if similarity > DUPLICATE_THRESHOLD:
                        print(f"❌ Duplicate detected with {existing_person_id} (similarity: {similarity:.3f})")
                        notify_registration_failed(f"Person already registered as {existing_person_id}", "Duplicate detected")
                        cap.release()
                        return False
        
        # Save embeddings
        known_faces[person_id] = collected_embeddings
        
        with open(KNOWN_FACES_JSON, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_faces = {}
            for pid, embeddings in known_faces.items():
                serializable_faces[pid] = [emb.tolist() if hasattr(emb, 'tolist') else emb for emb in embeddings]
            json.dump(serializable_faces, f, indent=2)
        
        print(f"✅ Registration successful! Saved {len(collected_embeddings)} embeddings for {person_id}")
        notify_registration_complete(person_id, True, f"Successfully registered {person_id}")
        
        cap.release()
        return True
        
    except Exception as e:
        print(f"❌ Registration error: {e}")
        notify_registration_failed(f"Registration error: {str(e)}", "System error")
        cap.release()
        return False

# ===== CAMERA AND SYSTEM FUNCTIONS =====

def test_camera_access():
    """Test camera access before starting main loop"""
    print("📹 Testing camera access...")
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            height, width = frame.shape[:2]
            print(f"✅ Camera accessible: {width}x{height}")
            cap.release()
            return True
        else:
            print("❌ Camera opened but cannot read frames")
            cap.release()
            return False
    else:
        print("❌ Cannot open camera")
        return False

def test_frontend_connection():
    """Test local FastAPI connection"""
    print("🔌 Testing local FastAPI server connection...")
    try:
        # FIXED: Test local FastAPI server
        response = requests.get(f"http://localhost:8001/", timeout=3)
        print(f"✅ Local FastAPI server accessible: Status {response.status_code}")
        return True
    except Exception as e:
        print(f"⚠️  Local FastAPI server not accessible: {e}")
        print("   💡 Make sure FastAPI server is running on localhost:8001")
        return False

def main_headless():
    """Main function with enhanced frontend communication"""
    print("=" * 80)
    print("🖥️  FACE RECOGNITION SYSTEM - ENHANCED FRONTEND COMMUNICATION")
    print("📹 Camera active with registration guidance")
    print("🎯 Motion Detection + Distance Validation + Multi-Angle Registration")
    print("🛑 Press Ctrl+C to stop")
    print("=" * 80)
    print(f"🌐 Frontend URL: {FRONTEND_BASE_URL}")
    print("=" * 80)
    
    # Test camera first
    if not test_camera_access():
        print("💡 Camera troubleshooting needed")
        return
    
    # Initialize components
    print("🔄 Initializing system components...")
    
    try:
        face_analyzer = FaceAnalyzer()
        print("✅ Face analyzer initialized")
    except Exception as e:
        print(f"❌ Face analyzer failed: {e}")
        return
    
    known_faces = load_known_faces()
    print(f"✅ Loaded {len(known_faces)} known faces")
    
    if known_faces:
        build_faiss_index(known_faces)
        print("✅ FAISS index built")
    
    # Setup camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Camera not accessible after initialization.")
        return
        
    cap.set(cv2.CAP_PROP_BUFFERSIZE, CAMERA_BUFFER_SIZE)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
    
    unknown_tracker = UnknownFaceTracker()
    print("✅ Unknown tracker initialized")
    
    # Statistics
    stats = {
        "recognitions": 0, 
        "registrations": 0, 
        "duplicates": 0, 
        "motion_prevented": 0, 
        "invalid_faces": 0
    }
    
    last_detected_id = None
    last_detection_time = 0
    frame_count = 0
    last_stats_print = time.time()
    person_was_in_area = False
    
    print("\n🚀 Starting enhanced face recognition with frontend guidance...")
    print("🌐 All events and registration steps will be communicated to frontend")
    print()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            frame_count += 1
            faces = face_analyzer.get_faces(frame)
            current_time = time.time()
            
            # Print stats periodically
            if current_time - last_stats_print >= 30:
                print(f"📊 [STATS] Known: {len(known_faces)} | Recognitions: {stats['recognitions']} | "
                      f"Registrations: {stats['registrations']} | Motion Prevented: {stats['motion_prevented']}")
                last_stats_print = current_time
            
            # Process faces
            if faces:
                face = find_closest_valid_face(faces, frame.shape)
                
                if face is None:
                    # Person detected but out of range
                    if person_was_in_area:
                        print("⚠️  Person moved out of detection area")
                        notify_person_left_area()
                        person_was_in_area = False
                    stats["invalid_faces"] += 1 if frame_count % 30 == 0 else 0
                else:
                    # Valid face within range
                    if not person_was_in_area:
                        person_was_in_area = True
                    
                    is_valid, validation_info, distance = FaceValidator.validate_face(face, frame.shape)
                    
                    if frame_count % 3 == 0:
                        person_id, similarity = find_matching_person_fast(face.embedding, SIMILARITY_THRESHOLD)
                        
                        if person_id:
                            # Known person detected
                            unknown_tracker.reset()
                            
                            if person_id != last_detected_id or (current_time - last_detection_time) > 30:
                                print(f"✅ [RECOGNIZED] {person_id} | Confidence: {similarity:.3f} | Distance: {distance:.0f}cm")
                                
                                # Notify frontend of recognized person
                                notify_person_detected(
                                    person_id=person_id,
                                    confidence=similarity,
                                    is_unknown=False,
                                    message=f"Welcome back, {person_id}!",
                                    distance=distance
                                )
                                
                                stats["recognitions"] += 1
                                last_detected_id = person_id
                                last_detection_time = current_time
                                log_event("recognition", person_id, similarity)
                        else:
                            # Unknown person detected
                            should_register, status_message, _ = unknown_tracker.track_unknown_face(
                                face.embedding, frame, face.bbox.astype(int), current_time)
                            
                            if "MOVING" in status_message:
                                if frame_count % 90 == 0:
                                    print("🔄 Unknown person moving - waiting for stability...")
                                stats["motion_prevented"] += 1 if frame_count % 30 == 0 else 0
                            
                            if should_register:
                                print("🆕 [STABLE] Unknown person stable - asking frontend for registration decision...")
                                
                                # Notify frontend about unknown person (this will show dialog)
                                notify_person_detected(
                                    person_id=None,
                                    confidence=0,
                                    is_unknown=True,
                                    message="Unknown person detected. Would you like to register?",
                                    distance=distance
                                )
                                
                                # Wait for user decision WITH ENHANCED PERSON PRESENCE CHECK
                                print("⏳ Waiting for registration decision from frontend... (will cancel if person leaves)")
                                user_decided = False
                                person_still_present = True
                                frames_without_valid_person = 0
                                max_frames_without_person = 10  # ~5 seconds at 500ms checks = cancel
                                
                                while not user_decided and person_still_present:
                                    # Check for user decision from frontend
                                    has_response, should_register_user = check_user_registration_decision()
                                    
                                    if has_response:
                                        user_decided = True
                                        if should_register_user:
                                            print("✅ User approved registration - starting enhanced registration process...")
                                            
                                            person_id = generate_random_id()
                                            while person_id in known_faces:
                                                person_id = generate_random_id()
                                            
                                            print(f"📝 Starting enhanced registration for: {person_id}")
                                            
                                            # Release main camera for registration
                                            cap.release()
                                            
                                            # Enhanced registration process with frontend notifications
                                            known_faces = load_known_faces()
                                            registration_successful = enhanced_capture_embeddings_for_person(person_id, known_faces)
                                            
                                            if registration_successful:
                                                build_faiss_index(load_known_faces())
                                                print(f"🎉 [SUCCESS] {person_id} registered with enhanced guidance!")
                                                
                                                stats["registrations"] += 1
                                                log_event("registration", person_id)
                                                last_detected_id = person_id
                                            else:
                                                print(f"❌ [FAILED] Enhanced registration failed")
                                                stats["duplicates"] += 1
                                            
                                            unknown_tracker.reset()
                                            
                                            # Restart main camera
                                            print("📹 Restarting main camera...")
                                            cap = cv2.VideoCapture(0)
                                            cap.set(cv2.CAP_PROP_BUFFERSIZE, CAMERA_BUFFER_SIZE)
                                            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
                                            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
                                            cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
                                            
                                            last_detection_time = time.time()
                                            break
                                        else:
                                            print("❌ Registration cancelled by user")
                                            unknown_tracker.reset()
                                            break
                                    
                                    # ===== ENHANCED PERSON PRESENCE CHECK =====
                                    ret, current_frame = cap.read()
                                    if ret:
                                        current_faces = face_analyzer.get_faces(current_frame)
                                        
                                        if current_faces:
                                            # Check if there's a VALID face in detection area (not just any face)
                                            current_face = find_closest_valid_face(current_faces, current_frame.shape)
                                            if current_face is not None:
                                                # Check if it's within valid distance and pose
                                                is_valid, validation_info, current_distance = FaceValidator.validate_face(current_face, current_frame.shape)
                                                if is_valid:
                                                    # Person is still there and in valid range
                                                    frames_without_valid_person = 0
                                                    # Optionally: Check if it's the same person using embedding similarity
                                                else:
                                                    # Person detected but out of valid range
                                                    frames_without_valid_person += 1
                                                    if frames_without_valid_person == 3:  # Give some warnings
                                                        print("⚠️  Person moving out of detection area - please stay in position")
                                            else:
                                                # Face detected but not valid (too far, bad angle, etc.)
                                                frames_without_valid_person += 1
                                        else:
                                            # No faces detected at all
                                            frames_without_valid_person += 1
                                            if frames_without_valid_person == 3:
                                                print("⚠️  No person detected - please stay in front of camera")
                                        
                                        # If person has been absent/invalid for too long, cancel registration request
                                        if frames_without_valid_person >= max_frames_without_person:
                                            print("👤 Person left detection area or moved too far - cancelling registration request")
                                            notify_person_left_area()
                                            unknown_tracker.reset()
                                            person_still_present = False
                                            user_decided = True  # Exit the waiting loop
                                    else:
                                        # Could not read frame
                                        frames_without_valid_person += 1
                                    
                                    time.sleep(0.5)  # Check every 500ms
                                
                                if not person_still_present:
                                    print("❌ Registration request cancelled - person left area")
                                
                                continue
            else:
                # No faces detected
                if person_was_in_area:
                    print("👤 No faces detected - person left area")
                    notify_person_left_area()
                    person_was_in_area = False
                
            time.sleep(0.01)
                
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user")
    finally:
        cap.release()
        print("\n✅ Enhanced Session Summary:")
        for key, value in stats.items():
            print(f"   📊 {key.replace('_', ' ').title()}: {value}")

if __name__ == '__main__':
    print("🚀 Starting Face Recognition - Enhanced Frontend Communication Mode")
    print()
    
    # Test frontend connection
    test_frontend_connection()
    print()
    
    # Start local FastAPI server in background
    try:
        from fastapi_server import start_fastapi_server
        api_thread = threading.Thread(target=start_fastapi_server, daemon=True)
        api_thread.start()
        print("🌐 Local FastAPI WebSocket server started")
        time.sleep(2)
    except Exception as e:
        print(f"⚠️  Could not start local FastAPI server: {e}")
    
    print()
    main_headless()