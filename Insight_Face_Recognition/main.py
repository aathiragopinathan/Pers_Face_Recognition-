"""
Main application - Face Recognition with Camera Display + WebSocket Communication
Enhanced version with all features from main_headless.py but WITH camera window
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
from display_utils import draw_face_info, TempFace
from config import *

# =====  FRONTEND COMMUNICATION FUNCTIONS (Same as main_headless.py) =====

def notify_person_detected(person_id, confidence, is_unknown, message, distance):
    """Notify frontend that a person is detected"""
    try:
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
        url = f"http://localhost:8001/person_left_area"
        response = requests.post(url, timeout=2)
        print(f"🌐 [LOCAL_API] Person left area sent: Status {response.status_code}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ [LOCAL_API] Person left area error: {e}")
        return False

def notify_registration_start():
    """Notify frontend that registration is starting"""
    try:
        url = f"http://localhost:8001/registration_start"
        response = requests.post(url, timeout=2)
        print(f"🌐 [LOCAL_API] Registration start sent: Status {response.status_code}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ [LOCAL_API] Registration start error: {e}")
        return False

def notify_registration_angle_instruction(angle_name, instruction, angle_number, total_angles):
    """Notify frontend about current registration angle instruction"""
    try:
        url = f"http://localhost:8001/registration_angle_instruction"
        response = requests.post(url, json={
            "angle_name": angle_name,
            "instruction": instruction,
            "angle_number": angle_number,
            "total_angles": total_angles
        }, timeout=2)
        print(f"🌐 [LOCAL_API] Registration angle instruction sent: Status {response.status_code}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ [LOCAL_API] Registration angle instruction error: {e}")
        return False

def notify_registration_complete(person_id, success=True, message=None):
    """Notify frontend that registration is complete"""
    try:
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
        url = f"http://localhost:8001/get_user_response"
        response = requests.get(url, timeout=2)
        if response.status_code == 200:
            data = response.json()
            return data.get("has_response", False), data.get("register", False)
    except Exception:
        pass
    return False, False

def test_frontend_connection():
    """Test connection to FastAPI server"""
    try:
        response = requests.get("http://localhost:8001/", timeout=2)
        if response.status_code == 200:
            print("✅ [CONNECTION_TEST] FastAPI server is accessible")
            return True
        else:
            print(f"⚠️ [CONNECTION_TEST] FastAPI server responded with status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ [CONNECTION_TEST] Cannot reach FastAPI server: {e}")
        return False

# Enhanced capture function with WebSocket notifications (same as main_headless.py)
def enhanced_capture_embeddings_for_person(person_id, known_faces):
    """Enhanced capture with frontend notifications - WITH CAMERA DISPLAY"""
    from capture_manager import capture_embeddings_for_person
    from multi_angle_capture import MultiAngleCapture
    from registration_validator import RegistrationValidator
    
    print(f"🎯 Starting enhanced registration for {person_id}")
    
    # Notify frontend that registration is starting
    notify_registration_start()
    
    try:
        # Initialize components
        multi_angle_capture = MultiAngleCapture()
        registration_validator = RegistrationValidator()
        
        # Start multi-angle capture with notifications
        all_embeddings = []
        total_angles = len(ANGLE_INSTRUCTIONS)
        
        for angle_index, (angle_name, target_angle) in enumerate(ANGLE_INSTRUCTIONS, 1):
            instruction = f"Please {angle_name.lower().replace('_', ' ')}"
            
            # Notify frontend about current angle
            notify_registration_angle_instruction(
                angle_name=angle_name,
                instruction=instruction,
                angle_number=angle_index,
                total_angles=total_angles
            )
            
            print(f"📸 [{angle_index}/{total_angles}] {instruction}")
            
            # Capture embeddings for this angle - WITH CAMERA DISPLAY
            angle_embeddings = multi_angle_capture.capture_angle_embeddings(
                angle_name, target_angle, show_camera=True  # ← ENABLE CAMERA DISPLAY
            )
            
            if angle_embeddings:
                all_embeddings.extend(angle_embeddings)
                print(f"✅ Captured {len(angle_embeddings)} embeddings for {angle_name}")
            else:
                print(f"❌ Failed to capture {angle_name}")
                notify_registration_failed(f"Failed to capture {angle_name}", "capture_failed")
                return False
        
        # Validate registration
        if registration_validator.validate_registration(all_embeddings, known_faces):
            # Save to database
            from data_manager import save_known_faces
            
            known_faces[person_id] = {
                "embeddings": [emb.tolist() for emb in all_embeddings],
                "registered_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "angles_captured": len(ANGLE_INSTRUCTIONS)
            }
            
            save_known_faces(known_faces)
            
            # Notify success
            notify_registration_complete(person_id, True, f"Successfully registered {person_id}!")
            
            print(f"✅ Registration successful: {person_id}")
            return True
        else:
            print("❌ Registration failed: Person already exists (duplicate)")
            notify_registration_failed("Person already exists", "duplicate_person")
            return False
            
    except Exception as e:
        print(f"❌ Registration error: {e}")
        notify_registration_failed(f"Registration error: {str(e)}", "system_error")
        return False

def main():
    """Main function with camera display + WebSocket communication"""
    print("=" * 80)
    print("🏢 ENHANCED FACE RECOGNITION SYSTEM WITH CAMERA DISPLAY + WEBSOCKET")
    print("🎯 Motion Detection + Distance Validation + Frontend Communication + Camera View")
    print("🛑 Press 'q' or Ctrl+C to stop")
    print("=" * 80)
    
    # Test frontend connection
    test_frontend_connection()
    print()
    
    # Initialize components
    face_analyzer = FaceAnalyzer()
    known_faces = load_known_faces()
    
    if known_faces:
        build_faiss_index(known_faces)
        print(f"📊 Loaded {len(known_faces)} known faces")
    else:
        print("📊 No known faces found - starting fresh")
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Camera not accessible.")
        return
        
    cap.set(cv2.CAP_PROP_BUFFERSIZE, CAMERA_BUFFER_SIZE)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
    
    unknown_tracker = UnknownFaceTracker()
    
    # Statistics
    stats = {"recognitions": 0, "registrations": 0, "duplicates": 0, "motion_prevented": 0, "invalid_faces": 0}
    last_stats_print = time.time()
     
    last_detected_id = None
    last_detection_time = 0
    frame_count = 0
    person_in_area = False
    
    # Anti-flicker variables
    last_face_info = None
    no_face_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            frame_count += 1
            current_time = time.time()
            faces = face_analyzer.get_faces(frame)
            
            # Print stats periodically
            if current_time - last_stats_print >= 30:
                print(f"📊 [STATS] Known: {len(known_faces)} | Recognitions: {stats['recognitions']} | "
                      f"Registrations: {stats['registrations']} | Motion Prevented: {stats['motion_prevented']}")
                last_stats_print = current_time
            
            # Draw stats on camera feed
            info_lines = [
                f"Face Recognition + WebSocket Communication + Camera Display",
                f"Known People: {len(known_faces)} | Recognitions: {stats['recognitions']}",
                f"Registrations: {stats['registrations']} | Duplicates Prevented: {stats['duplicates']}",
                f"Motion Prevented: {stats['motion_prevented']} | Out of Range: {stats['invalid_faces']}"
            ]
            
            for i, line in enumerate(info_lines):
                font_size = 0.5 if i == 0 else 0.4
                cv2.putText(frame, line, (10, 25 + i*18), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), 1)
            
            # Process faces
            if faces:
                face = find_closest_valid_face(faces, frame.shape)
                
                if face is None:
                    # All faces out of range
                    stats["invalid_faces"] += 1 if frame_count % 30 == 0 else 0
                    no_face_count += 1
                    
                    if no_face_count > FACE_PERSISTENCE:
                        if person_in_area:
                            notify_person_left_area()
                            person_in_area = False
                        unknown_tracker.reset()
                        last_face_info = None
                    
                    cv2.putText(frame, "Move closer to camera (within 50cm)", 
                               (10, frame.shape[0]-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                else:
                    # Valid face within range
                    no_face_count = 0
                    person_in_area = True
                    is_valid, validation_info, distance = FaceValidator.validate_face(face, frame.shape)
                    
                    # Process every 3rd frame for performance (can be adjusted to every 60 frames for 2-second intervals)
                    if frame_count % 3 == 0:  # Change to % 60 for 2-second intervals
                        person_id, similarity = find_matching_person_fast(face.embedding, SIMILARITY_THRESHOLD)
                        
                        if person_id:
                            # Known person recognized
                            last_face_info = {
                                "type": "recognized",
                                "person_id": person_id,
                                "similarity": similarity,
                                "validation_info": validation_info,
                                "bbox": face.bbox
                            }
                            unknown_tracker.reset()
                            
                            # Throttle notifications (2 seconds minimum between same person)
                            if person_id != last_detected_id or (current_time - last_detection_time) > 2:
                                print(f"👋 [RECOGNIZED] Welcome back, {person_id}! (Similarity: {similarity:.4f}, Distance: {distance:.0f}cm)")
                                
                                # Notify frontend
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
                            # Unknown person
                            should_register, status_message, _ = unknown_tracker.track_unknown_face(
                                face.embedding, frame, face.bbox.astype(int), current_time)
                            
                            # Visual status
                            if "MOVING" in status_message:
                                visual_status = "unknown_moving"
                                if frame_count % 90 == 0:  # Every 90 frames (≈ every 3 seconds)
                                    print("🔄 Unknown person moving - waiting for stability...")
                                    stats["motion_prevented"] += 1
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
                                print("🆕 [STABLE] Unknown person stable - asking for registration decision...")
                                
                                # Notify frontend about unknown person
                                notify_person_detected(
                                    person_id=None,
                                    confidence=0,
                                    is_unknown=True,
                                    message="Unknown person detected. Register this person?",
                                    distance=distance
                                )
                                
                                # Wait for user decision via WebSocket
                                print("⏳ Waiting for registration decision from frontend...")
                                decision_timeout = 10  # 10 seconds
                                start_time = time.time()
                                user_decided = False
                                person_still_present = True
                                
                                while not user_decided and person_still_present and (time.time() - start_time < decision_timeout):
                                    # Continue showing camera feed during decision
                                    ret, decision_frame = cap.read()
                                    if ret:
                                        cv2.putText(decision_frame, "WAITING FOR REGISTRATION DECISION...", 
                                                   (10, decision_frame.shape[0]-60), 
                                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                                        cv2.putText(decision_frame, "Check your frontend for approval dialog", 
                                                   (10, decision_frame.shape[0]-30), 
                                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                                        cv2.imshow("Face Recognition System", decision_frame)
                                        
                                        if cv2.waitKey(1) & 0xFF == ord('q'):
                                            return
                                    
                                    # Check for user response
                                    has_response, should_register_user = check_user_registration_decision()
                                    if has_response:
                                        user_decided = True
                                        if should_register_user:
                                            print("✅ Registration approved by user")
                                            
                                            # Generate unique person ID
                                            person_id = generate_random_id()
                                            while person_id in known_faces:
                                                person_id = generate_random_id()
                                            
                                            # Show registration starting message on camera
                                            cv2.putText(frame, "STARTING REGISTRATION - FOLLOW INSTRUCTIONS", 
                                                       (10, frame.shape[0]-80), 
                                                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                                            cv2.imshow("Face Recognition System", frame)
                                            cv2.waitKey(2000)
                                            
                                            # Perform enhanced registration with camera display
                                            print(f"🎯 Starting registration for {person_id}")
                                            
                                            # Reload known faces before registration
                                            known_faces = load_known_faces()
                                            registration_successful = enhanced_capture_embeddings_for_person(person_id, known_faces)
                                            
                                            if registration_successful:
                                                # Rebuild index with new face
                                                known_faces = load_known_faces()  # Reload after registration
                                                build_faiss_index(known_faces)
                                                print(f"✅ [SUCCESS] Person {person_id} registered successfully!")
                                                
                                                stats["registrations"] += 1
                                                log_event("registration", person_id)
                                                last_detected_id = person_id
                                                last_detection_time = time.time()
                                            else:
                                                print(f"❌ [FAILED] Registration failed or cancelled")
                                                stats["duplicates"] += 1
                                            
                                            unknown_tracker.reset()
                                            last_face_info = None
                                            break
                                        else:
                                            print("❌ Registration denied by user")
                                            unknown_tracker.reset()
                                            break
                                    
                                    time.sleep(0.5)  # Check every 500ms
                                
                                if not user_decided:
                                    print("⏰ Registration decision timeout - continuing without registration")
                                    unknown_tracker.reset()
                                
                                continue
            else:
                # No faces detected
                no_face_count += 1
                
                if no_face_count > FACE_PERSISTENCE * 2:
                    if person_in_area:
                        notify_person_left_area()
                        person_in_area = False
                    unknown_tracker.reset()
                    last_face_info = None
                    cv2.putText(frame, "No faces detected - position yourself in front of camera", 
                               (10, frame.shape[0]-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Draw persistent face info (anti-flicker) on camera feed
            if last_face_info:
                temp_face = TempFace(last_face_info["bbox"])
                
                if last_face_info["type"] == "recognized":
                    frame = draw_face_info(frame, temp_face, last_face_info["person_id"], 
                                         last_face_info["similarity"], "recognized", "", 
                                         last_face_info["validation_info"])
                else:
                    frame = draw_face_info(frame, temp_face, None, 0, last_face_info["type"], 
                                         last_face_info.get("motion_info", ""), 
                                         last_face_info["validation_info"])
                    
                    # Show status messages on camera
                    if "status_message" in last_face_info:
                        status_lines = last_face_info["status_message"].split('|')
                        for i, line in enumerate(status_lines):
                            cv2.putText(frame, line.strip(), (10, frame.shape[0] - 100 + i*20), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # Show camera feed
            cv2.imshow("Face Recognition System", frame)
            
            # Exit condition
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user.")
    finally:
        # Cleanup
        if person_in_area:
            notify_person_left_area()
        
        cap.release()
        cv2.destroyAllWindows()
        
        print("\n✅ Session Summary:")
        for key, value in stats.items():
            print(f"   - {key.replace('_', ' ').title()}: {value}")
        print(f"   - Known People: {len(load_known_faces())}")

if __name__ == '__main__':
    print("🚀 Starting Face Recognition with Camera Display + WebSocket Communication")
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
        print("💡 Try running 'python fastapi_server.py' manually in another terminal")
    
    print()
    main()