"""
Main application - simplified imports from separate files
"""
import cv2
import time
from face_analyzer import FaceAnalyzer
from face_validator import FaceValidator, find_closest_valid_face
from unknown_tracker import UnknownFaceTracker
from data_manager import load_known_faces, log_event, generate_random_id
from faiss_manager import build_faiss_index, find_matching_person_fast
from capture_manager import capture_embeddings_for_person
from display_utils import draw_face_info, TempFace
from config import *

def main():
    print("=" * 80)
    print("🏢 SIMPLIFIED FACE RECOGNITION SYSTEM")
    print("🎯 Motion Detection + Distance/Pose Validation + Duplicate Prevention")
    print("🛑 Press 'q' or Ctrl+C to stop")
    print("=" * 80)
    
    # Initialize components
    face_analyzer = FaceAnalyzer()
    known_faces = load_known_faces()
    
    if known_faces:
        build_faiss_index(known_faces)
    
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
    
    last_detected_id = None
    last_detection_time = 0
    frame_count = 0
    
    # Anti-flicker variables
    last_face_info = None
    no_face_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            frame_count += 1
            faces = face_analyzer.get_faces(frame)
            
            # Draw stats
            info_lines = [
                f"Face Recognition + Motion + Distance Validation (50cm max)",
                f"Known People: {len(known_faces)} | Recognitions: {stats['recognitions']}",
                f"Registrations: {stats['registrations']} | Duplicates Prevented: {stats['duplicates']}",
                f"Motion Prevented: {stats['motion_prevented']} | Out of Range: {stats['invalid_faces']}"
            ]
            
            for i, line in enumerate(info_lines):
                font_size = 0.7 if i == 0 else 0.5
                cv2.putText(frame, line, (10, 30 + i*25), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), 2)
            
            # Process faces
            if faces:
                face = find_closest_valid_face(faces, frame.shape)
                current_time = time.time()
                
                if face is None:
                    # All faces out of range
                    stats["invalid_faces"] += 1 if frame_count % 30 == 0 else 0
                    no_face_count += 1
                    
                    if no_face_count > FACE_PERSISTENCE:
                        unknown_tracker.reset()
                        last_face_info = None
                    
                    cv2.putText(frame, "Move closer to camera (within 50cm)", 
                               (10, frame.shape[0]-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                else:
                    # Valid face within 50cm
                    no_face_count = 0
                    is_valid, validation_info, distance = FaceValidator.validate_face(face, frame.shape)
                    
                    # Process every 3rd frame for performance
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
                            # Unknown person
                            should_register, status_message, _ = unknown_tracker.track_unknown_face(
                                face.embedding, frame, face.bbox.astype(int), current_time)
                            
                            # Visual status
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
                                    print(f"[CANCELLED] Registration cancelled due to duplicate")
                                    stats["duplicates"] += 1
                                
                                unknown_tracker.reset()
                                last_face_info = None
                                
                                # Restart camera
                                cap = cv2.VideoCapture(0)
                                cap.set(cv2.CAP_PROP_BUFFERSIZE, CAMERA_BUFFER_SIZE)
                                cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
                                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
                                cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
                                
                                last_detection_time = time.time()
                                continue
            else:
                # No faces detected
                no_face_count += 1
                
                if no_face_count > FACE_PERSISTENCE * 2:
                    unknown_tracker.reset()
                    last_face_info = None
                    cv2.putText(frame, "No faces detected", (10, frame.shape[0]-30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            # Draw persistent face info (anti-flicker)
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
                    
                    # Show status messages
                    if "status_message" in last_face_info:
                        status_lines = last_face_info["status_message"].split('|')
                        for i, line in enumerate(status_lines):
                            cv2.putText(frame, line.strip(), (10, frame.shape[0] - 90 + i*20), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
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