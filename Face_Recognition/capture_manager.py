"""
Face capture and registration management
"""
import cv2
import time
import numpy as np
from tqdm import tqdm
from face_analyzer import FaceAnalyzer
from face_validator import FaceValidator
from data_manager import save_known_faces
from faiss_manager import check_for_duplicate_during_registration
from config import ANGLE_INSTRUCTIONS, EMBEDDINGS_PER_ANGLE, THRESHOLD

def capture_embeddings_for_person(person_id, known_faces):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return False

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    cap.set(cv2.CAP_PROP_FPS, 15)

    app = FaceAnalyzer().app  # Get the InsightFace app
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
                    face.embedding, known_faces, 0.75)
                
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