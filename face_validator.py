"""
Face validation for distance and pose checking
"""
import numpy as np
from config import MIN_FACE_SIZE, MAX_FACE_SIZE, MAX_YAW_ANGLE, MAX_PITCH_ANGLE

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

def find_closest_valid_face(faces, frame_shape):
    """Find the closest face that meets validation criteria"""
    if not faces:
        return None
    
    valid_faces = []
    for face in faces:
        is_valid, validation_info, distance = FaceValidator.validate_face(face, frame_shape)
        if is_valid:
            valid_faces.append((face, distance))
    
    if not valid_faces:
        return None
    
    # Return the face with minimum distance
    closest_face = min(valid_faces, key=lambda x: x[1])
    return closest_face[0]