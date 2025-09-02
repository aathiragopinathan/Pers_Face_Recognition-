"""
Configuration with Frontend Communication Settings
"""

# File paths
KNOWN_FACES_JSON = "known_faces.json"
LOG_FILE = "face_recognition_log.json"

# Recognition thresholds
THRESHOLD = 15.0
SIMILARITY_THRESHOLD = 0.65
DUPLICATE_THRESHOLD = 0.75
EMBEDDINGS_PER_ANGLE = 3

# Motion detection parameters
MOTION_THRESHOLD = 1500
STABILITY_TIME = 2.0
MOTION_HISTORY_SIZE = 8
REGISTRATION_DELAY = 3.0

# Distance & Pose Validation
MIN_FACE_SIZE = 80
MAX_FACE_SIZE = 400
MAX_YAW_ANGLE = 30
MAX_PITCH_ANGLE = 25

# Registration angle instructions
ANGLE_INSTRUCTIONS = [
    ("Look Straight", 0), 
    ("Turn Left", 20), 
    ("Turn Right", -20),
    ("Look Up", -10), 
    ("Look Down", 10)
]

# Camera settings
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30
CAMERA_BUFFER_SIZE = 1

# Display settings
FACE_PERSISTENCE = 5
FRAME_SKIP = 3

# ===== FRONTEND COMMUNICATION SETTINGS =====
# TODO: https://infopoint-frontend.digitalplatform-dev.g8m.wepacloud.eu
FRONTEND_HOST = "infopoint-frontend.digitalplatform-dev.g8m.wepacloud.eu"  # Example: "myapp.example.com" or "localhost"
FRONTEND_PORT = 443  # Example: 443 for HTTPS, 80 for HTTP, 3000 for development
FRONTEND_PROTOCOL = "https"  # "https" or "http"

# Constructed frontend base URL
FRONTEND_BASE_URL = f"{FRONTEND_PROTOCOL}://{FRONTEND_HOST}:{FRONTEND_PORT}"

# Communication Events
class Events:
    PERSON_DETECTED = "person_detected"
    PERSON_LEFT_AREA = "person_left_area"
    REGISTRATION_REQUEST = "registration_request"
    REGISTRATION_APPROVED = "registration_approved"
    REGISTRATION_CANCELLED = "registration_cancelled"
    REGISTRATION_START = "registration_start"
    REGISTRATION_ANGLE_INSTRUCTION = "registration_angle_instruction"
    REGISTRATION_ANGLE_COMPLETE = "registration_angle_complete"
    REGISTRATION_COMPLETE = "registration_complete"
    REGISTRATION_FAILED = "registration_failed"
    PERSON_RECOGNIZED = "person_recognized"
    SYSTEM_READY = "system_ready"