"""
Configuration settings for Face Recognition System
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
MOTION_THRESHOLD = 1500         # Reduced from 2500 - more tolerant
STABILITY_TIME = 2.0           # Reduced from 3.0 seconds
MOTION_HISTORY_SIZE = 8        # Reduced from 10 - faster adaptation
REGISTRATION_DELAY = 3.0       # Reduced from 4.0 seconds

# Distance & Pose Validation
MIN_FACE_SIZE = 80             # 50cm distance
MAX_FACE_SIZE = 400            # 10cm distance
MAX_YAW_ANGLE = 40             # Looking towards camera
MAX_PITCH_ANGLE = 25

# Registration angle instructions
ANGLE_INSTRUCTIONS = [
    ("Look Straight", +30), #if_centered=0
    ("Turn Left", +50), #if_centered=20
    ("Turn Right", +10), #if_centered=-20
    ("Look Up", -10), 
    ("Look Down", 10)
]

# Camera settings
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30
CAMERA_BUFFER_SIZE = 1

# Display settings
FACE_PERSISTENCE = 5           # Frames to keep showing last face info
FRAME_SKIP = 3                 # Process every 3rd frame for performance