"""
External API communication for sending data to server
"""
import requests
import time
from config import EXTERNAL_API_ENDPOINT, EXTERNAL_API_TIMEOUT

def send_face_id_to_external_api(face_id, additional_data=None):
    """Send face ID to the external API endpoint"""
    try:
        payload = {
            "face_id": str(face_id),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "source": "nuc_camera_system"
        }
        
        if additional_data:
            payload.update(additional_data)
        
        headers = {"Content-Type": "application/json"}
        
        response = requests.post(
            EXTERNAL_API_ENDPOINT, 
            json=payload, 
            headers=headers, 
            timeout=EXTERNAL_API_TIMEOUT
        )
        
        if response.status_code == 200:
            print(f"[EXTERNAL_API] ✅ Successfully sent face ID {face_id} to server")
            return True
        else:
            print(f"[EXTERNAL_API] ⚠️ Server responded with status {response.status_code} for face ID {face_id}")
            return False
            
    except requests.exceptions.Timeout:
        print(f"[EXTERNAL_API] ⏰ Timeout while sending face ID {face_id} to server")
        return False
    except requests.exceptions.ConnectionError:
        print(f"[EXTERNAL_API] 🔌 Connection error while sending face ID {face_id} to server")
        return False
    except requests.exceptions.RequestException as e:
        print(f"[EXTERNAL_API] ❌ Error sending face ID {face_id} to server: {e}")
        return False

def send_recognition_event(person_id, confidence):
    """Send recognition event to external API"""
    additional_data = {
        "event_type": "recognition",
        "confidence": confidence
    }
    return send_face_id_to_external_api(person_id, additional_data)

def send_registration_event(person_id):
    """Send registration event to external API"""
    additional_data = {
        "event_type": "registration"
    }
    return send_face_id_to_external_api(person_id, additional_data)