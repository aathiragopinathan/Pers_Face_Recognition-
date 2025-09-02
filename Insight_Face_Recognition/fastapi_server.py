"""
FastAPI server with WebSocket for real-time frontend communication
Enhanced for registration process guidance
"""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import json
import asyncio
from typing import Optional, List

app = FastAPI(title="Face Recognition Communication Server with Registration Guidance")

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# WebSocket Connection Manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"🔌 WebSocket connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        print(f"🔌 WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    async def send_to_all(self, message: dict):
        """Send message to all connected frontends"""
        if self.active_connections:
            message_str = json.dumps(message)
            print(f"📡 Broadcasting: {message.get('type', 'unknown')} - {message.get('data', {}).get('message', '')}")
            
            failed_connections = []
            for connection in self.active_connections:
                try:
                    await connection.send_text(message_str)
                except Exception as e:
                    print(f"❌ Failed to send to connection: {e}")
                    failed_connections.append(connection)
            
            for failed in failed_connections:
                self.disconnect(failed)
        else:
            print("⚠️  No WebSocket connections active")

manager = ConnectionManager()

# Enhanced System State
class SystemState:
    def __init__(self):
        self.current_event = None
        self.person_data = None
        self.registration_in_progress = False
        self.waiting_for_user_response = False
        self.user_response = None
        self.current_registration_angle = None
        self.registration_progress = 0
        self.person_in_detection_area = False

system_state = SystemState()

# Request/Response Models
class PersonDetected(BaseModel):
    person_id: Optional[str] = None
    confidence: Optional[float] = None
    is_unknown: bool = False
    message: str
    distance: Optional[float] = None

class RegistrationRequest(BaseModel):
    register: bool

class RegistrationAngle(BaseModel):
    angle_name: str
    instruction: str
    angle_number: int
    total_angles: int

class RegistrationComplete(BaseModel):
    person_id: str
    success: bool
    message: str

# WebSocket Endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    
    # Send initial status when connected
    await websocket.send_text(json.dumps({
        "type": "connected",
        "data": {
            "message": "Connected to Face Recognition System",
            "current_event": system_state.current_event,
            "system_ready": True
        }
    }))
    
    try:
        while True:
            # Listen for messages from frontend
            data = await websocket.receive_text()
            message = json.loads(data)
            
            print(f"📨 Received WebSocket message: {message}")
            
            # Handle registration decision via WebSocket
            if message.get("type") == "registration_decision":
                decision = message.get("data", {}).get("register", False)
                
                system_state.waiting_for_user_response = False
                system_state.user_response = decision
                
                if decision:
                    system_state.registration_in_progress = True
                    system_state.current_event = "registration_approved"
                    
                    await manager.send_to_all({
                        "type": "registration_approved",
                        "data": {"message": "Registration approved - starting capture process..."}
                    })
                else:
                    system_state.current_event = "registration_cancelled"
                    
                    await manager.send_to_all({
                        "type": "registration_cancelled", 
                        "data": {"message": "Registration cancelled by user"}
                    })
                
                print(f"✅ Registration decision: {'Approved' if decision else 'Cancelled'}")
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"❌ WebSocket error: {e}")
        manager.disconnect(websocket)

# HTTP API Endpoints
@app.get("/")
async def root():
    return {
        "message": "Face Recognition Communication Server with Registration Guidance",
        "websocket_url": "ws://localhost:8001/ws",
        "active_connections": len(manager.active_connections),
        "system_status": system_state.current_event
    }

@app.get("/system_status")
async def get_system_status():
    """Get current system status for frontend"""
    return {
        "current_event": system_state.current_event,
        "registration_in_progress": system_state.registration_in_progress,
        "waiting_for_user_response": system_state.waiting_for_user_response,
        "person_data": system_state.person_data,
        "websocket_connections": len(manager.active_connections),
        "current_registration_angle": system_state.current_registration_angle,
        "registration_progress": system_state.registration_progress,
        "person_in_detection_area": system_state.person_in_detection_area
    }

@app.post("/person_detected")
async def person_detected(data: PersonDetected):
    """Called when person is detected in camera range"""
    system_state.current_event = "person_detected"
    system_state.person_data = data.dict()
    system_state.person_in_detection_area = True
    
    if data.is_unknown:
        system_state.waiting_for_user_response = True
        
        await manager.send_to_all({
            "type": "unknown_person_detected",
            "data": {
                "message": data.message,
                "confidence": data.confidence,
                "person_id": data.person_id,
                "distance": data.distance,
                "action_required": True,
                "show_registration_dialog": True
            }
        })
        
        return {"status": "waiting_for_registration_decision", "websocket_sent": True}
    else:
        await manager.send_to_all({
            "type": "person_recognized",
            "data": {
                "person_id": data.person_id,
                "message": data.message,
                "confidence": data.confidence,
                "distance": data.distance
            }
        })
        
        return {"status": "person_recognized", "websocket_sent": True}

@app.post("/person_left_area")
async def person_left_area():
    """Called when person leaves detection area"""
    system_state.person_in_detection_area = False
    system_state.current_event = "person_left_area"
    
    await manager.send_to_all({
        "type": "person_left_area",
        "data": {
            "message": "Person left detection area",
            "show_waiting_message": True
        }
    })
    
    return {"status": "person_left_area", "websocket_sent": True}

@app.post("/registration_start")
async def registration_start():
    """Called when registration process begins"""
    system_state.registration_in_progress = True
    system_state.current_event = "registration_start"
    system_state.registration_progress = 0
    
    await manager.send_to_all({
        "type": "registration_start",
        "data": {
            "message": "Registration starting - please follow the instructions",
            "show_instruction_panel": True
        }
    })
    
    return {"status": "registration_start", "websocket_sent": True}

@app.post("/registration_angle_instruction")
async def registration_angle_instruction(angle_data: RegistrationAngle):
    """Called before capturing each angle"""
    system_state.current_registration_angle = angle_data.angle_name
    system_state.registration_progress = int((angle_data.angle_number / angle_data.total_angles) * 100)
    
    await manager.send_to_all({
        "type": "registration_angle_instruction",
        "data": {
            "angle_name": angle_data.angle_name,
            "instruction": angle_data.instruction,
            "angle_number": angle_data.angle_number,
            "total_angles": angle_data.total_angles,
            "progress": system_state.registration_progress,
            "message": f"Please {angle_data.instruction.lower()}"
        }
    })
    
    return {"status": "instruction_sent", "websocket_sent": True}

@app.post("/registration_angle_complete")
async def registration_angle_complete(angle_data: dict):
    """Called after capturing each angle"""
    await manager.send_to_all({
        "type": "registration_angle_complete",
        "data": {
            "angle_name": angle_data.get("angle_name"),
            "message": f"{angle_data.get('angle_name')} captured successfully",
            "progress": system_state.registration_progress
        }
    })
    
    return {"status": "angle_complete", "websocket_sent": True}

@app.post("/registration_complete")
async def registration_complete(data: RegistrationComplete):
    """Called when registration is completed"""
    system_state.registration_in_progress = False
    system_state.current_event = "registration_complete"
    system_state.current_registration_angle = None
    system_state.registration_progress = 100
    
    await manager.send_to_all({
        "type": "registration_complete",
        "data": {
            "person_id": data.person_id,
            "success": data.success,
            "message": data.message,
            "show_success_message": True
        }
    })
    
    return {"status": "registration_complete", "websocket_sent": True}

@app.post("/registration_failed")
async def registration_failed(data: dict):
    """Called when registration fails"""
    system_state.registration_in_progress = False
    system_state.current_event = "registration_failed"
    system_state.current_registration_angle = None
    
    await manager.send_to_all({
        "type": "registration_failed",
        "data": {
            "message": data.get("message", "Registration failed"),
            "reason": data.get("reason", "Unknown error"),
            "show_error_message": True
        }
    })
    
    return {"status": "registration_failed", "websocket_sent": True}

@app.get("/get_user_response")
async def get_user_response():
    """Face recognition script polls for user response (HTTP fallback)"""
    if system_state.user_response is not None:
        response = system_state.user_response
        system_state.user_response = None
        return {"has_response": True, "register": response}
    return {"has_response": False}

@app.post("/reset_system")
async def reset_system():
    """Reset system state"""
    system_state.current_event = "system_ready"
    system_state.person_data = None
    system_state.registration_in_progress = False
    system_state.waiting_for_user_response = False
    system_state.user_response = None
    system_state.current_registration_angle = None
    system_state.registration_progress = 0
    system_state.person_in_detection_area = False
    
    await manager.send_to_all({
        "type": "system_ready",
        "data": {"message": "System ready - waiting for next person"}
    })
    
    return {"status": "system_ready", "websocket_sent": True}

def start_fastapi_server():
    print("🚀 Starting FastAPI server with enhanced registration guidance...")
    uvicorn.run(app, host="localhost", port=8001)

if __name__ == "__main__":
    start_fastapi_server()