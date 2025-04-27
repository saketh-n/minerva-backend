import asyncio
import json
import random
import websockets
from datetime import datetime

# Sample message templates
MESSAGE_TEMPLATES = [
    {
        "action": "PATROL",
        "vehicle": "Armored Vehicle",
        "callSign": "Badger 1-3",
        "explanation": "Completed routine patrol of western perimeter, all sectors secure",
        "category": "neutral"
    },
    {
        "action": "ENGAGE",
        "vehicle": "Attack Helicopter",
        "callSign": "Viper 6-4",
        "enemy": "Infantry Squad",
        "explanation": "Engaged and dispersed hostile infantry group in sector B2",
        "category": "negative"
    },
    {
        "action": "RETURN TO BASE",
        "vehicle": "Cargo Helicopter",
        "callSign": "Atlas 2-2",
        "explanation": "Mission accomplished, returning to base with all units",
        "category": "positive"
    },
    {
        "action": "ENEMY CONTACT",
        "vehicle": "Fighter Squadron",
        "callSign": "Eagle 3-1",
        "enemy": "Bomber Formation",
        "explanation": "Successfully intercepted enemy bombers before reaching defensive line",
        "category": "negative"
    },
    {
        "action": "RESCUE COMPLETE",
        "vehicle": "Medical Helicopter",
        "callSign": "Angel 4-2",
        "explanation": "Successfully extracted wounded personnel, all units safe",
        "category": "positive"
    },
    {
        "action": "RECON",
        "vehicle": "UAV Drone",
        "callSign": "Reaper 1-1",
        "enemy": "Command Center",
        "explanation": "Identified enemy command post location in grid E7",
        "category": "neutral"
    },
    {
        "action": "UNDER FIRE",
        "vehicle": "Infantry Unit",
        "callSign": "Ghost 2-5",
        "enemy": "Sniper Team",
        "explanation": "Taking heavy fire from enemy position, requesting support",
        "category": "negative"
    },
    {
        "action": "MISSION SUCCESS",
        "vehicle": "Special Forces Team",
        "callSign": "Stalker 1-1",
        "explanation": "Primary objective secured, all teams accounted for",
        "category": "positive"
    }
]

async def send_messages(websocket):
    message_id = 1
    while True:
        # Select a random message template
        message_template = random.choice(MESSAGE_TEMPLATES)
        
        # Create message with unique ID
        message = {
            "id": message_id,
            "action": message_template["action"],
            "vehicle": message_template["vehicle"],
            "callSign": message_template["callSign"],
            "explanation": message_template["explanation"],
            "category": message_template["category"]
        }
        
        # Add enemy field if present in template
        if "enemy" in message_template:
            message["enemy"] = message_template["enemy"]
        
        # Send the message
        await websocket.send(json.dumps(message))
        print(f"Sent message: {message}")
        
        message_id += 1
        await asyncio.sleep(3)  # Wait for 3 seconds

async def main():
    async with websockets.serve(send_messages, "localhost", 8765):
        print("WebSocket server started on ws://localhost:8765")
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    asyncio.run(main()) 