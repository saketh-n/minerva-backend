import asyncio
import json
import random
import websockets
import os
import glob
from datetime import datetime

# Sample message templates
MESSAGE_TEMPLATES = [
    {
        "action": "PATROL",
        "vehicle": "Armored Vehicle",
        "callSign": "Badger 1-3",
        "explanation": "Completed routine patrol of western perimeter, all sectors secure",
        "category": "neutral",
        "action_mapping": "No-Op"
    }
]

# Function to load the latest influence analysis data
def load_influence_data():
    # Get all step folders in order
    step_dirs = sorted(glob.glob("influence_analysis/step_*"), key=lambda x: int(x.split('_')[1]))
    
    if not step_dirs:
        return generate_mock_influence_data()
    
    # Get the latest step directory
    latest_step_dir = step_dirs[-1]
    influence_file = os.path.join(latest_step_dir, "influence_analysis.json")
    
    if os.path.exists(influence_file):
        with open(influence_file, 'r') as f:
            data = json.load(f)
            # Extract the data for the latest step
            step_key = os.path.basename(latest_step_dir).replace('step_', '')
            if step_key in data:
                return data[step_key]
    
    # If we couldn't load the data, generate mock data
    return generate_mock_influence_data()

# Generate mock influence data with the appropriate structure
def generate_mock_influence_data():
    actions = ["No-Op", "Move", "Return to Base", "Engage Target"]
    
    result = {}
    for action in actions:
        # Generate random entity scores (100 values for example)
        entity_scores = [random.uniform(0.5, 5.0) for _ in range(100)]
        
        # Get top 5 entities
        top_entities_with_scores = sorted(enumerate(entity_scores), key=lambda x: x[1], reverse=True)[:5]
        top_entities = [idx for idx, _ in top_entities_with_scores]
        top_entity_scores = [score for _, score in top_entities_with_scores]
        
        # Top 3 entities
        top_3_entities = top_entities[:3]
        top_3_entity_scores = top_entity_scores[:3]
        
        # Generate random mission scores (7 features for example)
        mission_scores = [random.uniform(0.01, 0.5) for _ in range(7)]
        
        # Get top features
        top_features_with_scores = sorted(enumerate(mission_scores), key=lambda x: x[1], reverse=True)[:3]
        top_features = [idx for idx, _ in top_features_with_scores]
        top_feature_scores = [score for _, score in top_features_with_scores]
        
        # Visibility scores
        legacy_score = random.uniform(2.0, 5.0)
        dynasty_score = random.uniform(2.0, 5.0)
        
        # Sort visibility by score
        visibility_items = [("legacy", legacy_score), ("dynasty", dynasty_score)]
        visibility_items.sort(key=lambda x: x[1], reverse=True)
        top_visibility_names = [name for name, _ in visibility_items]
        top_visibility_scores = [score for _, score in visibility_items]
        
        # Overall top features
        all_features = [
            ("entity_" + str(top_entities[0]), top_entity_scores[0], "entities"),
            ("entity_" + str(top_entities[1]), top_entity_scores[1], "entities"),
            ("mission_" + str(top_features[0]), top_feature_scores[0], "mission"),
            ("visibility_" + top_visibility_names[0], top_visibility_scores[0], "visibility"),
            ("visibility_" + top_visibility_names[1], top_visibility_scores[1], "visibility")
        ]
        
        all_features.sort(key=lambda x: x[1], reverse=True)
        top_3_overall_features = [feature for feature, _, _ in all_features[:3]]
        top_3_overall_scores = [score for _, score, _ in all_features[:3]]
        top_3_overall_components = [component for _, _, component in all_features[:3]]
        
        result[action] = {
            "entities": {
                "scores": entity_scores,
                "top_entities": top_entities,
                "top_entity_scores": top_entity_scores,
                "top_3": {
                    "indices": top_3_entities,
                    "scores": top_3_entity_scores
                }
            },
            "mission": {
                "scores": mission_scores,
                "top_features": top_features,
                "top_feature_scores": top_feature_scores,
                "top_3": {
                    "indices": top_features,
                    "scores": top_feature_scores
                }
            },
            "visibility": {
                "legacy": legacy_score,
                "dynasty": dynasty_score,
                "top_3": {
                    "names": top_visibility_names,
                    "scores": top_visibility_scores
                }
            },
            "top_3_overall": {
                "features": top_3_overall_features,
                "scores": top_3_overall_scores,
                "components": top_3_overall_components
            }
        }
    
    return result

async def send_messages(websocket):
    message_id = 1
    while True:
        # Select a random message template
        message_template = random.choice(MESSAGE_TEMPLATES)
        
        # Load latest influence analysis data
        influence_data = load_influence_data()
        
        # Create message with unique ID and additional data
        message = {
            "id": message_id,
            "action": message_template["action"],
            "vehicle": message_template["vehicle"],
            "callSign": message_template["callSign"],
            "explanation": message_template["explanation"],
            "category": message_template["category"],
            "timestamp": datetime.now().isoformat(),
            "influence_analysis": influence_data
        }
        
        # Add enemy field if present in template
        if "enemy" in message_template:
            message["enemy"] = message_template["enemy"]
        
        # Send the message
        await websocket.send(json.dumps(message))
        print(f"Sent message ID: {message_id}")
        
        # Save the message to a file with timestamp
        filename = datetime.now().strftime("%d-%m-%Y %H-%M-%S.json")
        with open(filename, 'w') as f:
            json.dump(message, f, indent=2)
        
        message_id += 1
        await asyncio.sleep(3)  # Wait for 3 seconds

async def main():
    async with websockets.serve(send_messages, "localhost", 8765):
        print("WebSocket server started on ws://localhost:8765")
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    asyncio.run(main()) 