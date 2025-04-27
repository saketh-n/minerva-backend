# Military Message Simulator Server

A WebSocket server that simulates military-style message broadcasts. This server generates and sends random military-themed messages to connected clients at regular intervals.

## Features

- Real-time message broadcasting using WebSocket protocol
- Various message types including patrol reports, engagement notifications, and mission status updates
- Message categories: positive, negative, and neutral
- Unique message IDs for tracking
- Configurable message templates

## Prerequisites

- Python 3.x
- WebSockets library

## Installation

1. Clone the repository:
```bash
git clone <your-repository-url>
cd server
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the server:
```bash
python message_server.py
```

2. The WebSocket server will start on `ws://localhost:8765`

3. Connect to the server using a WebSocket client to receive messages

## Message Format

Messages are sent in JSON format with the following structure:
```json
{
    "id": 1,
    "action": "PATROL",
    "vehicle": "Armored Vehicle",
    "callSign": "Badger 1-3",
    "explanation": "Completed routine patrol of western perimeter, all sectors secure",
    "category": "neutral"
}
```

Some messages may include additional fields like "enemy" depending on the message type.