# Minerva Backend

Reinforcement learning backend for the Flag Frenzy simulation environment.

## Project Structure

```
├── analysis/           # Attribution analysis and visualization
├── config/             # Configuration settings
├── env/                # Flag Frenzy environment 
├── models/             # Neural network model definitions
├── ray_results/        # Training results and checkpoints
├── Replays/            # Simulation replays
├── tests/              # Test modules
│   ├── env_tests/      # Environment test modules
│   └── ...             # Other test modules
├── analyze_model.py    # Model attribution analysis script
├── message_server.py   # WebSocket message server
├── register_env.py     # Environment registration utility
├── test_env.py         # Test runner
└── train.py            # PPO training script
```

## Commands

- Test environment: `python test_env.py [--test {all,sim,gym,influence,policy}]`
- Train model: `python train.py`
- Analyze model: `python analyze_model.py --checkpoint [checkpoint_path] --episodes [num_episodes]`
- Run message server: `python message_server.py`

## Environment

The Flag Frenzy environment is a custom simulation environment for reinforcement learning. It provides a Gymnasium-compatible interface for training agents.

## Model Architecture

The core neural network architecture is defined in `models/model.py`. It includes:

- An entity encoder for processing entity features
- A visibility encoder for radar visibility
- A mission status encoder
- A controllable entities encoder
- Combined features processing
- Action type and parameter outputs

## Training

Training uses Ray RLlib with the PPO algorithm. The configuration can be adjusted in `config/constants.py`.

## Attribution Analysis

The system includes attribution analysis capabilities to help understand how different input features influence the model's decisions. Use `analyze_model.py` or the test functions in `tests/influence_tests.py` to analyze a trained model.

## Message Server

A WebSocket server that simulates military-style message broadcasts. This server generates and sends random military-themed messages to connected clients at regular intervals.

### Message Format

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

## License

Copyright © 2025