# Cat Chase Gym Environment

This project provides a simple OpenAI Gym environment where an agent (green circle) tries to catch a cat (sprite) on an 8x8 grid. The cat moves randomly after every agent move.

Files:
- `cat_env.py`: The Gym environment implementation (class `CatChaseEnv`).
- `example.py`: Simple runner that plays a random policy and renders the environment.
- `play.py`: Interactive version where you control the agent with arrow keys.
- `images/peekaboo.png`: Cat sprite used when rendering (already in the workspace).
- `requirements.txt`: Dependencies.

Quick run (PowerShell):

```powershell
# Install dependencies
python -m pip install -r requirements.txt

# Watch random agent play
python example.py

# Play yourself using arrow keys
python play.py
```

Controls (when using play.py):
- ↑ Move up
- ↓ Move down
- ← Move left
- → Move right
- Q Quit game

If `pygame` fails to open a window under certain remote or headless environments, try running locally with an active display.
