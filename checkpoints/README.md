# Pretrained Checkpoints

This directory stores trained PPO checkpoints and exported model formats for the balancing bot.

## Checkpoint Files
- `best_model.cleanrl_model`: Best checkpoint state dictionary saved during training based on mean episodic return.

## Loading a Checkpoint

### PyTorch
```python
import torch
from rl.ppo import build_mlp

obs_size = 4
action_size = 2
actor = build_mlp(input_size=obs_size, output_size=action_size, num_hidden_layers=2, hidden_layer_size=16)

state_dict = torch.load("checkpoints/best_model.cleanrl_model", map_location="cpu", weights_only=True)
actor_weights = {k.replace("actor_mean.", ""): v for k, v in state_dict.items() if k.startswith("actor_mean.")}
actor.load_state_dict(actor_weights)
actor.eval()
```

### CLI Evaluation
```bash
python scripts/evaluate.py --model checkpoints/best_model.cleanrl_model --episodes 5 --viewer
```

