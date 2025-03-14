# **🚀 Alpha Tank - Multi-Agent Tank Battle**
**Alpha Tank** is a **multi-agent tank battle** game built with Pygame and designed for Reinforcement Learning (RL) training. We want to create a **fully customizable RL pipeline** (from environment to learning algorithms) as a deomstartion of showcasing how RL may learn from their opponents (depends on who, maybe another RL agent (i.e. PPO, SAC) or an intelligent bot (i.e. BFS bot, A* bot)) and use their charcteristics, along with the environement setup, to fight againts them and optimzie the reward.

## **🛠 Installation**
### **1️⃣ Create a Conda Environment**
```bash
conda create -n alpha_tank python=3.9
conda activate alpha_tank
```

### **2️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
```

---

## **🎮 Human vs. Human Controls**
| **Player** | **Movement** | **Shoot** | **Reset Game** |
|-----------|------------|---------|--------------|
| **Player 1** | `WASD` | `F` | `R` |
| **Player 2** | `Arrow Keys` | `Space` | `R` |

- Press **`R`** to reset the game.
- **Bullets will bounce off walls**


## **🤖 Random Action Rendering**
```python
python play_env.py --mode play
python play_env.py --mode random
python play_env.py --mode bot
```

---

## **🚀 Training A PPO/SAC Agent**
```python
python train_ppo_bot.py
python train_ppo_ppo.py
```

---
## **🤖 Inference Rendering**
```python
python inference.py --mode bot
python inference.py --mode agent
```

---
### **Run a Pretrained AI Model**
#### Coming Soon
---
