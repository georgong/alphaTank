# **🚀 Alpha Tank - Multi-Agent Tank Battle**
**Alpha Tank** is a **multi-agent tank battle** game built with Pygame and designed for Reinforcement Learning (RL) training. We want to create a **fully customizable RL pipeline** (from environment to learning algorithms) as a deomstartion of showcasing how RL may learn from their opponents (depends on who, maybe another RL agent (i.e. PPO, SAC) or an intelligent bot (i.e. BFS bot, A* bot)) and use their charcteristics, along with the environement setup, to fight againts them and optimzie the reward.

Checkout real time tarining's [wandb report](https://wandb.ai/kaiwenbian107/multiagent-ppo-bot/reports/AlphaTank-Training--VmlldzoxMTgxNjU0MQ)

<p align="center">
  <img src="docs/assets/demo.gif" width="400"/>
</p>

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

- **Bullets will bounce off walls**

```python
python play_env.py --mode play
```

### ⌨️**Keyboard controls**
- Press **`R`** to reset the game.
- Press **`V`** to enable/disable visualizing the tank aiming direction.
- Press **`T`** to enable/disable visualizing the bullet trajectory.
- Press **`B`** to enable/disable visualizing the BFS shortest path.

---

## **🤖 Random Action Rendering**
```python
python play_env.py --mode play
python play_env.py --mode random
python play_env.py --mode bot
```

---

## **🤖 Bot Arena**
We support a variety of "intelligent" (manual crafted strategy) bot/expert to tarin our learning agent, run the following to see bots fighting aaginst each other (choose from `smart`, `random`, `aggressive`, `defensive`, `dodge`):

```python
python bot_arena.py --bot1 defensive  --bot2 dodge
```

---

## **🚀 Training A PPO/SAC Agent**
When training, choose **bot type** from `smart`, `random`, `aggressive`, `defensive`, `dodge`.
```python
python train_ppo_bot.py --bot-type smart
python train_ppo_ppo.py
```

---
## **🚀 Inference Rendering**
When inference, choose **bot type** from `smart`, `random`, `aggressive`, `defensive`, `dodge`.
```python
python inference.py --mode bot --bot-type smart
python inference.py --mode agent
```

---
### **🚀 Run a Pretrained AI Model**
Run oretrain model against bot:

```python
python inference.py --mode bot --bot-type aggressive --demo True
```
---
