# **🚀 Alpha Tank - Multi-Agent Tank Battle**
**Alpha Tank** is a **multi-agent tank battle** game built with **Pygame** and designed for **Reinforcement Learning (RL) training**

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
python play_env.py --mode random
```

---

## **🚀 Training A PPO/SAC Agent**
```python
python train_ppo_bot.py
python train_ppo_ppo.py
python train_sac_sac.py
```

---
### **Run a Pretrained AI Model**
#### Coming Soon
---
