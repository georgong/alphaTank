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

- Press **`R`** to reset the game.
- **Bullets will bounce off walls**

```python
python play_env.py --mode play
```
---

## **🤖 Random Action Rendering**
```python
python play_env.py --mode random
```

---

## **🚀 Training A PPO/SAC Agent**
```python
python train_ppo.py
python train_sac.py
```

---
### **Run a Pretrained AI Model**
#### Coming Soon
---
