# **🚀 Alpha Tank - Multi-Agent Tank Battle**
**Alpha Tank** is a **multi-agent tank battle** game built with Pygame and designed for Reinforcement Learning (RL) training. We want to create a **fully customizable RL pipeline** (from environment to learning algorithms) as a demonstration of showcasing how RL may learn from their opponents (depends on who, maybe another RL agent (i.e. PPO, SAC) or an intelligent bot (i.e. BFS bot, A* bot)) and use their charcteristics, along with the environement setup, to fight againts them and optimzie the reward.

 We support both wandb loggings as well as saving `agent_dict` with `agent_parameters` both in the `.pt` file ([this base config file](configs/config_teams.py) need to remain the same for training and inference). Checkout real time training on this [wandb report](https://wandb.ai/kaiwenbian107/multiagent-ppo-bot/reports/AlphaTank-Training--VmlldzoxMTgxNjU0MQ).

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

### ⌨️**Keyboard controls**
- Press **`R`** to reset the game.
- Press **`V`** to enable/disable visualizing the tank aiming direction.
- Press **`T`** to enable/disable visualizing the bullet trajectory.
- Press **`B`** to enable/disable visualizing the BFS shortest path.

The complete documentation of the environment is in [here](docs/structure.md)

---

## **💣 Battle's Setting**
We support many many different modes, to avoid confusion, we will be going over them one by one, the general structure goes like the following:

```
algorithms
├── bot_mode
│   ├── single_mode
│   ├── cycle_learning
│   ├── team_mode
├── agent_mode
│   ├── single_mode
│   ├── team_mode
```

Notice that this is not specifically how the code are structured but rather a conceptual framework of our system:
- `Algorithms`: include two popular RL algorithms: PPO & SAC, this is the main algorithm for training the learning agent, we will explain more later.
  - `Bot mode`: include many different types of human heuritsic bots.
    - Supports cycle training + curriculum learning, but only for single agent-to-bot mode.
    - Supports team playing against team of agents, team is fully customziable with mixes between agents, bots, and human players.
  - `Agent mode`: include different algorithm fighting againts each other.
    - Supports team playing against team of agents, team is fully customziable with mixes between agents, bots, and human players.

We try to keep our codebase as modularzie and conatined as possible, so we have seperated out the ***base team playing environment*** and single agent-to-agent or agent-to-bot environment while maintaining a coherent API call. Similarly, we have seperated out the ***main learning agent training/inference loop*** for clearness for now, we will make abstract classes for agent similar to how we have done with the bot later on.

---
### **⛰️ Environment Demostrations**

#### **Random Action Rendering**
```python
python play_env.py --mode play
python play_env.py --mode team
python play_env.py --mode bot
```

#### **Bot Arena**
We support a variety of "intelligent" (manual crafted strategy) bot/exper using our very own ***bot factory*** to train our learning agent, run the following to see bots fighting aginst each other (choose from `smart`, `random`, `aggressive`, `defensive`, `dodge`), the complete documentation of the environment is in [here](docs/bots.md).

```python
python bot_arena.py --bot1 defensive  --bot2 dodge
```

---

### **🚀 Training A Agent**

For all training, the varaiable `TERMINATE_TIME` in [this config file](configs/config_basic.py) is very critical as it detrenmines whether or not the agent will have infinite life during training, promoting chasing and attacking actions, this flag should be set to None during ineference at all time.

#### **Training Single Agent-to-X**
When training, choose **bot type** from `smart`, `random`, `aggressive`, `defensive`, `dodge`. All the basic environmental configs are taken in as dictionary specified in [this config file](configs/config_basic.py).

```python
python train_ppo_bot.py --bot-type smart
python train_ppo_cycle.py
python train_ppo_ppo.py
```

*Team playing is also a form of cycle learning as well, just happening all at once.

#### **Training Team Players**
All the specific `config_var` are taken in as dictionary specified in [this config file](configs/config_teams.py), notice that this `experiment_name` should be consitent for training and inference.

```python
python train_multi_ppo.py --experiment_name 2a_vs_2b --config_var team_vs_bot_configs
```

---

### **🤖 Inference Modes**

#### **General Inference Rendering**
When inference with single agent-to-bot setting, you can choose **bot type** from `smart`, `random`, `aggressive`, `defensive`, `dodge`.

```python
python inference.py --mode bot --bot-type smart --weakness 0.1 --algorithm ppo
python inference.py --mode agent --algorithm ppo 
```

#### **Team Inference Rendering**
Similar with team training, all the configs are taken in as dictionary specified in [this config file](configs/config_teams.py). Notice that this  `experiment_name` should be consitent for training and inference. There is no need to pass in the `config_var` as everything will be saved in teh checkpointing system.

```python 
python inference_multi.py --experiment_name 2a_vs_2b
```

#### **Run a Pretrained AI Model**
We have provided a [list of checkpoints](/demo_checkpoints/) that we have ran to the users for baseline comparisons. Run our trained single agent-to-bot model by the following. The demos will be defaulted to running the following, you can change the `bot-type` args or `experiment_name` to change demo check points to load.

Single agent checkpoints:
- Single: ppo v.s. aggersive bot (octagon)
- Single: ppo v.s. smart bot (non-octagon)

Team players checkpoints:
- Team: 2A ppo v.s. 2B smart (non-octagon)
- Team: 2A ppo v.s. 3B smart + defensive (octagon)
- Team: 2A ppo v.s. 1B defensive (octagon)

For team playing mode, we support using a joy_stick_controller to play against the agent, you will be replacing one of the bot, only supported for demo mode.

```python
python inference.py --mode bot --bot-type aggressive --algorithm ppo --demo True
python inference_multi.py --experiment_name 2a_vs_3b --demo True --joy_stick_controller True
```
---
