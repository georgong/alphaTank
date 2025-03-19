from configs.config_basic import * # import from outside

"""-----------TEAM TRAINING SETTING-----------"""

team_configs = {
    "Tank1":{"team":"TeamA", 
            "color":RED, 
            "mode": "agent",
            },
    "Tank2":{"team":"TeamA", 
            "color":RED, 
            "mode": "agent",
        },
    "Tank3":{"team":"TeamB", 
            "color":GREEN, 
            "mode": "agent",
        },
    "Tank4":{"team":"TeamB", 
            "color":GREEN, 
            "mode": "agent",
        },
}

pair_configs = {
    "Tank1":{"team":"TeamA", 
            "color":RED, 
            "mode": "agent",
            },
    "Tank2":{"team":"TeamB", 
            "color":GREEN, 
            "mode": "agent",
        },
}

team_vs_bot_configs = {
    "Tank1":{"team":"TeamA", 
        "color":RED, 
        "mode": "agent",
        },
    "Tank2":{"team":"TeamA", 
        "color":RED, 
        "mode": "agent",
        },
    "Tank3":{"team":"TeamB", 
            "color":GREEN, 
            "mode": "bot",
            "bot_type": "smart",
        },
    "Tank4":{"team":"TeamB", 
            "color":GREEN, 
            "mode": "bot",
            "bot_type": "smart",
        },
}

team_vs_bot_hard_configs = {
    "Tank1":{"team":"TeamA", 
        "color":RED, 
        "mode": "agent",
        },
    "Tank2":{"team":"TeamA", 
        "color":RED, 
        "mode": "agent",
        },
    "Tank3":{"team":"TeamB", 
            "color":GREEN, 
            "mode": "bot",
            "bot_type": "aggressive",
        },
    "Tank4":{"team":"TeamB", 
            "color":GREEN, 
            "mode": "bot",
            "bot_type": "smart",
        },
    "Tank5":{"team":"TeamB", 
            "color":GREEN, 
            "mode": "bot",
            "bot_type": "defensive",
        },
}

bot_team_configs = {
    "Tank1": {
        "team": "TeamA",
        "color": GREEN,
        "mode": "bot",
        "bot_type": "smart",
    },
    "Tank2": {
        "team": "TeamA",
        "color": GREEN,
        "mode": "bot",
        "bot_type": "smart",
    },
    "Tank3": {
        "team": "TeamB",
        "color": RED,
        "mode": "bot",
        "bot_type": "aggressive",
    },
    "Tank4": {
        "team": "TeamB",
        "color": RED,
        "mode": "bot",
        "bot_type": "aggressive",
    },
    "Tank5": {
        "team": "TeamC",
        "color": GRAY,
        "mode": "bot",
        "bot_type": "defensive",
    },
     "Tank6": {
        "team": "TeamC",
        "color": GRAY,
        "mode": "bot",
        "bot_type": "defensive",
    },
}

mixed_team_configs = {
    "Tank1": {
        "team": "TeamA",
        "color": GREEN,
        "mode": "human",
        "keys": {
            "left": pygame.K_a,
            "right": pygame.K_d,
            "up": pygame.K_w,
            "down": pygame.K_s,
            "shoot": pygame.K_f,
        },
    },
    "Tank2": {
        "team": "TeamA",
        "color": GREEN,
        "mode": "bot",
        "bot_type": "smart",
    },
    "Tank3": {
        "team": "TeamA",
        "color": GREEN,
        "mode": "agent",
    },
    "Tank4": {
        "team": "TeamB",
        "color": RED,
        "mode": "human",
        "keys": {
            "left": pygame.K_j,
            "right": pygame.K_l,
            "up": pygame.K_i,
            "down": pygame.K_k,
            "shoot": pygame.K_p,
        },
    },
    "Tank5": {
        "team": "TeamB",
        "color": RED,
        "mode": "bot",
        "bot_type": "smart",
    },
    "Tank6": {
        "team": "TeamB",
        "color": RED,
        "mode": "agent",
    },
    "Tank7": {
        "team": "TeamC",
        "color": GRAY,
        "mode": "bot",
        "bot_type": "aggressive",
    },
     "Tank8": {
        "team": "TeamC",
        "color": GRAY,
        "mode": "bot",
        "bot_type": "smart",
    },
}

"""-----------TEAM INFERENCE SETTING-----------"""

# Need to be consistent with tarining
inference_agent_configs = {"Tank1":"checkpoints/team_ppo/ppo_agent_0.pt", "Tank2":"checkpoints/team_ppo/ppo_agent_1.pt"}
