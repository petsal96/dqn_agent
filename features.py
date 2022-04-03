import torch   
import numpy as np   
import copy

def calc_features(game_state: dict):
    
    # useful vars:
    agent_pos = game_state["self"][3]
    arena = game_state["field"].copy()
    coins = game_state["coins"]
    bombs = game_state["bombs"]
    others = game_state["others"]
    crates = []
    for x in range(arena.shape[0]):
        for y in range(arena.shape[1]):
            crates.append((x, y))

    # self map:
    self_map = torch.zeros(arena.shape).float()
    self_map[agent_pos[0], agent_pos[1]] = 1

    # wall map:
    wall_map = np.zeros(arena.shape)
    wall_map[arena == -1] = 1
    wall_map = torch.from_numpy(wall_map).float()

    # coin map:
    coin_map = torch.zeros(arena.shape).float()
    for coin_x, coin_y in coins:
        coin_map[coin_x, coin_y] = 1

    # explosion map:
    explosion_map = torch.from_numpy(game_state["explosion_map"].copy()).float()

    # bomb timers:
    bomb_timers = torch.zeros(arena.shape).float()
    for (bomb_x, bomb_y), bomb_t in bombs:
        bomb_timers[bomb_x, bomb_y] = (bomb_t+1)/4

    # others:
    others_map = torch.zeros(arena.shape).float()
    for other in others:
        others_map[other[3][0], other[3][1]] = 1


    return torch.stack([self_map, wall_map, coin_map, explosion_map, bomb_timers, others_map])
