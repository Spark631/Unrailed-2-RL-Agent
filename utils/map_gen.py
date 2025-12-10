from __future__ import annotations
import numpy as np
from collections import deque
from typing import Dict, Any, Tuple, List

# ---------------------------------------------------------------------
# Channel indices (encoded 3D grid: H, W, C)
# ---------------------------------------------------------------------
AGENT          = 0
TRAIN_STORAGE  = 1   # storage wagon (2 tiles)
TRAIN_CRAFTER  = 2   # crafter wagon (2 tiles)
TRAIN_HEAD     = 3   # head/engine (3 tiles)
RAILROADS      = 4   # laid tracks (train path, non-pickable)
OBSTACLES      = 5
TREES          = 6   # terrain: tree block
STONE          = 7   # terrain: rock block
WOOD           = 8   # floor resource: loose wood (from trees)
METAL          = 9   # floor resource: loose metal (from rocks)
TRACK_ITEM     = 10  # floor resource: loose tracks (craft result)
STATION        = 11
PICKAXE        = 12  # floor item
AXE            = 13  # floor item

NUM_CHANNELS = 14

# ---------------------------------------------------------------------
# Internal terrain + floor enums
# ---------------------------------------------------------------------
TERRAIN_EMPTY = -1

FLOOR_NONE     = 0
FLOOR_WOOD     = 1
FLOOR_METAL    = 2
FLOOR_TRACK    = 3
FLOOR_PICKAXE  = 4
FLOOR_AXE      = 5


# =====================================================================
# Public API
# =====================================================================

def generate_map(config: Dict[str, Any]):
    """
    Generate a 4x16 map with:
      - terrain grid: trees / rocks / obstacles / station / laid track
      - floor grid: one of {wood, metal, track_item, pickaxe, axe, none}
      - stacked resources: wood/metal/track counts on their tiles (1..3)
      - agent + train row, where train is split into:
          storage (2 tiles), crafter (2 tiles), head (3 tiles),
        and a pre-placed laid track tile in front (col 7).
      - guaranteed logical path through the 4x5 middle region:
          path cells may have trees/rocks, but NEVER obstacles.
    """

    H, W = 4, 16
    rng = np.random.default_rng(config.get("seed", None))

    # terrain distribution in middle region
    p_obstacle = config.get("p_obstacle", 0.15)
    p_tree     = config.get("p_tree", 0.10)
    p_rock     = config.get("p_rock", 0.10)
    # remainder = empty

    # ------------------------------------------------------------------
    # 1. Internal grids
    # ------------------------------------------------------------------
    terrain = np.full((H, W), TERRAIN_EMPTY, dtype=np.int32)

    floor_type  = np.full((H, W), FLOOR_NONE, dtype=np.int8)
    floor_count = np.zeros((H, W), dtype=np.int8)  # for wood/metal/track stacks

    # ------------------------------------------------------------------
    # 2. Train row & tools row
    # ------------------------------------------------------------------
    train_row = int(rng.integers(0, H))
    tools_row = int(rng.choice([r for r in range(H) if r != train_row]))

    # train split into: storage (2), crafter (2), head (3), plus 1 laid track
    storage_cols = list(range(0, 2))  # [0,1]
    crafter_cols = list(range(2, 4))  # [2,3]
    head_cols    = list(range(4, 7))  # [4,5,6]
    track_col    = 7                  # [7] = pre-placed laid track

    train_length = len(storage_cols) + len(crafter_cols) + len(head_cols)  # = 7
    train_front_pos = (train_row, storage_cols[0])

    # pre-placed laid track ahead of the head
    terrain[train_row, track_col] = RAILROADS

    # ------------------------------------------------------------------
    # 3. Station: 3x3 at top or bottom in cols 13..15
    # ------------------------------------------------------------------
    station_top_row = int(rng.choice([0, 1]))  # rows 0..2 or 1..3
    for r in range(station_top_row, station_top_row + 3):
        for c in range(13, 16):
            terrain[r, c] = STATION

    station_center_row = station_top_row + 1
    station_entry = (station_center_row, 12)  # middle-left of station

    # ------------------------------------------------------------------
    # 4. Path in 4x5 middle area (rows 0..3, cols 8..12)
    # ------------------------------------------------------------------
    start = (train_row, 8)
    goal = station_entry
    path = _find_path_in_middle(start, goal, rng)
    path_set = set(path)

    # ------------------------------------------------------------------
    # 5. Fill middle terrain:
    #    - On path cells: EMPTY / TREES / STONE only (no OBSTACLES)
    #    - Off path: full distribution (can include OBSTACLES)
    # ------------------------------------------------------------------
    for r in range(H):
        for c in range(8, 13):
            if (r, c) in path_set:
                # sample until we get something that is NOT an obstacle
                while True:
                    x = rng.random()
                    if x < p_obstacle:
                        t = OBSTACLES
                    elif x < p_obstacle + p_tree:
                        t = TREES
                    elif x < p_obstacle + p_tree + p_rock:
                        t = STONE
                    else:
                        t = TERRAIN_EMPTY

                    if t != OBSTACLES:
                        terrain[r, c] = t
                        break
            else:
                # off-path: full distribution including obstacles
                x = rng.random()
                if x < p_obstacle:
                    terrain[r, c] = OBSTACLES
                elif x < p_obstacle + p_tree:
                    terrain[r, c] = TREES
                elif x < p_obstacle + p_tree + p_rock:
                    terrain[r, c] = STONE
                else:
                    terrain[r, c] = TERRAIN_EMPTY

    # ------------------------------------------------------------------
    # 6. Tools + starting resources + agent on tools_row
    #    Layout (all inside cols 0..7):
    #      [AGENT][PICKAXE][AXE][WOOD-stack][METAL-stack]
    # ------------------------------------------------------------------
    segment_len = 5
    # must fit entirely in cols 0..7 → base_col in [0, 8 - segment_len] = [0,3]
    base_col = int(rng.integers(0, 8 - segment_len + 1))  # 0..3

    col_agent   = base_col
    col_pickaxe = base_col + 1
    col_axe     = base_col + 2
    col_wood    = base_col + 3
    col_metal   = base_col + 4

    # underlying terrain is empty for these tiles
    for c in [col_agent, col_pickaxe, col_axe, col_wood, col_metal]:
        terrain[tools_row, c] = TERRAIN_EMPTY

    # agent (overlay only)
    agent_pos = (tools_row, col_agent)

    # floor items
    floor_type[tools_row, col_pickaxe]  = FLOOR_PICKAXE
    floor_count[tools_row, col_pickaxe] = 1

    floor_type[tools_row, col_axe]  = FLOOR_AXE
    floor_count[tools_row, col_axe] = 1

    # 3 wood stacked on ONE tile
    floor_type[tools_row, col_wood]  = FLOOR_WOOD
    floor_count[tools_row, col_wood] = 3

    # 3 metal stacked on the NEXT tile
    floor_type[tools_row, col_metal]  = FLOOR_METAL
    floor_count[tools_row, col_metal] = 3

    # (Initial loose TRACK_ITEM stacks are 0 everywhere in this version.)

    # ------------------------------------------------------------------
    # 7. Encode into (H, W, C) grid
    # ------------------------------------------------------------------
    grid = _encode_grid(
        terrain=terrain,
        floor_type=floor_type,
        floor_count=floor_count,
        agent_pos=agent_pos,
        train_row=train_row,
        storage_cols=storage_cols,
        crafter_cols=crafter_cols,
        head_cols=head_cols,
        track_col=track_col,
    )

    # ------------------------------------------------------------------
    # 8. Return essential state for Env
    # ------------------------------------------------------------------
    # Env expects: grid, agent_pos, train_pos, station_pos
    # train_pos is (row, first_head_col)
    train_pos = (train_row, head_cols[0])

    return grid, agent_pos, train_pos, station_entry


# =====================================================================
# Pathfinding in 4x5 middle area
# =====================================================================

def _find_path_in_middle(
    start: Tuple[int, int],
    goal: Tuple[int, int],
    rng: np.random.Generator,
) -> List[Tuple[int, int]]:
    """
    BFS in 4x5 middle region (rows 0..3, cols 8..12).
    """
    H_min, H_max = 0, 3
    C_min, C_max = 8, 12

    sr, sc = start
    gr, gc = goal

    sr = int(np.clip(sr, H_min, H_max))
    sc = int(np.clip(sc, C_min, C_max))
    start = (sr, sc)

    q = deque([start])
    parent = {start: None}
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    while q:
        r, c = q.popleft()
        if (r, c) == (gr, gc):
            break

        for dr, dc in rng.permutation(dirs):
            nr, nc = r + dr, c + dc
            if not (H_min <= nr <= H_max and C_min <= nc <= C_max):
                continue
            if (nr, nc) in parent:
                continue
            parent[(nr, nc)] = (r, c)
            q.append((nr, nc))

    if (gr, gc) not in parent:
        # fallback: simple L-shaped path
        path: List[Tuple[int, int]] = []
        r, c = start
        path.append((r, c))
        while c != gc:
            c += 1 if gc > c else -1
            path.append((r, c))
        while r != gr:
            r += 1 if gr > r else -1
            path.append((r, c))
        return path

    # reconstruct
    path: List[Tuple[int, int]] = []
    cur = (gr, gc)
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path


# =====================================================================
# Encoding terrain + floor + overlays into (H, W, C)
# =====================================================================

def _encode_grid(
    terrain: np.ndarray,
    floor_type: np.ndarray,
    floor_count: np.ndarray,
    agent_pos: Tuple[int, int],
    train_row: int,
    storage_cols: List[int],
    crafter_cols: List[int],
    head_cols: List[int],
    track_col: int,
) -> np.ndarray:
    """
    Build grid of shape (H, W, C).

    - terrain: TERRAIN_EMPTY or one of {RAILROADS, OBSTACLES, TREES, STONE, STATION}
    - floor_type: FLOOR_*; wood/metal/track stacks live here
    - floor_count: for WOOD/METAL/TRACK tiles, 1..3 (tools ignore count)
    - train segments encoded in three separate channels:
        TRAIN_STORAGE, TRAIN_CRAFTER, TRAIN_HEAD
    """
    H, W = terrain.shape
    grid = np.zeros((H, W, NUM_CHANNELS), dtype=np.float32)

    # terrain layer (including initial laid track)
    for r in range(H):
        for c in range(W):
            t = terrain[r, c]
            if t == TERRAIN_EMPTY:
                continue
            grid[r, c, t] = 1.0

    # floor layer
    for r in range(H):
        for c in range(W):
            ftype = floor_type[r, c]
            if ftype == FLOOR_NONE:
                continue

            if ftype == FLOOR_WOOD:
                grid[r, c, WOOD] = float(max(1, floor_count[r, c]))
            elif ftype == FLOOR_METAL:
                grid[r, c, METAL] = float(max(1, floor_count[r, c]))
            elif ftype == FLOOR_TRACK:
                grid[r, c, TRACK_ITEM] = float(max(1, floor_count[r, c]))
            elif ftype == FLOOR_PICKAXE:
                grid[r, c, PICKAXE] = 1.0
            elif ftype == FLOOR_AXE:
                grid[r, c, AXE] = 1.0

    # agent overlay
    ar, ac = agent_pos
    grid[ar, ac, AGENT] = 1.0

    # train segments
    for c in storage_cols:
        grid[train_row, c, TRAIN_STORAGE] = 1.0
    for c in crafter_cols:
        grid[train_row, c, TRAIN_CRAFTER] = 1.0
    for c in head_cols:
        grid[train_row, c, TRAIN_HEAD] = 1.0

    # (track_col is already encoded as RAILROADS via terrain)

    return grid
