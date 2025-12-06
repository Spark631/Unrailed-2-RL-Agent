# visualize_map.py

from utils.map_gen import (
    generate_map,
    OBSTACLES,
    TREES,
    STONE,
    STATION,
    RAILROADS,
    TERRAIN_EMPTY,
    FLOOR_NONE,
    FLOOR_WOOD,
    FLOOR_METAL,
    FLOOR_TRACK,
    FLOOR_PICKAXE,
    FLOOR_AXE,
)

def ascii_from_state(state):
    """
    ASCII representation from the internal `state` dict returned by generate_map().
    """
    terrain     = state["terrain"]
    floor_type  = state["floor_type"]
    floor_count = state["floor_count"]

    H, W = terrain.shape

    agent_r, agent_c = state["agent_pos"]
    train_row        = state["train_row"]
    storage_cols     = state["train_storage_cols"]
    crafter_cols     = state["train_crafter_cols"]
    head_cols        = state["train_head_cols"]
    track_col        = state["initial_track_col"]

    lines = []

    for r in range(H):
        row_chars = []
        for c in range(W):

            # --- base terrain ---
            t = terrain[r, c]
            if t == TERRAIN_EMPTY:
                ch = "."
            elif t == OBSTACLES:
                ch = "#"
            elif t == TREES:
                ch = "Y"   # tree
            elif t == STONE:
                ch = "R"   # rock block
            elif t == STATION:
                ch = "@"
            elif t == RAILROADS:
                ch = "-"   # laid track
            else:
                ch = "?"   # unexpected / debug

            # --- floor items (only one per tile) ---
            ftype = floor_type[r, c]
            fcnt  = floor_count[r, c]

            if ftype != FLOOR_NONE:
                if ftype == FLOOR_WOOD:
                    ch = "w" if fcnt <= 1 else str(min(fcnt, 9))
                elif ftype == FLOOR_METAL:
                    ch = "m" if fcnt <= 1 else str(min(fcnt, 9))
                elif ftype == FLOOR_TRACK:
                    ch = "t" if fcnt <= 1 else str(min(fcnt, 9))
                elif ftype == FLOOR_PICKAXE:
                    ch = "P"
                elif ftype == FLOOR_AXE:
                    ch = "X"

            # --- train overlay (storage / crafter / head) ---
            if r == train_row:
                if c in storage_cols:
                    ch = "S"   # storage wagon
                elif c in crafter_cols:
                    ch = "C"   # crafter wagon
                elif c in head_cols:
                    ch = "H"   # head / engine

            # --- agent overlay (highest priority) ---
            if (r, c) == (agent_r, agent_c):
                ch = "A"

            row_chars.append(ch)
        lines.append("".join(row_chars))

    return lines


def main():
    config = {
        "seed": None,
        "p_obstacle": 0.15,
        "p_tree": 0.4,
        "p_rock": 0.25,
    }

    grid, state = generate_map(config)

    print("Grid shape (H, W, C):", grid.shape)
    print()

    lines = ascii_from_state(state)

    print("ASCII map visualization:")
    for line in lines:
        print(line)

    print("\nLegend:")
    print("  .  = empty ground")
    print("  #  = obstacle (terrain)")
    print("  Y  = tree (terrain)")
    print("  R  = rock block (terrain)")
    print("  @  = station (terrain)")
    print("  -  = laid track (RAILROADS, train path)")
    print("  P  = pickaxe on floor")
    print("  X  = axe on floor")
    print("  w  = 1 wood on floor (2–9 = count)")
    print("  m  = 1 metal on floor (2–9 = count)")
    print("  t  = 1 track item on floor (2–9 = count)")
    print("  S  = storage wagon (train)")
    print("  C  = crafter wagon (train)")
    print("  H  = head/engine wagon (train)")
    print("  A  = agent\n")


if __name__ == "__main__":
    main()
