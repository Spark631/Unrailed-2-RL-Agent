# visualize_map.py

from utils.map_gen import (
    generate_map,
    AGENT,
    TRAIN_STORAGE,
    TRAIN_CRAFTER,
    TRAIN_HEAD,
    RAILROADS,
    OBSTACLES,
    TREES,
    STONE,
    WOOD,
    METAL,
    TRACK_ITEM,
    STATION,
    PICKAXE,
    AXE,
)

def ascii_from_grid(grid):
    """
    ASCII representation from the 3D grid returned by generate_map().
    """
    H, W, C = grid.shape
    lines = []

    for r in range(H):
        row_chars = []
        for c in range(W):
            ch = "." # Default empty

            # Terrain
            if grid[r, c, OBSTACLES] == 1: ch = "#"
            elif grid[r, c, TREES] == 1: ch = "T"
            elif grid[r, c, STONE] == 1: ch = "R"
            elif grid[r, c, STATION] == 1: ch = "@"
            elif grid[r, c, RAILROADS] == 1: ch = "="

            # Items
            if grid[r, c, PICKAXE] == 1: ch = "p"
            elif grid[r, c, AXE] == 1: ch = "x"
            
            # Resources with counts
            if grid[r, c, WOOD] >= 1:
                cnt = int(grid[r, c, WOOD])
                ch = "w" if cnt == 1 else str(min(cnt, 9))
            elif grid[r, c, METAL] >= 1:
                cnt = int(grid[r, c, METAL])
                ch = "m" if cnt == 1 else str(min(cnt, 9))
            elif grid[r, c, TRACK_ITEM] >= 1:
                cnt = int(grid[r, c, TRACK_ITEM])
                ch = "r" if cnt == 1 else str(min(cnt, 9))

            # Train
            if grid[r, c, TRAIN_STORAGE] == 1: ch = "S"
            elif grid[r, c, TRAIN_CRAFTER] == 1: ch = "C"
            elif grid[r, c, TRAIN_HEAD] == 1: ch = "H"

            # Agent
            if grid[r, c, AGENT] == 1: ch = "A"

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

    grid, agent_pos, train_pos, station_pos = generate_map(config)

    print("Grid shape (H, W, C):", grid.shape)
    print()

    lines = ascii_from_grid(grid)

    print("ASCII map visualization:")
    for line in lines:
        print(line)

    print("\nLegend:")
    print("  .  = empty ground")
    print("  #  = obstacle (terrain)")
    print("  T  = tree (terrain)")
    print("  R  = rock block (terrain)")
    print("  @  = station (terrain)")
    print("  =  = laid track (RAILROADS, train path)")
    print("  p  = pickaxe on floor")
    print("  x  = axe on floor")
    print("  w  = 1 wood on floor (2-9 = count)")
    print("  m  = 1 metal on floor (2-9 = count)")
    print("  r  = 1 rail/track item on floor (2-9 = count)")
    print("  S  = storage wagon (train)")
    print("  C  = crafter wagon (train)")
    print("  H  = head/engine wagon (train)")
    print("  A  = agent\n")


if __name__ == "__main__":
    main()
