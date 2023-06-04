from gym_hierarchical_subgoal_automata.automata.common import get_param
from gym_minigrid.minigrid import Grid, MiniGridEnv, Chicken, Cow, Door, Iron, Lava, Rabbit, Redstone, Squid, Sugarcane, Table, Wall, Wheat, Workbench, COLOR_NAMES
import math
import numpy as np


class BaseRoom:
    def __init__(self, top, size):
        # top-left corner and size (tuples)
        self.top = top
        self.size = size


class ChainRoom(BaseRoom):
    def __init__(self, top, size, entry_door_pos, exit_door_pos):
        super().__init__(top, size)
        self.entry_door_pos = entry_door_pos
        self.exit_door_pos = exit_door_pos


class CorridorRoom(BaseRoom):
    def __init__(self, top, size):
        super().__init__(top, size)

        # list of door objects and door positions (order of the doors is right, down, left, up)
        self.doors = [None] * 4
        self.door_pos = [None] * 4

        # list of rooms adjacent to this one (order of the neighbors is right, down, left, up)
        self.neighbors = [None] * 4

        # indicates if this room is behind a locked door
        self.locked = False


class CustomMiniGrid(MiniGridEnv):
    AGENT_VIEW_SIZE = "agent_view_size"
    SEE_THROUGH_WALLS = "see_through_walls"
    USE_SHARED_COLOR = "use_shared_color"  # all custom objects share the same color
    USE_LAVA_WALLS = "use_lava_walls"      # use lava walls instead of normal walls

    USE_LAVA = "use_lava"  # whether to use extra lava locations in the grid
    NUM_LAVA = "num_lava"  # how many lava locations to generate if the above is true (this is only used in some cases)

    GRID_PARAMS = "grid_params"
    GRID_TYPE = "grid_type"
    GRID_DOOR_STATE = "door_state"

    GRID_TYPE_OPEN_PLAN = "open_plan"   # a single room without walls
    GRID_HEIGHT = "height"
    GRID_WIDTH = "width"

    GRID_TYPE_CORRIDOR = "corridor"     # a corridor connected to two columns of interconnected rooms
    GRID_NUM_ROWS = "num_rows"
    GRID_NUM_COLS = "num_cols"
    GRID_ROOM_SIZE = "room_size"

    GRID_TYPE_ROOM_CHAIN = "multi_room"  # a succession of rooms connected with each other
    GRID_MIN_NUM_ROOMS = "min_num_rooms"
    GRID_MAX_NUM_ROOMS = "max_num_rooms"
    GRID_MAX_ROOM_SIZE = "max_room_size"

    GRID_TYPE_FOUR_ROOMS = "four_rooms"
    GRID_SIZE = "size"
    GRID_RIGHT_ROOMS_EVEN = "right_rooms_even"
    GRID_LAVA_LOCS = "lava_locations"
    GRID_LAVA_LOCS_DOOR_INTERSECTIONS = "door_intersections"
    GRID_LAVA_LOCS_ALL_CORNERS = "all_corners"
    GRID_LAVA_LOCS_SOME_CORNERS = "some_corners"

    GRID_TYPE_NINE_ROOMS = "nine_rooms"

    NUM_DIRECTIONS = 4

    MAX_NUM_OBJS_PER_CLASS = "max_objs_per_class"
    OBJ_CLASSES = [Iron, Table, Cow, Sugarcane, Wheat, Chicken, Redstone, Rabbit, Squid, Workbench]

    def __init__(self, params, seed):
        self.grid_params = get_param(params, CustomMiniGrid.GRID_PARAMS)
        self.use_shared_color = get_param(params, CustomMiniGrid.USE_SHARED_COLOR, False)
        self.use_lava_walls = get_param(self.grid_params, CustomMiniGrid.USE_LAVA_WALLS, False)
        self.use_lava = get_param(self.grid_params, CustomMiniGrid.USE_LAVA, False)

        width, height = self._get_grid_size()
        super().__init__(
            agent_view_size=get_param(params, CustomMiniGrid.AGENT_VIEW_SIZE, 7),
            height=height,
            max_steps=math.inf,  # the learning algorithm is the responsible for bounding this
            seed=seed,
            see_through_walls=get_param(params, CustomMiniGrid.SEE_THROUGH_WALLS, False),
            width=width
        )

    def contains_lava(self):
        return self.use_lava_walls or self.use_lava

    def gen_obs(self):
        """
        The original method is overidden to return a dummy observation to avoid spending useless time forming an
        egocentric observation that is not going to be used.
        """
        return {
            'image': None,
            'direction': self.agent_dir,
            'mission': self.mission
        }

    def _get_grid_size(self):
        grid_type = get_param(self.grid_params, CustomMiniGrid.GRID_TYPE)
        if grid_type == CustomMiniGrid.GRID_TYPE_OPEN_PLAN:
            return get_param(self.grid_params, CustomMiniGrid.GRID_WIDTH), \
                   get_param(self.grid_params, CustomMiniGrid.GRID_HEIGHT)
        elif grid_type == CustomMiniGrid.GRID_TYPE_CORRIDOR:
            room_size = get_param(self.grid_params, CustomMiniGrid.GRID_ROOM_SIZE)
            num_rows = get_param(self.grid_params, CustomMiniGrid.GRID_NUM_ROWS)
            num_cols = get_param(self.grid_params, CustomMiniGrid.GRID_NUM_COLS)
            return (room_size - 1) * num_cols + 1, (room_size - 1) * num_rows + 1
        elif grid_type == CustomMiniGrid.GRID_TYPE_ROOM_CHAIN:
            return 25, 25
        elif grid_type == CustomMiniGrid.GRID_TYPE_FOUR_ROOMS:
            grid_size = get_param(self.grid_params, CustomMiniGrid.GRID_SIZE)
            if grid_size is None or grid_size < 11 or grid_size % 2 == 0:
                raise RuntimeError("Error: The grid for four rooms must be at least 11x11 and the size must be an odd "
                                   "number (i.e., 11, 13, ...).")
            return grid_size, grid_size
        elif grid_type == CustomMiniGrid.GRID_TYPE_NINE_ROOMS:
            return 13, 13
        else:
            raise RuntimeError(f"Error: Unknown grid type {grid_type}.")

    def reset(self):
        self.mission = ""
        return super().reset()

    def get_current_object(self):
        return self.grid.get(*self.agent_pos)

    def put_agent(self, pos, dir):
        # Important: This is only used to help plotting the value functions.
        self.agent_pos = pos
        self.agent_dir = dir

    def _gen_grid(self, width, height):
        grid_type = get_param(self.grid_params, CustomMiniGrid.GRID_TYPE)
        if grid_type == CustomMiniGrid.GRID_TYPE_OPEN_PLAN:
            self._gen_open_plan_grid()
        elif grid_type == CustomMiniGrid.GRID_TYPE_CORRIDOR:
            self._gen_corridor_grid()
        elif grid_type == CustomMiniGrid.GRID_TYPE_ROOM_CHAIN:
            self._gen_room_chain()
        elif grid_type == CustomMiniGrid.GRID_TYPE_FOUR_ROOMS:
            self._gen_four_rooms()
        elif grid_type == CustomMiniGrid.GRID_TYPE_NINE_ROOMS:
            self._gen_nine_rooms()
        else:
            raise RuntimeError(f"Error: Unknown grid type {grid_type}.")

    '''
    Helper methods for generation
    '''
    def _place_agent_and_objects(self, room_list, add_lava):
        agent_room = self._rand_room_idx(room_list)
        self.place_agent(room_list[agent_room].top, room_list[agent_room].size)

        for obj_class in CustomMiniGrid.OBJ_CLASSES:
            quantity = self._rand_int(1, get_param(self.grid_params, CustomMiniGrid.MAX_NUM_OBJS_PER_CLASS, 1) + 1)
            self._place_obj_in_rand_room(room_list, obj_class, quantity)

        if add_lava:
            self._place_obj_in_rand_room(room_list, Lava, get_param(self.grid_params, CustomMiniGrid.NUM_LAVA, 1))

    def _place_obj_in_rand_room(self, room_list, obj_class, quantity):
        for _ in range(quantity):
            rand_room = self._rand_room_idx(room_list)
            if self.use_shared_color and obj_class != Lava:  # lava already has a fixed color
                obj = obj_class('red')
            else:  # just call the empty constructor (the default color object -one per object- is there)
                obj = obj_class()
            self.place_obj(obj, room_list[rand_room].top, room_list[rand_room].size)

    def _rand_room_idx(self, room_list):
        return self._rand_int(0, len(room_list))

    '''
    Open plan grid generation
    '''
    def _gen_open_plan_grid(self):
        """
        Generates a grid of the size specified by self.width and self.height without walls (the only walls are those in
        the boundaries). The objects and the agent are placed in random positions and do not share positions. This
        method is based on the one for GymMinigrid's Empty environment.
        """
        self.grid = Grid(self.width, self.height)           # create an empty grid
        self.grid.wall_rect(0, 0, self.width, self.height)  # generate the surrounding walls
        self._place_agent_and_objects([ChainRoom((0, 0), (self.width, self.height), None, None)], self.use_lava)

    '''
    Corridor grid generation
    '''
    def _gen_corridor_grid(self):
        room_size = get_param(self.grid_params, CustomMiniGrid.GRID_ROOM_SIZE)
        num_rows = get_param(self.grid_params, CustomMiniGrid.GRID_NUM_ROWS)
        num_cols = get_param(self.grid_params, CustomMiniGrid.GRID_NUM_COLS)

        self.grid = Grid(self.width, self.height)
        room_grid = []

        for j in range(0, num_rows):  # for each row of rooms
            row = []
            for i in range(0, num_cols):  # for each column of rooms
                room = CorridorRoom(
                    (i * (room_size - 1), j * (room_size - 1)),
                    (room_size, room_size)
                )
                row.append(room)

                # generate the walls for this room
                self.grid.wall_rect(*room.top, *room.size)
            room_grid.append(row)

        for j in range(0, num_rows):  # for each row of rooms
            for i in range(0, num_cols):  # for each column of rooms
                room = room_grid[j][i]

                x_l, y_l = (room.top[0] + 1, room.top[1] + 1)
                x_m, y_m = (room.top[0] + room.size[0] - 1, room.top[1] + room.size[1] - 1)

                # door positions, order is right, down, left, up
                if i < num_cols - 1:
                    room.neighbors[0] = room_grid[j][i+1]
                    room.door_pos[0] = (x_m, self._rand_int(y_l, y_m))
                if j < num_rows - 1:
                    room.neighbors[1] = room_grid[j+1][i]
                    room.door_pos[1] = (self._rand_int(x_l, x_m), y_m)
                if i > 0:
                    room.neighbors[2] = room_grid[j][i-1]
                    room.door_pos[2] = room.neighbors[2].door_pos[0]
                if j > 0:
                    room.neighbors[3] = room_grid[j-1][i]
                    room.door_pos[3] = room.neighbors[3].door_pos[1]

        # connect the middle column rooms into a hallway
        for j in range(1, num_rows):
            self._remove_wall(1, j, 3, num_rows, num_cols, room_grid)

        room_list = sum(room_grid, [])
        self._place_agent_and_objects(room_list, self.use_lava)

        # make sure all rooms are accessible
        self._connect_all(num_rows, num_cols, room_size, room_grid)

    def _remove_wall(self, i, j, wall_idx, num_rows, num_cols, room_grid):
        """
        Remove a wall between two rooms
        """
        room = self._get_room_from_grid(i, j, num_rows, num_cols, room_grid)

        assert wall_idx >= 0 and wall_idx < 4
        assert room.doors[wall_idx] is None, "door exists on this wall"
        assert room.neighbors[wall_idx], "invalid wall"

        neighbor = room.neighbors[wall_idx]

        tx, ty = room.top
        w, h = room.size

        # Ordering of walls is right, down, left, up
        if wall_idx == 0:
            for i in range(1, h - 1):
                self.grid.set(tx + w - 1, ty + i, None)
        elif wall_idx == 1:
            for i in range(1, w - 1):
                self.grid.set(tx + i, ty + h - 1, None)
        elif wall_idx == 2:
            for i in range(1, h - 1):
                self.grid.set(tx, ty + i, None)
        elif wall_idx == 3:
            for i in range(1, w - 1):
                self.grid.set(tx + i, ty, None)
        else:
            assert False, "invalid wall index"

        # Mark the rooms as connected
        room.doors[wall_idx] = True
        neighbor.doors[(wall_idx+2) % 4] = True

    def _get_room_from_grid(self, i, j, num_rows, num_cols, room_grid):
        assert i < num_cols
        assert j < num_rows
        return room_grid[j][i]

    def _get_grid_room_from_pos(self, x, y, num_rows, num_cols, room_size, room_grid):
        """Get the room a given position maps to"""
        assert x >= 0
        assert y >= 0

        i = x // (room_size - 1)
        j = y // (room_size - 1)

        assert i < num_cols
        assert j < num_rows

        return room_grid[j][i]

    def _connect_all(self, num_rows, num_cols, room_size, room_grid, door_colors=COLOR_NAMES, max_itrs=5000):
        """
        Make sure that all rooms are reachable by the agent from its
        starting position
        """
        start_room = self._get_grid_room_from_pos(*self.agent_pos, num_rows, num_cols, room_size, room_grid)

        added_doors = []

        def find_reach():
            reach = set()
            stack = [start_room]
            while len(stack) > 0:
                room = stack.pop()
                if room in reach:
                    continue
                reach.add(room)
                for i in range(0, 4):
                    if room.doors[i]:
                        stack.append(room.neighbors[i])
            return reach

        num_itrs = 0

        while True:
            # This is to handle rare situations where random sampling produces
            # a level that cannot be connected, producing in an infinite loop
            if num_itrs > max_itrs:
                raise RecursionError('_connect_all failed')
            num_itrs += 1

            # If all rooms are reachable, stop
            reach = find_reach()
            if len(reach) == num_rows * num_cols:
                break

            # Pick a random room and door position
            i = self._rand_int(0, num_cols)
            j = self._rand_int(0, num_rows)
            k = self._rand_int(0, 4)
            room = self._get_room_from_grid(i, j, num_rows, num_cols, room_grid)

            # If there is already a door there, skip
            if not room.door_pos[k] or room.doors[k]:
                continue

            if room.locked or room.neighbors[k].locked:
                continue

            color = self._rand_elem(door_colors)
            door, _ = self._add_door(i, j, num_rows, num_cols, room_grid, door_idx=k, color=color, locked=False)
            added_doors.append(door)

        return added_doors

    def _add_door(self, i, j, num_rows, num_cols, room_grid, door_idx=None, color=None, locked=None):
        """
        Add a door to a room, connecting it to a neighbor
        """
        room = self._get_room_from_grid(i, j, num_rows, num_cols, room_grid)

        if door_idx == None:
            # Need to make sure that there is a neighbor along this wall
            # and that there is not already a door
            while True:
                door_idx = self._rand_int(0, 4)
                if room.neighbors[door_idx] and room.doors[door_idx] is None:
                    break

        if color == None:
            color = self._rand_color()

        if locked is None:
            locked = self._rand_bool()

        assert room.doors[door_idx] is None, "door already exists"

        room.locked = locked
        door = Door(color, True, is_locked=locked)

        pos = room.door_pos[door_idx]
        self.grid.set(*pos, door)
        door.cur_pos = pos

        neighbor = room.neighbors[door_idx]
        room.doors[door_idx] = door
        neighbor.doors[(door_idx + 2) % 4] = door

        return door, pos

    '''
    Chain of rooms grid generation
    '''
    def _gen_room_chain(self):
        """
        Generates a chain of interconnected rooms. The objects and the agent are placed in random rooms and in random
        positions within them such that they do not share locations. This method is based on the one for GymMinigrid's
        Multiroom environment.
        """
        min_num_rooms = get_param(self.grid_params, CustomMiniGrid.GRID_MIN_NUM_ROOMS)
        max_num_rooms = get_param(self.grid_params, CustomMiniGrid.GRID_MAX_NUM_ROOMS)
        max_room_size = get_param(self.grid_params, CustomMiniGrid.GRID_MAX_ROOM_SIZE)

        assert min_num_rooms > 0
        assert max_num_rooms >= min_num_rooms
        assert max_room_size >= 4

        room_list = []
        num_rooms = self._rand_int(min_num_rooms, max_num_rooms + 1)

        while len(room_list) < num_rooms:
            curr_room_list = []
            entry_door_pos = (self._rand_int(0, self.width - 2), self._rand_int(0, self.width - 2))
            self._place_room(num_rooms, curr_room_list, 4, max_room_size, 2, entry_door_pos)
            if len(curr_room_list) > len(room_list):
                room_list = curr_room_list

        # store the list of rooms in this environment
        assert len(room_list) > 0
        self.rooms = room_list

        # create the grid
        self.grid = Grid(self.width, self.height)
        wall = Wall()

        prev_door_color = None

        # for each room
        for idx, room in enumerate(room_list):
            top_x, top_y = room.top
            size_x, size_y = room.size

            # draw the top and bottom walls
            for i in range(0, size_x):
                self.grid.set(top_x + i, top_y, wall)
                self.grid.set(top_x + i, top_y + size_y - 1, wall)

            # draw the left and right walls
            for j in range(0, size_y):
                self.grid.set(top_x, top_y + j, wall)
                self.grid.set(top_x + size_x - 1, top_y + j, wall)

            # if this isn't the first room, place the entry door
            if idx > 0:
                # pick a door color different from the previous one
                door_colors = set(COLOR_NAMES)
                if prev_door_color:
                    door_colors.remove(prev_door_color)
                # Note: the use of sorting here guarantees determinism,
                # this is needed because Python's set is not deterministic
                door_color = self._rand_elem(sorted(door_colors))

                entry_door = Door(door_color, get_param(self.grid_params, CustomMiniGrid.GRID_DOOR_STATE, True))
                self.grid.set(*room.entry_door_pos, entry_door)
                prev_door_color = door_color

                prev_room = room_list[idx - 1]
                prev_room.exit_door_pos = room.entry_door_pos

        self._place_agent_and_objects(room_list, self.use_lava)

    def _place_room(self, num_left, room_list, min_size, max_size, entry_door_wall, entry_door_pos):
        # choose the room size randomly
        size_x = self._rand_int(min_size, max_size + 1)
        size_y = self._rand_int(min_size, max_size + 1)

        # the first room will be at the door position
        if len(room_list) == 0:
            top_x, top_y = entry_door_pos
        elif entry_door_wall == 0:  # entry on the right
            top_x = entry_door_pos[0] - size_x + 1
            y = entry_door_pos[1]
            top_y = self._rand_int(y - size_y + 2, y)
        elif entry_door_wall == 1:  # entry wall on the south
            x = entry_door_pos[0]
            top_x = self._rand_int(x - size_x + 2, x)
            top_y = entry_door_pos[1] - size_y + 1
        elif entry_door_wall == 2:  # entry wall on the left
            top_x = entry_door_pos[0]
            y = entry_door_pos[1]
            top_y = self._rand_int(y - size_y + 2, y)
        elif entry_door_wall == 3:  # entry wall on the top
            x = entry_door_pos[0]
            top_x = self._rand_int(x - size_x + 2, x)
            top_y = entry_door_pos[1]
        else:
            assert False, entry_door_wall

        # if the room is out of the grid, can't place a room here
        if top_x < 0 or top_y < 0:
            return False
        if top_x + size_x > self.width or top_y + size_y >= self.height:
            return False

        # if the room intersects with previous rooms, can't place it here
        for room in room_list[:-1]:
            non_overlap = \
                top_x + size_x < room.top[0] or \
                room.top[0] + room.size[0] <= top_x or \
                top_y + size_y < room.top[1] or \
                room.top[1] + room.size[1] <= top_y

            if not non_overlap:
                return False

        # add this room to the list
        room_list.append(ChainRoom((top_x, top_y), (size_x, size_y), entry_door_pos, None))

        # if this was the last room, stop
        if num_left == 1:
            return True

        # try placing the next room
        for i in range(0, 8):
            # pick which wall to place the out door on
            wall_set = {0, 1, 2, 3}
            wall_set.remove(entry_door_wall)
            exit_door_wall = self._rand_elem(sorted(wall_set))
            next_entry_wall = (exit_door_wall + 2) % 4

            # pick the exit door position
            if exit_door_wall == 0:  # exit on right wall
                exit_door_pos = (top_x + size_x - 1,
                                 top_y + self._rand_int(1, size_y - 1))
            elif exit_door_wall == 1:  # exit on south wall
                exit_door_pos = (top_x + self._rand_int(1, size_x - 1),
                                 top_y + size_y - 1)
            elif exit_door_wall == 2:  # exit on left wall
                exit_door_pos = (top_x,
                                 top_y + self._rand_int(1, size_y - 1))
            elif exit_door_wall == 3:  # exit on north wall
                exit_door_pos = (top_x + self._rand_int(1, size_x - 1),
                                 top_y)
            else:
                assert False

            # recursively create the other rooms
            success = self._place_room(num_left - 1, room_list, min_size, max_size, next_entry_wall, exit_door_pos)
            if success:
                break

        return True

    '''
    Four rooms generation
    '''
    def _gen_four_rooms(self):
        self.grid = Grid(self.width, self.height)

        if self.use_lava_walls:
            wall_type = Lava
        else:
            wall_type = Wall

        # Generate the surrounding walls
        self.grid.horz_wall(0, 0, obj_type=wall_type)
        self.grid.horz_wall(0, self.height - 1, obj_type=wall_type)
        self.grid.vert_wall(0, 0, obj_type=wall_type)
        self.grid.vert_wall(self.width - 1, 0, obj_type=wall_type)

        mid_size = self.width // 2  # both height and width are guaranteed to be equal

        # Generate vertical wall
        self.grid.vert_wall(mid_size, 0, obj_type=wall_type)

        # Generate horizontal walls
        self.grid.horz_wall(0, mid_size, mid_size, obj_type=wall_type)
        if get_param(self.grid_params, CustomMiniGrid.GRID_RIGHT_ROOMS_EVEN):
            right_rooms_offset = 0
        else:
            right_rooms_offset = 1
        self.grid.horz_wall(mid_size, mid_size + right_rooms_offset, mid_size, obj_type=wall_type)

        # Door between upper-left and lower-left rooms
        lv_door, rv_door = (2, mid_size), (mid_size + mid_size // 2, mid_size + right_rooms_offset)
        th_door, bh_door = (mid_size,  mid_size // 2), (mid_size, self.height - 3)

        # Lava locations
        if self.use_lava:
            door_locs = get_param(self.grid_params, CustomMiniGrid.GRID_LAVA_LOCS, CustomMiniGrid.GRID_LAVA_LOCS_DOOR_INTERSECTIONS)
            if door_locs == CustomMiniGrid.GRID_LAVA_LOCS_DOOR_INTERSECTIONS:
                doors = np.array([lv_door, rv_door, th_door, bh_door])
                min_x, min_y = doors.min(axis=0)
                max_x, max_y = doors.max(axis=0)
                for x in [min_x, max_x]:
                    for y in [min_y, max_y]:
                        self.grid.set(x, y, Lava())
            elif door_locs == CustomMiniGrid.GRID_LAVA_LOCS_ALL_CORNERS:
                for corner in [
                    (1, 1), (1, mid_size - 1), (mid_size - 1, mid_size - 1), (mid_size - 1, 1), (mid_size + 1, 1),
                    (self.width - 2, 1), (mid_size + 1, mid_size + right_rooms_offset - 1),
                    (self.width - 2, mid_size + right_rooms_offset - 1), (1, mid_size + 1), (mid_size - 1, mid_size + 1),
                    (1, self.height - 2), (mid_size - 1, self.height - 2), (mid_size + 1, mid_size + 1 + right_rooms_offset),
                    (self.width - 2, mid_size + 1 + right_rooms_offset), (mid_size + 1, self.height - 2),
                    (self.width - 2, self.height - 2)
                ]:
                    self.grid.set(*corner, Lava())
            elif door_locs == CustomMiniGrid.GRID_LAVA_LOCS_SOME_CORNERS:
                for corner in [
                    (1, 1), (mid_size - 1, mid_size - 1), (mid_size + 1, mid_size + right_rooms_offset - 1),
                    (self.width - 2, 1), (mid_size + 1, mid_size + right_rooms_offset + 1), (self.width - 2, self.height - 2),
                    (1, self.height - 2), (mid_size - 1, mid_size + 1)
                ]:
                    self.grid.set(*corner, Lava())
            else:
                raise RuntimeError(f"Error: Unknown door locations '{door_locs}.")

        v_doors, h_doors = [lv_door, rv_door], [th_door, bh_door]

        for door_x, door_y in v_doors:
            self.grid.set(door_x, door_y - 1, Wall())
            self.grid.set(door_x, door_y + 1, Wall())

        for door_x, door_y in h_doors:
            self.grid.set(door_x - 1, door_y, Wall())
            self.grid.set(door_x + 1, door_y, Wall())

        # Agent and object locations
        self._place_agent_and_objects([ChainRoom((0, 0), (self.width, self.height), None, None)], False)

        for door_x, door_y in v_doors:
            for offset in [-1, 0, 1]:
                self.grid.set(door_x, door_y + offset, None)

        for door_x, door_y in h_doors:
            for offset in [-1, 0, 1]:
                self.grid.set(door_x + offset, door_y, None)

    def _gen_nine_rooms(self):
        self.grid = Grid(self.width, self.height)

        grid_size = self.width
        room_size = self.width // 3

        # Walls
        for i in range(0, grid_size, room_size):
            self.grid.vert_wall(i, 0)
            self.grid.horz_wall(0, i)

        # Lava locations
        if self.use_lava:
            self.grid.set(room_size + room_size // 2, room_size // 2, Lava())
            self.grid.set(room_size + room_size // 2, grid_size - room_size // 2 - 1, Lava())
            self.grid.set(room_size // 2, room_size + room_size // 2, Lava())
            self.grid.set(grid_size - room_size // 2 - 1, room_size + room_size // 2, Lava())

        # Agent and objects
        self._place_agent_and_objects([ChainRoom((0, 0), (self.width, self.height), None, None)], False)

        # Add horizontal doors
        self.grid.set(room_size, room_size // 2, None)
        self.grid.set(grid_size - room_size - 1, room_size // 2, None)
        self.grid.set(room_size, grid_size - room_size // 2 - 1, None)
        self.grid.set(grid_size - room_size - 1, grid_size - room_size // 2 - 1, None)

        # Add vertical doors
        self.grid.set(room_size // 2, room_size, None)
        self.grid.set(room_size // 2, grid_size - room_size - 1, None)
        self.grid.set(room_size + room_size // 2, room_size, None)
        self.grid.set(grid_size - room_size // 2 - 1, room_size, None)
        self.grid.set(grid_size - room_size // 2 - 1, grid_size - room_size - 1, None)
