import numpy as np
import copy

from enum import Enum


class Direction(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

    
class PositioningError(Exception):
    pass


class TargetUnreachableError(Exception):
    pass


class Point(object):
    """2D Point"""
    def __init__(self, x, y):
        self.X = x
        self.Y = y

    @staticmethod
    def fromtuple(position_tuple):
        return Point(position_tuple[0], position_tuple[1])
        
    def __str__(self):
        return "[X=%s, Y=%s]" % (self.X, self.Y)
    
    def __repr__(self):
        return "<X=%s, Y=%s>" % (self.X, self.Y)
    
    def __eq__(self, other):
        if isinstance(other, Point):
            return self.X == other.X and self.Y == other.Y
        else:
            return NotImplemented
    
    def __add__(self, other):
        """overload + operator"""
        if other is None:
            return self
        elif isinstance(other, Direction):
            if other == Direction.UP:
                return Point(self.X, self.Y - 1)
            elif other == Direction.RIGHT:
                return Point(self.X + 1, self.Y)
            elif other == Direction.DOWN:
                return Point(self.X, self.Y + 1)
            elif other == Direction.LEFT:
                return Point(self.X - 1, self.Y)
            else:
                raise NotImplementedError("Directions other than up, right, down and left are not implemented.")
        else:
            raise TypeError("Expecting Direction, got %s" % type(other))
    
    def __hash__(self):
        return hash((self.X, self.Y))
    
    def get_x(self):
        return self.X
    
    def get_y(self):
        return self.Y

    def get_flat(self, gridsize):
        return self.Y * gridsize[1] + self.X

    def assert_inside_bounds(self, x_upper, y_upper, x_low=0, y_low=0):
        return self.X >= x_low and self.X < x_upper and self.Y >= y_low and self.Y <y_upper


class Drone(object):
    
    def __init__(self, grid, position, id):
        assert isinstance(id, int), "Expecting int value as ID"
        self.position = position
        self.grid = grid
        self.id = id
        self.trace = []
        self.grid.position_drone(self)

    def __str__(self):
        return "Drone %s at position %s" % (self.id, self.position)

    def __hash__(self):
        return hash(self.id)
    
    def move(self, direction):
        self.grid.move_drone(self, direction)   # possibly throws exception
        self.trace.append(self.position)
        self.position = self.position + direction
        self.grid._update_obstacles_discovered_map(self)
        self.grid._update_locations_seen_map(self)
            
    def observe_surrounding(self):
        # Order: Top, right, down, left
        surrounding = {}
        for d in Direction:
            p = self.position + d
            surrounding.update({p: self.grid.get_value(p)})
        return surrounding

    def observe_obstacles(self):
        return (np.array(self.observe_surrounding()) == "O").astype(np.int8)
    
    def get_position(self):
        return self.position

    def get_position_flat(self):
        return self.position.get_flat(self.grid.size)
    
    def get_position_one_hot(self):
        pos_one_hot = np.zeros(self.grid.size, dtype=np.int8).ravel()
        pos_one_hot[self.get_position_flat()] = 1
        return pos_one_hot
        
    def get_id(self):
        return self.id
        
    def get_trace(self):
        return self.trace

        
class Grid(object):

    def __init__(self, size_y, size_x, obstacles_seed):
        self._grid = np.full([size_y, size_x], None)
        self.discovery_map = np.zeros(size_y * size_x, dtype=np.int64)
        self.obstacles_discovered_map = np.zeros(size_y * size_x, dtype=np.int64)
        self.locations_seen_map = np.zeros(size_y * size_x, dtype=np.int64)
        self.size = (size_y, size_x)
        self.rs_obstacles = np.random.RandomState(obstacles_seed)
        self.drones = []

    def __str__(self):
        return "Simulator grid with size %s:\n%s" % (self.size, self._grid)
        
    def set_obstacles(self, num_obstacles):
        """Obstacles appear in grid as "O"."""
        i = 0
        points = list(self.asdict().keys())
        while i < num_obstacles:
            if len(points) == 0:
                raise ValueError("More obstacles set as there is space. Left side of the has to be clear of obstacles")
            p = self.rs_obstacles.choice(points)
            # left most column is free of obstacles
            if p.assert_inside_bounds(x_upper=self.size[1], y_upper=self.size[0], x_low=1, y_low=0):
                try:
                    self.set_value(p, "O")
                    i += 1
                except PositioningError:
                    points.remove(p)
            else:
                points.remove(p)

    def asdict(self):
        """Returns a dict with cells as keys and content as value"""
        result = {}
        for y in range(self.size[0]):
            for x in range(self.size[1]):
                p = Point(x, y)
                result[p] = self.get_value(p)
        return result
        
    def position_drone(self, drone):
        self.drones.append(drone)
        p = drone.get_position()
        self.set_value(p, drone.get_id())
        self.discovery_map += drone.get_position_one_hot()
        self._update_obstacles_discovered_map(drone)
        self._update_locations_seen_map(drone)
        self.are_drones_set = True
        
    def move_drone(self, drone, direction):
        p = drone.get_position()
        p_new = p + direction
        try:
            self.set_value(p_new, drone.get_id())   # possibly throws exception
            self.reset_value(p)
            p_new_flat = p_new.get_y() * self.size[1] + p_new.get_x()
            self.discovery_map[p_new_flat] += 1  # drone.get_position_one_hot() not possible: position not yet updated
        except IndexError as e:
            p_flat = p.get_y() * self.size[1] + p.get_x()
            self.discovery_map[p_flat] += 1  # drone.get_position_one_hot() not possible: position not yet updated
            raise e
        except PositioningError as e:
            p_flat = p.get_y() * self.size[1] + p.get_x()
            self.discovery_map[p_flat] += 1  # drone.get_position_one_hot() not possible: position not yet updated
            raise e
        
    def get_value(self, point):
        x = point.get_x()
        y = point.get_y()
        if x < 0 or y < 0 or x >= self.size[1] or y >= self.size[0]:
            return "O"   # Treat cells outside of grid borders as obstacles
        else:
            return self._grid[y, x] 
    
    def set_value(self, point, value):
        x = point.get_x()
        y = point.get_y()
        if not point.assert_inside_bounds(x_upper=self.size[1], y_upper=self.size[0]):
            raise IndexError("Index out of bound. [x=%s, y=%s" % (x, y))
        elif self._grid[y, x] is None:
            self._grid[y, x] = value
        else:
            raise PositioningError("Location %s is already used! Content: %s" % (point, self.get_value(point)))
        
    def reset_value(self, point):
        self._grid[point.get_y(), point.get_x()] = None
        
    def get_obstacles_flat(self):
        return (self._grid == "O").astype(int).ravel()

    def get_other_drones_map_flat(self, id):
        result = np.zeros(self.size).ravel()
        for drone in self.drones:
            result[drone.get_position_flat()] = 1
        return result

    def get_discovery_map_flat(self):
        return self.discovery_map.copy().ravel()

    def get_obstacles_discovered_map_flat(self):
        return self.obstacles_discovered_map.copy()

    def get_locations_seen_map_flat(self):
        return self.locations_seen_map.copy()

    def _update_obstacles_discovered_map(self, drone):
        surroundings = drone.observe_surrounding()
        for point, value in surroundings.items():
            if value == "O" and point.assert_inside_bounds(x_upper=self.size[1], y_upper=self.size[0]):
                p = point.get_flat(self.size)
                self.obstacles_discovered_map[p] = 1

    def _update_locations_seen_map(self, drone):
        surroundings = drone.observe_surrounding()
        for point in surroundings.keys():
            if point.assert_inside_bounds(x_upper=self.size[1], y_upper=self.size[0]):
                p = point.get_flat(self.size)
                self.locations_seen_map[p] = 1

    def get_rgb_array(self):
        """
        returns 3d array of rgb values:
           obstacles: black
           empty space: white
           drones: red-->yellow (each drone has a unique color)
        Areas not yet discovered are grey-ish.
        """

        obstacles_grid = self.get_obstacles_flat()
        seen_map = self.get_locations_seen_map_flat()
        
        for i in range(len(obstacles_grid)):
            if obstacles_grid[i] == 0 and seen_map[i] == 1:
                obstacles_grid[i] = 255
                
            #seen obstacle (black)
            elif obstacles_grid[i] == 1 and seen_map[i] == 1:
                obstacles_grid[i]=0
                
                #unseen obstacle (dark grey)
            elif obstacles_grid[i]==1 and seen_map[i] == 0:
                obstacles_grid[i] = 150
            
            #unseen empty space (light grey)
            else:
                obstacles_grid[i] = 200

        # Create channels
        red = obstacles_grid.copy()
        green = obstacles_grid.copy()
        blue = obstacles_grid.copy()

        # first drone is red and subsequent drones are closer to yellow
        red_intensity = 255
        green_intensity = 1
        intensity = int(255 / len(self.drones))

        for drone in self.drones:
            drone_loc = np.argmax(drone.get_position_one_hot())

            red[drone_loc] = red_intensity
            green[drone_loc] = (green_intensity + (99 - drone.get_id()) * intensity)
            blue[drone_loc] = 1

        scale_factor = 50

        red = red.reshape(self.size)
        red = np.kron(red, np.ones((scale_factor, scale_factor)))
        green = green.reshape(self.size)
        green = np.kron(green, np.ones((scale_factor, scale_factor)))
        blue = blue.reshape(self.size)
        blue = np.kron(blue, np.ones((scale_factor, scale_factor)))

        output = np.array((red, green, blue))
        return output


class DroneEnvironment(object):

    MOVEMENT_PENALTY = -0.1
    OBSTACLE_HIT_PENALTY = -1
    NEW_CELL_DISCOVERED_REWARD = 1

    def __init__(self, num_drones, grid_size, num_obstacles, included_states, obstacles_seed, create_gifs=False):
        self.num_drones = num_drones
        self.grid_size = grid_size
        self.num_obstacles = num_obstacles
        self.included_states = included_states
        self.drone_id = 99
        self.obstacles_seed = obstacles_seed
        self.initial_position = Point(0, 0)
        self.create_gifs = create_gifs

        self.step_num = 0
        self.grid = None
        self.drones = {}

        self.reset(num_obstacles, obstacles_seed)

    def __str__(self):
        return str(self.grid)
        
    def reset(self, num_obstacles=None, obstacles_seed=None):
        if num_obstacles is not None: self.num_obstacles = num_obstacles
        if obstacles_seed is not None: self.obstacles_seed = obstacles_seed

        if self.create_gifs: self.gif_frames = []

        self.grid = Grid(self.grid_size[0], self.grid_size[1], self.obstacles_seed)
        self.grid.set_obstacles(self.num_obstacles)
        positions = self.find_neighbor_positions(self.initial_position)
        for i in range(self.num_drones):
            id = self.drone_id - i
            self.drones.update({id: Drone(self.grid, position=positions[i], id=id)})
        
        self.step_num = 0

    def copy(self):
        return copy.deepcopy(self)

    def write_gif(self, file="Test.gif"):
        import array2gif
        array2gif.write_gif(self.gif_frames, file, fps=4)

    def find_neighbor_positions(self, initial_p):
        """
        Function to place all drones next to the initial starting position on the same border

        :param initial_p:
        :return: An array of positions for all drones
        """
        border_positions = []
        if initial_p.X > 0 and initial_p.X < self.grid_size[1]:
            for x in range(self.grid_size[1]):
                border_positions.append(Point(x, initial_p.Y))
        else:
            for y in range(self.grid_size[0]):
                border_positions.append(Point(initial_p.X, y))

        # Sort by distance from initial position
        positions = sorted(border_positions, key=lambda p: np.abs(p.X - initial_p.X) + np.abs(p.Y - initial_p.Y))
        return positions[:self.num_drones]

    def get_state(self, id):
        self.allowed_states = {"drone_location": self.drones[id].get_position_one_hot(),
                               "location_seen_map": self.grid.get_locations_seen_map_flat(),
                               "obstacles_discovered_map": self.grid.get_obstacles_discovered_map_flat(),
                               "other_drones_map": self.grid.get_other_drones_map_flat(id)}

        flat_states = np.concatenate([self.allowed_states[i] for i in self.included_states])
        return flat_states.reshape(len(self.included_states), self.grid_size[0], self.grid_size[1])

    def get_states(self):
        states = {}
        for id in self.drones.keys():
            states[id] = self.get_state(id)
        return states

    def step(self, actions):
        assert isinstance(actions, dict), "got %s with value %s. Expecting dict." % (type(actions), actions)

        # Collect all states before movement
        states = self.get_states()

        rewards = {}
        dones = {}

        # Initializes gif
        if self.create_gifs and self.step_num==0:
            self.gif_frames.append(self.grid.get_rgb_array())

        for id in self.drones.keys():
            old_seen_counts = sum(self.grid.get_locations_seen_map_flat())
            done = False
            reward = self.MOVEMENT_PENALTY
            try:
                self.drones[id].move(Direction(actions[id]))
                new_seen_counts = sum(self.grid.get_locations_seen_map_flat())

                reward += self.NEW_CELL_DISCOVERED_REWARD * (new_seen_counts - old_seen_counts)

                if new_seen_counts == self.grid.size[0]*self.grid.size[1]:
                    # All cells discovered
                    done = True
            except (PositioningError, IndexError):
                reward += self.OBSTACLE_HIT_PENALTY

            rewards[id] = reward
            dones[id] = done

        next_states = self.get_states()

        if self.create_gifs:
            self.gif_frames.append(self.grid.get_rgb_array())

        self.step_num += 1

        return states, rewards, dones, next_states
