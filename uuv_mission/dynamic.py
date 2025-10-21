from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from .terrain import generate_reference_and_limits

class Submarine:
    def __init__(self):

        self.mass = 1
        self.drag = 0.1
        self.actuator_gain = 1

        self.dt = 1 # Time step for discrete time simulation

        self.pos_x = 0
        self.pos_y = 0
        self.vel_x = 1 # Constant velocity in x direction
        self.vel_y = 0


    def transition(self, action: float, disturbance: float):
        self.pos_x += self.vel_x * self.dt
        self.pos_y += self.vel_y * self.dt

        force_y = -self.drag * self.vel_y + self.actuator_gain * (action + disturbance)
        acc_y = force_y / self.mass
        self.vel_y += acc_y * self.dt

    def get_depth(self) -> float:
        return self.pos_y
    
    def get_position(self) -> tuple:
        return self.pos_x, self.pos_y
    
    def reset_state(self):
        self.pos_x = 0
        self.pos_y = 0
        self.vel_x = 1
        self.vel_y = 0
    
class Trajectory:
    def __init__(self, position: np.ndarray):
        self.position = position  
        
    def plot(self):
        plt.plot(self.position[:, 0], self.position[:, 1])
        plt.show()

    def plot_completed_mission(self, mission: Mission):
        x_values = np.arange(len(mission.reference))
        min_depth = np.min(mission.cave_depth)
        max_height = np.max(mission.cave_height)

        plt.fill_between(x_values, mission.cave_height, mission.cave_depth, color='blue', alpha=0.3)
        plt.fill_between(x_values, mission.cave_depth, min_depth*np.ones(len(x_values)), 
                         color='saddlebrown', alpha=0.3)
        plt.fill_between(x_values, max_height*np.ones(len(x_values)), mission.cave_height, 
                         color='saddlebrown', alpha=0.3)
        plt.plot(self.position[:, 0], self.position[:, 1], label='Trajectory')
        plt.plot(mission.reference, 'r', linestyle='--', label='Reference')
        plt.legend(loc='upper right')
        plt.show()

@dataclass
class Mission:
    reference: np.ndarray
    cave_height: np.ndarray
    cave_depth: np.ndarray

    @classmethod
    def random_mission(cls, duration: int, scale: float):
        (reference, cave_height, cave_depth) = generate_reference_and_limits(duration, scale)
        return cls(reference, cave_height, cave_depth)

    @classmethod
    def from_csv(cls, file_name: str):
        """Load mission data from a CSV file and return a Mission instance.

        The CSV is expected to have three columns with headers:
        `reference`, `cave_height`, `cave_depth`.

        Args:
            file_name: Path to the CSV file.

        Returns:
            Mission: an instance populated with numpy arrays for
            reference, cave_height and cave_depth.

        Raises:
            ValueError: if the required columns are missing or lengths mismatch.
        """
        try:
            import pandas as pd
        except Exception:
            pd = None

        if pd is not None:
            df = pd.read_csv(file_name)
            required = ["reference", "cave_height", "cave_depth"]
            if not all(col in df.columns for col in required):
                raise ValueError(f"CSV must contain columns: {required}")

            reference = df["reference"].to_numpy(dtype=float)
            cave_height = df["cave_height"].to_numpy(dtype=float)
            cave_depth = df["cave_depth"].to_numpy(dtype=float)
        else:
            # Fallback to builtin csv reader
            import csv
            reference_list = []
            cave_height_list = []
            cave_depth_list = []
            with open(file_name, newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                required = ["reference", "cave_height", "cave_depth"]
                if not all(col in reader.fieldnames for col in required):
                    raise ValueError(f"CSV must contain columns: {required}")
                for row in reader:
                    reference_list.append(float(row["reference"]))
                    cave_height_list.append(float(row["cave_height"]))
                    cave_depth_list.append(float(row["cave_depth"]))

            import numpy as _np
            reference = _np.array(reference_list, dtype=float)
            cave_height = _np.array(cave_height_list, dtype=float)
            cave_depth = _np.array(cave_depth_list, dtype=float)

        # Validate lengths
        if not (len(reference) == len(cave_height) == len(cave_depth)):
            raise ValueError("Columns must have the same length")

        return cls(reference, cave_height, cave_depth)


class ClosedLoop:
    def __init__(self, plant: Submarine, controller):
        self.plant = plant
        self.controller = controller

    def simulate(self,  mission: Mission, disturbances: np.ndarray) -> Trajectory:

        T = len(mission.reference)
        if len(disturbances) < T:
            raise ValueError("Disturbances must be at least as long as mission duration")
        
        positions = np.zeros((T, 2))
        actions = np.zeros(T)
        self.plant.reset_state()

        # If the controller has a reset method, call it to clear internal state
        if hasattr(self.controller, "reset"):
            try:
                self.controller.reset()
            except Exception:
                pass

        for t in range(T):
            positions[t] = self.plant.get_position()
            observation_t = self.plant.get_depth()

            # Compute control action using the provided controller
            # Controller is expected to be callable: u = controller(reference, observation)
            ref_t = float(mission.reference[t])
            try:
                actions[t] = float(self.controller(ref_t, observation_t))
            except Exception:
                # If controller can't be called directly, assume it's a function taking (ref, obs, t)
                actions[t] = float(self.controller(ref_t, observation_t, t))

            # Apply plant transition with computed action and disturbance
            self.plant.transition(actions[t], disturbances[t])

        return Trajectory(positions)
        
    def simulate_with_random_disturbances(self, mission: Mission, variance: float = 0.5) -> Trajectory:
        disturbances = np.random.normal(0, variance, len(mission.reference))
        return self.simulate(mission, disturbances)
