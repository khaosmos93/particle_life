class SimulationConfig:
    def __init__(self, steps: int, n_particles: int, n_types: int):
        self.steps = steps
        self.n_particles = n_particles
        self.n_types = n_types


class Simulator:
    def __init__(self, config: SimulationConfig):
        self.config = config
