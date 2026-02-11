class SimulationConfig:
    def __init__(self, steps: int, n_particles: int):
        self.steps = steps
        self.n_particles = n_particles


class Simulator:
    def __init__(self, config: SimulationConfig):
        self.config = config
