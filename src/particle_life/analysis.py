class AnalysisConfig:
    def __init__(self, input_path: str, output_path: str):
        self.input_path = input_path
        self.output_path = output_path


class Analyzer:
    def __init__(self, config: AnalysisConfig):
        self.config = config
