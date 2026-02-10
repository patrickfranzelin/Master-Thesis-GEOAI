class Pipeline:
    name: str

    def execute(self, ctx):
        raise NotImplementedError
