class ReplayBuffer:
    def __init__(self, group_size: int):
        self.group_size = group_size
        self.buffer = []

    def clear(self):
        self.buffer = []

    def push(self, group: list[tuple[str, str, str]]):
        self.buffer.extend(group)

    def replay(self) -> list[tuple[str, str, str]]:
        return self.buffer
