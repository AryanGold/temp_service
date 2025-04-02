# Container for Client data, sumbos list, models list etc
# Concurent safe.
class ClientDataManager:
    def __init__(self, manager):
        self.manager = manager
        self.data = self.manager.dict()
        self.locks = self.manager.dict()

    def init(self, key, value):
        if key not in self.data:
            self.data[key] = value
        if key not in self.locks:
            self.locks[key] = self.manager.Lock()

    def get(self, key):
        return self.data.get(key, None)

    def get_keys(self):
        return list(self.data.keys())

    def update(self, key, new_value):
        self.data[key] = new_value

    def acquire_lock(self, key):
        return self.locks[key]

    ###
    # Functional for allow iterate over data
    def __iter__(self):
        # Return an iterator over a snapshot of the current keys
        return iter(list(self.data.keys()))

    def items(self):
        # Safely return a copy of key-value pairs
        return list(self.data.items())

    def values(self):
        return list(self.data.values())

    def keys(self):
        return list(self.data.keys())
    ###