# Container global and configiration data

from multiprocessing import Manager, Process, Value, Lock

# Data loaded from service_conf.yaml
data = {}

# Global multiprocessing.Manager for share data between processes.
# Should be initialize in main()
glob_manager = None

# List of options for monitoring.
options = set()

# Dict of connected clients and their parameters.
clients = {}

# Default model, uses if user not chosed specific models for evaluation
default_model = "ssvi"

# Data Provider instance, see "dataagent/DataProviders"
dp = None

# Prcoesses container
processes = {}

# Concurent Safe variable for store client_id incrementor
class ClientId_Manager:
    def __init__(self):
        self.lock = Lock()
        self.client_id = Value('i', 0)

    # Quick getter
    def __call__(self):
        with self.lock:
            return self.client_id.value

    def inc(self):
        with self.lock:
            self.client_id.value += 1
            return self.client_id.value

client_id = ClientId_Manager()

