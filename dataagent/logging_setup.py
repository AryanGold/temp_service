from datetime import datetime, timezone
import logging
import os

utc_now = datetime.now(timezone.utc)
formatted_utc = utc_now.strftime("%Y-%m-%d_%H-%M-%S")

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"{formatted_utc}.log")

# Configure logging globally
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    filename=log_file,
    filemode="a"
)

# Console logging settings
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
console.setFormatter(formatter)
logging.getLogger().addHandler(console)
