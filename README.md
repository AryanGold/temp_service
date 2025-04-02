# DataAgent

Back-end service for retrieve Options and Models data.

Service main tasks:

- retrieve Quotes raw data from data provider (ThetaData etc).

- normalize and processing data (model evaluation).

- send finish data to fron-end clients via WebSocket.

### System requirements

- Linux OS (tested Debian 12, Ubuntu 25.04)

- Docker (version 28.0.1)

- Python 3 (on host)

### How to develop/test

- 1) Build and run Docker guest console:
```
python3 run.py
```

- 2) Inside guest console run tmux and create two terminals:
```
tmux
```
Press Ctr + B + %

- 3) In first console run service:
```
clear && python3 dataagent/dataagent_service.py
```
For stop service press Ctr + C

- 4) In second console run test script (which connected to service by WebSocket)<br>
Switch to another console by press Ctr + B + o<br>
```
clear && p dataagent/test/test_client.py
```

