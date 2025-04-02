# Instance of DataProviders for ThetaData 
# thetadata.net

import subprocess
import os
import signal
import atexit
from pathlib import Path
import sys
import requests
import json
import logging
import pandas as pd

sys.path.insert(0, '../..')
import utils
import conf

class ThetaData:
    def __init__(self):
        self.jar_file = "ThetaTerminal.jar"
        self.process = None

        self.REST_URL_1 = "http://127.0.0.1:25510/v2/bulk_snapshot/option/quote?root="
        self.REST_URL_2 = "&exp=0"

        # Load specific DataProvider settings
        dp_file = os.path.dirname(os.path.realpath(__file__)) + '/' + conf.data['data_providers']['theta_data']['conf_file']
        if not Path(dp_file).is_file():
            logging.error(f"[ThetaData] Error: Not found configuration file for DataProvider 'ThetaData': {dp_file}")
            exit(-1)
        self.dp_conf = utils.load_config(dp_file)

        self.java_log_file = os.path.dirname(os.path.realpath(__file__)) + '/java_log.txt'

    def start(self):
        # Start the process in the background.
        if self.process is None:
            with open(self.java_log_file, "w") as log:
                jar_path = os.path.dirname(os.path.realpath(__file__)) + '/' + self.jar_file
                command = ["java", "-jar", jar_path] + [self.dp_conf['main']["login"], self.dp_conf['main']["pass"]]
                self.process = subprocess.Popen(
                    command,
                    stdout=log, # Redirect stdout to log file
                    stderr=log, # Redirect stderr to log file
                    preexec_fn=os.setpgrp  # Create a new process group
                )
                logging.info(f"[ThetaData] Started Java process with PID: {self.process.pid}")
        else:
            logging.warning("[ThetaData] Java process is already running.")

    def stop(self):
        # Stop the Java process.
        if self.process is not None:
            logging.info("[ThetaData] Stopping Java process...")
            os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)  # Kill the Java process group
            self.process.wait()
            logging.info("[ThetaData] Java process stopped.")
            self.process = None
        else:
            logging.warning("[ThetaData] No Java process is running.")

    def is_running(self):
        # Check if the Java process is still running.
        return self.process is not None and self.process.poll() is None

    def get_chain(self, symbol):
        data = self.get_data_rest(symbol)
        if data is None:
            return None

        provider_latency_ms = None

        # Parse tickers
        isParseSuccess = False
        try:
            provider_latency_ms = data['header']['latency_ms']

            all_chain_data = []
            quotes = data['response']
            for quote in quotes:
                contract = quote['contract']
                ticks = quote['ticks'][0]
                option_data = {
                    'ticker': contract['root'],
                    'expiration_dates': utils.date_with_separator(str(contract['expiration'])),
                    'option_types': "p" if (contract['right'] == 'P') else "c",
                    'strikes': contract['strike'],
                    'bids': ticks[3],
                    'asks': ticks[7],
                    'bid_sizes': ticks[1],
                    'ask_sizes': ticks[5],
                    'volume': 0,
                    'symbol': contract['root']
                }
                all_chain_data.append(option_data)

            isParseSuccess = True
        except Exception as e:
            logging.warning("[ThetaData]->get_chain_obj(): Unable to parse tickers:", e)

        if not isParseSuccess:
            return None, None

        df = pd.DataFrame(all_chain_data)

        return df, provider_latency_ms

    def get_data_rest(self, symbol):
        isSuccess = False

        try:
            # http://127.0.0.1:25510/v2/bulk_snapshot/option/quote?root=AAPL&exp=0
            url = self.REST_URL_1 + symbol + self.REST_URL_2
            response = requests.get(url, timeout=5)
            response.raise_for_status()

            if response.status_code != 200:
                return None

            data = response.json()

            isSuccess = True

        except requests.exceptions.Timeout:
            logging.error("[ThetaData]->get_data_rest(): Timeout occurred while connecting to the API.")
        except requests.exceptions.ConnectionError:
            logging.error("[ThetaData]->get_data_rest(): Failed to connect to the server.")
        except requests.exceptions.HTTPError as e:
            logging.error(f"[ThetaData]->get_data_rest(): HTTP error occurred: {e} ({e.response.status_code})")
        except requests.exceptions.RequestException as e:
            logging.error("[ThetaData]->get_data_rest(): General RequestException occurred:", e)
        except Exception as e:
            logging.error("[ThetaData]->get_data_rest(): Unexpected error:", e)

        if not isSuccess:
            return None

        return data

    # https://http-docs.thetadata.us/operations/get-bulk_hist-option-quote.html
    def get_bulk_historical_option(self, symbol, start_date, end_date, expiry, interval):
        THETA_URL = "http://127.0.0.1:25510"
        endpoint = f"{THETA_URL}/v2/bulk_hist/option/quote"
        params = {
            "root": symbol,
            "exp": expiry,
            "start_date": start_date, 
            "end_date": end_date, 
            "ivl": interval  
        }
        
        response = requests.get(endpoint, params=params)
        
        if response.status_code == 200:
            return response.json()  # Return JSON response
        else:
            logging.error(f"[DP_ThetaData] Error {response.status_code}: {response.text}")
            return None

def cleanup():
    if conf.dp:
        conf.dp.stop()

atexit.register(cleanup)  # Register cleanup function to stop DataProvider on exit