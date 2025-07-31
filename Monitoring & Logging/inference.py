import time
from prometheus_exporter import simulate_inference

if __name__ == "__main__":
    print("Running inference simulation loop...")
    while True:
        simulate_inference()
        time.sleep(1)