from prometheus_client import start_http_server, Summary, Gauge, Counter, Histogram
import random
import time

# Metrics
inference_requests_total = Counter('inference_requests_total', 'Total inference requests')
inference_success_total = Counter('inference_success_total', 'Total successful inferences')
inference_failure_total = Counter('inference_failure_total', 'Total failed inferences')
inference_latency = Summary('inference_latency_seconds', 'Time spent on inference')
inference_queue = Gauge('inference_queue_size', 'Current queue size')
model_accuracy = Gauge('model_accuracy', 'Current model accuracy')
cpu_usage = Gauge('cpu_usage_percent', 'CPU usage percentage')
memory_usage = Gauge('memory_usage_percent', 'Memory usage percentage')
request_size = Histogram('inference_request_size_bytes', 'Size of inference requests')
model_version = Gauge('model_version', 'Version of the deployed model')

# Inference simulator fuction
@inference_latency.time()
def simulate_inference():
    inference_requests_total.inc()
    if random.random() > 0.1:
        inference_success_total.inc()
        print("Request success")
    else:
        inference_failure_total.inc()
        print("Request failed")

    inference_queue.set(random.randint(0, 10))
    model_accuracy.set(random.uniform(0.85, 0.95))
    cpu_usage.set(random.uniform(20, 80))
    memory_usage.set(random.uniform(30, 70))
    request_size.observe(random.randint(100, 2000))
    model_version.set(1.0)

# Only run exporter if main
if __name__ == '__main__':
    start_http_server(8000)
    print("Exporter running on http://localhost:8000/metrics")
    while True:
        simulate_inference()
        time.sleep(1)