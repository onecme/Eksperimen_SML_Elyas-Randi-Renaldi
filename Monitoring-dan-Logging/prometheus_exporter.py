from prometheus_client import start_http_server, Counter, Gauge, Histogram
from time import time, sleep
import psutil
import random

# Metrik dasar
total_requests = Counter("requests_total", "Total inference requests")
success_requests = Counter("requests_success", "Successful inference requests")
failed_requests = Counter("requests_failed", "Failed inference requests")

# Metrik waktu inferensi
inference_time = Histogram("inference_time_seconds", "Time spent for inference")

# Metrik monitoring sistem
cpu_usage = Gauge("cpu_usage_percent", "CPU usage percentage")
memory_usage = Gauge("memory_usage_percent", "Memory usage percentage")

# Metrik input
input_mean = Gauge("input_mean", "Mean value of input features")
input_std = Gauge("input_std", "Std dev of input features")

# Metrik output
prediction_output = Gauge("prediction_output", "Last prediction output")

# Metrik manual — akurasi model dari training
model_accuracy = Gauge("model_accuracy", "Model accuracy score")
model_accuracy.set(0.83)  # isi sesuai hasil modelling.py Anda


def simulate_prediction():
    total_requests.inc()

    # simulasi input
    x = [random.random() for _ in range(5)]
    input_mean.set(sum(x)/len(x))
    input_std.set((sum((i - input_mean._value.get())**2 for i in x)/len(x))**0.5)

    start = time()
    sleep(0.2)  # simulasi inferensi
    pred = random.choice([0, 1])
    prediction_output.set(pred)

    # success
    success_requests.inc()

    # waktu inferensi
    inference_time.observe(time() - start)

    # Update sistem
    cpu_usage.set(psutil.cpu_percent())
    memory_usage.set(psutil.virtual_memory().percent)


if __name__ == "__main__":
    print("Prometheus Exporter running at :8000")
    start_http_server(8000)

    while True:
        simulate_prediction()
        sleep(2)
