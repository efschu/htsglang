## Run benchmark

### Benchmark flliper

Run Llama-8b

```bash
python3 -m flliper.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct --port 30000
```

Benchmark

```bash
python3 bench_flliper.py
```
