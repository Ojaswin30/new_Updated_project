import argparse
import json
import time
import psutil
import os

from ml.src.pipeline.late_fusion import LateFusionPipeline, PipelineConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--text", required=True)
    args = parser.parse_args()

    process = psutil.Process(os.getpid())

    start_time = time.perf_counter()
    cpu_start = psutil.cpu_percent(interval=None)
    mem_start = process.memory_info().rss

    pipe = LateFusionPipeline(PipelineConfig())
    out = pipe.run(args.image, args.text)

    end_time = time.perf_counter()
    cpu_end = psutil.cpu_percent(interval=None)
    mem_end = process.memory_info().rss

    out["performance"] = {
        "execution_time_seconds": round(end_time - start_time, 4),
        "cpu_percent_before": cpu_start,
        "cpu_percent_after": cpu_end,
        "memory_usage_mb_before": round(mem_start / (1024 * 1024), 2),
        "memory_usage_mb_after": round(mem_end / (1024 * 1024), 2),
    }

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
