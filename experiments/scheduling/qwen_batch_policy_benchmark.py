import statistics
from experiments.scheduling.qwen_batch_process_demo import run_qwen_batch_policy


ROUNDS =3

POLICIES = [
    ("batch_size=2, wait=50ms", 2, 50),
    ("batch_size=4, wait=50ms", 4, 50),
]

def main() ->None:

    for policy_name,max_batch_size,max_batch_wait_ms in POLICIES:
        all_latencies_ms = []
        round_tail_latencies_ms = []
        print(f"\n===== {policy_name} =====")
        for round_number in range(1,ROUNDS+1):
            print(f"\n----- Round {round_number} -----")
            lms = run_qwen_batch_policy(max_batch_size=max_batch_size,max_batch_wait_ms=max_batch_wait_ms)

            all_latencies_ms.extend(lms)
            round_tail_latencies_ms.append(max(lms))

        median_task_latency_ms = statistics.median(all_latencies_ms)
        median_tail_latency_ms = statistics.median(round_tail_latencies_ms)
        estimated_throughput = (3 * 1000 / median_tail_latency_ms)
        
        print(f"\n{policy_name} summary")
        print(
            f"Median task latency: "
            f"{median_task_latency_ms:.1f} ms"
        )
        print(
            f"Median tail latency: "
            f"{median_tail_latency_ms:.1f} ms"
        )
        print(
            f"Estimated request throughput: "
            f"{estimated_throughput:.2f} requests/s"
        )


if __name__ == "__main__":
    main()
