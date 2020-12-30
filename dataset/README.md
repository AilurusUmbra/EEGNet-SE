# Cloud utility dataset

* This dataset is sampled from Prometheus on Prometheus server of [MPC lab](https://mpc.cs.nctu.edu.tw/), since we don't have the permission to publish the real data of TWCC.

* The data is of the size (N, T, C)
    * N: The number of data instances.
    * T: The number of timestamps.
    * C: The number of channels.

* Selected channels:
    * container_accelerator_duty_cycle
    * container_accelerator_memory_used_bytes
    * container_cpu_usage_seconds_total
    * container_memory_max_usage_bytes
    * container_memory_rss
    * container_memory_swap
    