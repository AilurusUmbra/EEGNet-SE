# Cloud Utility Dataset

* This dataset is sampled from Prometheus on Prometheus server of [MPC lab](https://mpc.cs.nctu.edu.tw/), since we don't have the permission to publish the real data of TWCC.

* The data is of the size (N, T, C)
  * N: The number of data instances
  * T: The number of timestamps
  * C: The number of channels


* Selected channels:
    * `container_accelerator_duty_cycle`
    * `container_accelerator_memory_used_bytes`
    * `container_cpu_usage_seconds_total`
    * `container_memory_max_usage_bytes`
    * `container_memory_rss`
    * `container_memory_swap`
    

* `./raw/` include the segmented data from Prometheus server.

* `./aug/` only include the augmented data from `./raw/`
    * Data augmentation methods: DGW-sD from "Time Series Data Augmentation for Neu-ral Networks by Time Warping with a Discriminative Teacher". (ICPR 2021)
