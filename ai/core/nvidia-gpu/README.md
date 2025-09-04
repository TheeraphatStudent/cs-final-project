# Nvidia Gpu

## Nvidia driver container

**Install nvcc**

`Must be direct install nvidia driver first naja!`

- [Full guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)

```bash
sudo dnf config-manager addrepo --from-repofile https://developer.download.nvidia.com/compute/cuda/repos/fedora42/x86_64/cuda-fedora42.repo
sudo dnf clean all
sudo dnf -y install cuda-toolkit-13-0
```

```bash
sudo dnf install nvidia-open --allowerasing
```

```bash
sudo dnf -y install cuda-drivers --allowerasing
```

```bash
sudo dnf install kmod-nvidia-open-dkms --allowerasing
```

**Install nvidia container toolkit**
```bash
sudo dnf copr enable @ai-ml/nvidia-container-toolkit
```

**Install nvidia driver**
```bash
sudo dnf install xorg-x11-drv-nvidia-cuda
sudo dnf install nvidia-container-toolkit nvidia-container-toolkit-selinux
```

**Generate the Container Device Interface (CDI) configuration file**
```bash
sudo nvidia-ctk cdi generate -output /etc/cdi/nvidia.yaml
```

**Testing**
```bash
podman run --device nvidia.com/gpu=all --rm fedora:latest nvidia-smi
```

Output
```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.76.05              Driver Version: 580.76.05      CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 3050 ...    Off |   00000000:01:00.0 Off |                  N/A |
| N/A   55C    P0             15W /   75W |      15MiB /   4096MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
```

----

Test with compose


Container
```yaml
services:
  test:
    image: nvidia/cuda:12.9.0-base-ubuntu22.04
    command: nvidia-smi
    devices:
      - nvidia.com/gpu=all
```

Running command
```bash
podman compose -f ./cuda.docker-compose.yaml up -d
```

**Additional Setup**

1. Find cuda core install location

```
/usr/local/cuda
```

2. export cuda path

```bash
nano ~/.bashrc
```

```bash
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

```bash
source ~/.bashrc
```


## Nvidia cuda core checker

After working at `Additional Setup`

1. Complice an `check_cuda.cu` with `nvcc`

```bash
nvcc check_cuda.cu -o check_cuda
```

2. Execute check_cuda

```bash
./check_cuda
```

```bash
Found 1 CUDA-enabled device(s).

--- Device 0: NVIDIA GeForce RTX 3050 Laptop GPU ---
  Compute capability:          8.6
  Total global memory:         3768 MB
  Number of multiprocessors:   16
  CUDA Core count:             3072 (for reference, may vary)
```