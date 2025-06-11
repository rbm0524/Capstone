import torch
print(f"PyTorch Version: {torch.__version__}")

if torch.cuda.is_available():
        print(f"CUDA available: True")
        print(f"CUDA version PyTorch built with: {torch.version.cuda}")
        try:
            print(f"NCCL version: {torch.cuda.nccl.version()}")
        except AttributeError:
            print("torch.cuda.nccl.version() not available in this PyTorch version. Check NCCL installation.")
        else:
            print(f"CUDA available: False")

                                                # 분산 환경 초기화 후 백엔드 확인
                                                # import torch.distributed as dist
                                                # if dist.is_initialized():
                                                #     print(f"Distributed backend: {dist.get_backend()}")

