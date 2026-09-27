
import torch
from pathlib import Path 

import torch.distributed as dist 

import torch.multiprocessing as mp 

WORLD_SIZE = 2 
RENDEZVOUS_FILE = Path(__file__).with_name("gloo_rendezvous").resolve()

def worker(rank:int,world_size:int)->None:
    dist.init_process_group(
        backend="gloo",
        rank=rank,
        world_size=world_size,
        init_method=RENDEZVOUS_FILE.as_uri(),
    )

    print(f"Rank {rank}: process group initialized")

    tensor = torch.tensor([float(rank+1)])
    print(f"Rank {rank}: before all_reduce = {tensor.item()}")

    dist.all_reduce(tensor,op = dist.ReduceOp.SUM)
    print(f"Rank {rank}: after all_reduce = {tensor.item()}")

    dist.destroy_process_group()

def main() ->None:
    if RENDEZVOUS_FILE.exists():
        RENDEZVOUS_FILE.unlink()
    mp.spawn(worker,args = (WORLD_SIZE,),nprocs=WORLD_SIZE,join=True)

if __name__ == "__main__":
    main()