
import torch
import torch.nn as nn
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

    model = nn.Linear(1,1,bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    x = torch.tensor([[float(rank + 1)]])
    target = torch.tensor([[0.0]])
    prediction = model(x)
    loss = (prediction - target).pow(2).mean()

    loss.backward()

    print(
        f"Rank {rank}: local gradient = "
        f"{model.weight.grad.item():.1f}"
    )
    dist.all_reduce(
        model.weight.grad,
        op=dist.ReduceOp.SUM,
    )
    model.weight.grad/=world_size

    print(
        f"Rank {rank}: averaged gradient = "
        f"{model.weight.grad.item():.1f}"
    )


    with torch.no_grad():
        model.weight -= 0.1 * model.weight.grad

    print(
        f"Rank {rank}: updated weight = "
        f"{model.weight.item():.1f}"
    )

    dist.destroy_process_group()

def main() ->None:
    if RENDEZVOUS_FILE.exists():
        RENDEZVOUS_FILE.unlink()
    mp.spawn(worker,args = (WORLD_SIZE,),nprocs=WORLD_SIZE,join=True)

if __name__ == "__main__":
    main()