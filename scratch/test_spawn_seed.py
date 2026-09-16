import torch
import torch.multiprocessing as mp

def worker(rank, queue):
    torch.manual_seed(42)
    m = torch.nn.Linear(10, 10, bias=False)
    torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
    queue.put((rank, m.weight[0, 0].item()))

if __name__ == "__main__":
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = []
    for r in range(4):
        p = ctx.Process(target=worker, args=(r, q))
        p.start()
        procs.append(p)
    for p in procs: p.join()
    results = [q.get() for _ in range(4)]
    print("Spawned worker weights with manual_seed(42):", results)
