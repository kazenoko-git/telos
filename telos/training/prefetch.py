"""Cancellable CUDA batch pipeline with reusable pinned staging buffers."""
import queue
import threading


class CUDAPrefetcher:
    def __init__(self, dataset, cursor, batch_size, seq_len, stride, device, steps):
        import torch
        from .dataloader import get_global_targets_contiguous
        self.queue = queue.Queue(maxsize=1)
        self.stopped = threading.Event()
        self.device = device
        self.stream = torch.cuda.Stream(device=device)

        def put(value):
            while not self.stopped.is_set():
                try:
                    self.queue.put(value, timeout=.1)
                    return True
                except queue.Full:
                    pass
            return False

        def worker():
            try:
                torch.cuda.set_device(device)
                # Two buffers allow CPU filling while the previous transfer runs.
                buffers = [torch.empty((batch_size, seq_len), dtype=torch.long, pin_memory=True) for _ in range(2)]
                events = [None, None]
                position = cursor
                for step in range(steps):
                    if self.stopped.is_set():
                        break
                    slot = step % 2
                    if events[slot] is not None:
                        events[slot].synchronize()  # CPU may now overwrite this staging buffer.
                    batch, _ = get_global_targets_contiguous(dataset, position, batch_size, seq_len)
                    import numpy as np
                    np.copyto(buffers[slot].numpy(), batch, casting='unsafe')
                    position = (position + stride) % len(dataset)
                    with torch.cuda.stream(self.stream):
                        device_batch = buffers[slot].to(device, non_blocking=True)
                        ready = torch.cuda.Event()
                        ready.record(self.stream)
                    events[slot] = ready
                    if not put((device_batch, ready, None)):
                        break
                self.stream.synchronize()  # Keep staging storage alive through the final transfer.
            except Exception as exc:
                put((None, None, exc))

        self.thread = threading.Thread(target=worker, daemon=True)
        self.thread.start()

    def get(self):
        import torch
        batch, ready, error = self.queue.get()
        if error is not None:
            raise error
        consumer = torch.cuda.current_stream(self.device)
        consumer.wait_event(ready)
        batch.record_stream(consumer)
        return batch

    def close(self):
        self.stopped.set()
        self.thread.join()
