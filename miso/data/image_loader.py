from multiprocessing import Process, cpu_count, Queue
import skimage.io as skio
import numpy as np
from tqdm import tqdm


def produce(producer_queue, data, workers):
    for datum in data:
        producer_queue.put(datum)
    [producer_queue.put(None) for i in range(workers)]


def work(producer_queue, consumer_queue, transform_fn, transform_args):
    while True:
        res = producer_queue.get()
        if res is None:
            consumer_queue.put(None)
            break
        im = skio.imread(res[1])
        if transform_fn is not None:
            if transform_args is not None:
                im = transform_fn(im, *transform_args)
            else:
                im = transform_fn(im)
        consumer_queue.put((res[0], im))


def consume(consumer_queue, array, multiplier, num_workers):
    pbar = tqdm(total=len(array))
    while num_workers > 0:
        res = consumer_queue.get()
        if res is None:
            num_workers -= 1
            continue
        array[res[0]*multiplier:(res[0]+1)*multiplier] = res[1]
        pbar.update(multiplier)
    pbar.close()


def get_array_shape(filenames):
    im = skio.imread(filenames[0])
    return (len(filenames),) + im.shape


class ParallelImageLoader:
    def __init__(self, filenames, array, multiplier=1, transform_fn=None, transform_args=None):
        self.filenames = filenames
        self.array = array
        self.transform_fn = transform_fn
        self.transform_args = transform_args
        self.multiplier = multiplier

        self.producer_queue = Queue()
        self.consumer_queue = Queue(cpu_count() * 4)
        self.workers = None
        self.NUMBER_OF_PROCESSES = cpu_count()

    def load(self):
        print("Starting queue with {} workers".format(self.NUMBER_OF_PROCESSES))
        payload = zip(range(len(self.filenames)), self.filenames)
        self.workers = [
            Process(target=work, args=(self.producer_queue, self.consumer_queue, self.transform_fn, self.transform_args), name='producer {}'.format(i)) for i in
            range(self.NUMBER_OF_PROCESSES)]
        for w in self.workers:
            w.start()
        try:
            produce(self.producer_queue, payload, self.NUMBER_OF_PROCESSES)
            consume(self.consumer_queue, self.array, self.multiplier, self.NUMBER_OF_PROCESSES)
        except KeyboardInterrupt:
            print("Keyboard interrupt...")
        finally:
            for w in self.workers:
                w.terminate()
                w.join()
        self.producer_queue.close()
        self.consumer_queue.close()

