from multiprocessing import Process, cpu_count, JoinableQueue
import skimage.io as skio
import numpy as np
from tqdm import tqdm


def produce(producer_queue, data, workers):
    for datum in data:
        producer_queue.put(datum)
    [producer_queue.put(None) for i in range(workers)]


def work(producer_queue, consumer_queue, transform_fn, transform_args):
    while True:
        try:
            res = producer_queue.get()
            if res is None:
                producer_queue.task_done()
                consumer_queue.put(None)
                break
            im = skio.imread(res[1])
            if transform_fn is not None:
                if transform_args is not None:
                    im = transform_fn(im, *transform_args)
                else:
                    im = transform_fn(im)
            consumer_queue.put((res[0], im))
            producer_queue.task_done()
        except (KeyboardInterrupt, SystemExit):
            print("- exiting worker")
            break


def consume(consumer_queue, array, multiplier, num_workers):
    pbar = tqdm(total=len(array))
    while num_workers > 0:
        try:
            res = consumer_queue.get()
            if res is None:
                consumer_queue.task_done()
                num_workers -= 1
                continue
            array[res[0]*multiplier:(res[0]+1)*multiplier] = res[1]
            consumer_queue.task_done()
            pbar.update(multiplier)
        except (KeyboardInterrupt, SystemExit):
            print("- exiting consumer")
            break
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

        self.producer_queue = JoinableQueue()
        self.consumer_queue = JoinableQueue(cpu_count() * 4)
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
        produce(self.producer_queue, payload, self.NUMBER_OF_PROCESSES)
        consume(self.consumer_queue, self.array, self.multiplier, self.NUMBER_OF_PROCESSES)
        # self.producer_queue.join()
        # self.consumer_queue.join()
        self.producer_queue.close()
        self.consumer_queue.close()


if __name__ == "__main__":
    from mml.data.filenames_dataset import FilenamesDataset

    fs = FilenamesDataset(r"C:\data\SeagrassFrames")
    fs.split(0.05)

    shape = get_array_shape(fs.test_filenames)
    print("Array shape is {}".format(shape))

    from numpy.lib.format import open_memmap

    mmap = open_memmap(r"D:\temp\test.npy", mode='w+', dtype=np.int8, shape=shape)
    loader = ParallelImageLoader(fs.test_filenames, mmap)
    loader.load()
