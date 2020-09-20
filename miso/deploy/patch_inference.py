from miso.deploy.inference import load_from_xml
import numpy as np
import skimage.io as skio
import skimage.transform as skt


def tiles_from_image(im, tile_size, tile_step=None):
    if tile_step is None:
        tile_step = tile_size
    max_h = im.shape[0] - tile_size[0]
    max_w = im.shape[1] - tile_size[1]
    steps_h = np.arange(0, max_h, tile_step[0])
    steps_w = np.arange(0, max_w, tile_step[0])
    num_h = len(steps_h)
    num_w = len(steps_w)
    if np.ndim(im) == 2:
        array = np.zeros((num_h, num_w, *tile_size), dtype=im.dtype)
    else:
        array = np.zeros((num_h, num_w, *tile_size, im.shape[2]), dtype=im.dtype)
    for yi, y in enumerate(steps_h):
        for xi, x in enumerate(steps_w):
            array[yi, xi, ...] = im[y:y+tile_size[0], x:x+tile_size[1], ...]
    return array


class PatchClassifier(object):
    def __init__(self, xml_file):
        self.session, self.input, self.output, self.patch_size, self.cls_labels = load_from_xml(xml_file)
        print(self.patch_size)
        self.num_patches = None

    def classify(self, im):
        patches = tiles_from_image(im, self.patch_size[:2])
        self.num_patches = patches.shape[0:2]
        flat_patches = patches.reshape((patches.shape[0] * patches.shape[1], *self.patch_size))
        result = self.session.run(self.output, feed_dict={self.input: flat_patches})
        pred = np.argmax(result, axis=-1).reshape(patches.shape[0:2])
        score = result.reshape(patches.shape[0:2] + (result.shape[-1],))
        return pred, score

    def classify_file(self, filename):
        im = skio.imread(filename)
        im = im / 255
        pred, score = self.classify(im)
        return im, pred, score

    def upsample(self, pred, img_size):
        height = self.num_patches[0] * self.patch_size[0]
        width = self.num_patches[1] * self.patch_size[1]
        rescaled = np.repeat(np.repeat(pred, self.patch_size[0], axis=0), self.patch_size[1], axis=1)
        rescaled = np.pad(rescaled, [[0, img_size[0]-height], [0, img_size[1]-width]])
        return rescaled


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    pc = PatchClassifier(r"D:\Datasets\COTS\trained_models\COTS 204 v2 ResNet50 Cyclic TL (fast)_20200718-152919\model\network_info.xml")
    im, pred, score = pc.classify_file(r"D:\Datasets\COTS\test_images\20200715_102427_GOPR0002_frame_000995.jpg")
    import cv2
    cv2.imshow("image", cv2.cvtColor((im*255).astype(np.uint8), cv2.COLOR_RGB2BGR))

    plt.imshow(im)
    plt.show()
    plt.imshow(pred)
    plt.show()
    pred = pc.upsample(pred, im.shape)
    plt.imshow(pred)
    plt.show()
    pred = pc.upsample(score[:,:,1], im.shape)
    plt.imshow(pred)
    plt.show()


    import imageio
    import cv2

    vid = imageio.get_reader(r"D:\Datasets\video.mp4", 'ffmpeg')
    vid_out = imageio.get_writer(r"D:\Datasets\video_processed_2.mp4", fps=25)
    prev_score = None
    for idx, frame in enumerate(vid.iter_data()):
        print(idx)
        if idx < 20*60*3:
            continue
        im = frame.copy() / 255
        pred, score = pc.classify(im)
        pred_COTS = pc.upsample(score[:, :, 0], im.shape)
        if prev_score is None:
            print('*')
            prev_score = pred_COTS
        else:
            prev_score = pred_COTS
            print(prev_score.max())
        red = np.zeros((*prev_score.shape[0:2], 3))
        red[:, :, 2] = prev_score
        im = im + red * 0.5
        im[im > 1] = 1
        im = im * 255
        im = im.astype(np.uint8)
        vid_out.append_data(im)
        cv2.imshow("image", cv2.cvtColor(im[::2,::2,:], cv2.COLOR_RGB2BGR))
        if cv2.waitKey(1) == ord('q'):
            break
    cv2.destroyAllWindows()
    vid_out.close()

    # vid = imageio.get_reader("D:\Datasets\GOPR0003.MP4", 'ffmpeg')
    # vid_out = imageio.get_writer("D:\Datasets\GOPR0003_processed.mp4", fps=25)
    # for idx, frame in enumerate(vid.iter_data()):
    #     print(idx)
    #     # if idx < 1000:
    #     #     continue
    #     im = frame.copy() / 255
    #     pred, score = pc.classify(im)
    #     pred_COTS = pc.upsample(score[:, :, 0], im.shape)
    #     im = im + np.repeat(pred_COTS[..., np.newaxis], 3, axis=-1) * 0.2
    #     im[im > 1] = 1
    #     im = im * 255
    #     im = im.astype(np.uint8)
    #     vid_out.append_data(im)
    # vid_out.close()

    # vid = imageio.get_reader("D:\Datasets\GOPR0004.MP4", 'ffmpeg')
    # vid_out = imageio.get_writer("D:\Datasets\GOPR0004_processed.mp4", fps=25)
    # for idx, frame in enumerate(vid.iter_data()):
    #     print(idx)
    #     # if idx < 1000:
    #     #     continue
    #     im = frame.copy() / 255
    #     pred, score = pc.classify(im)
    #     pred_COTS = pc.upsample(score[:, :, 0], im.shape)
    #     im = im + np.repeat(pred_COTS[..., np.newaxis], 3, axis=-1) * 0.2
    #     im[im > 1] = 1
    #     im = im * 255
    #     im = im.astype(np.uint8)
    #     vid_out.append_data(im)
    # vid_out.close()

    # vid = imageio.get_reader(r"D:\Datasets\video.mp4", 'ffmpeg')
    # vid_out = imageio.get_writer(r"D:\Datasets\video_processed.mp4", fps=25)
    # for idx, frame in enumerate(vid.iter_data()):
    #     print(idx)
    #     # if idx < 1000:
    #     #     continue
    #     im = frame.copy() / 255
    #     pred, score = pc.classify(im)
    #     pred_COTS = pc.upsample(score[:, :, 0], im.shape)
    #     im = im + np.repeat(pred_COTS[..., np.newaxis], 3, axis=-1) * 0.2
    #     im[im > 1] = 1
    #     im = im * 255
    #     im = im.astype(np.uint8)
    #     vid_out.append_data(im)
    # vid_out.close()
