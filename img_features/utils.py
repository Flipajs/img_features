import numpy as np
from pkg_resources import resource_filename, Requirement
import cPickle as pickle


class ColorNaming:
    w2c = None

    def __init__(self):
        pass

    @staticmethod
    def __load_w2c_pkl():
        with open(resource_filename(__name__, "data/w2c.pkl")) as f:
            return pickle.load(f)

    @staticmethod
    def im2colors(im, out_type='color_names'):
        """
        out_type:
            'color_names': returns np.array((im.shape[0], im.shape[1]), dtype=np.uint8) with ids of one of 11 colors
            'probability_vector': returns np.array((im.shape[0], im.shape[1], 11), stype=np.float) with probability
                of each color

        NOTE: first call might take a while as the lookup table is being loaded...

        :param im:
        :param w2c:
        :param out_type:
        :return:
        """

        # color_values = {[0 0 0], [0 0 1], [.5 .4 .25], [.5 .5 .5], [0 1 0], [1 .8 0], [1 .5 1], [1 0 1], [1 0 0], [1 1 1],
        #               q  [1 1 0]};

        if ColorNaming.w2c is None:
            ColorNaming.w2c = ColorNaming.__load_w2c_pkl()

        im = np.asarray(im, dtype=np.float)

        h, w = im.shape[0], im.shape[1]

        RR = im[:, :, 0].ravel()
        GG = im[:, :, 1].ravel()
        BB = im[:, :, 2].ravel()

        index_im = np.asarray(np.floor(RR / 8) + 32 * np.floor(GG / 8) + 32 * 32 * np.floor(BB / 8), dtype=np.uint)

        if out_type == 'colored_image':
            pass
        elif out_type == 'probability_vector':
            out = ColorNaming.w2c[index_im].reshape((h, w, 11))
        else:
            w2cM = np.argmax(ColorNaming.w2c, axis=1)
            out = np.asarray(w2cM[index_im], dtype=np.uint8)
            out.shape = (h, w)

        return out


def __mat2pkl(path, name):
    from scipy.io import loadmat
    import cPickle as pickle

    w2c = loadmat(path + '/' + name + '.mat')['w2c']
    with open(path + '/' + name + '.pkl', 'w') as f:
        pickle.dump(w2c, f)


def im2colors(im, out_type='color_names'):
    return ColorNaming.im2colors(im, out_type)


def feature_descriptor(im, block_division=(2, 2), pyramid_levels=3, histogram_density=False):
    """
    Calls im2colors(im, out_type='color_names') and computes histograms (with 11 bins, given by 11 color names) on
    different pyramid levels based on grid.
    Parameters
    ----------
    im : np.array() 3channels image
    block_division : tuple(uint, uint), (3, 2) means on next pyramid level divide block height by 3 and width by 2
        (produces 6x more blocks)
    pyramid_levels : uint, 2 means on the first level compute a histogram of the whole image, then divide it according
        to block_division.
    histogram_density : bool, If ``False``, the result will contain the number of samples in
        each bin. If ``True``, the result is the value of the
        probability *density* function at the bin, normalized such that
        the *integral* over the range is 1 (numpy.histogram is used)
    Returns
    -------
    np.array of shape(n, ) histogram values arranged level by level, in each level row by row in given block arrangement

    """

    # it is given by 11 color names
    histogram_num_bins = 11

    pyramid_levels = int(pyramid_levels)
    if pyramid_levels < 1:
        raise Exception("number of pyramid levels cannot by smaller than 1")

    # get color naming
    cm = im2colors(im)
    h, w = cm.shape

    # allocate feature vector
    t_ = np.float if histogram_density else np.uint
    n = 0
    current_block_num = np.array((1, 1))
    for lvl in range(pyramid_levels):
        n += np.product(current_block_num) * histogram_num_bins
        current_block_num = np.multiply(current_block_num, np.array(block_division))

    features = np.empty((n, ), dtype=t_)

    current_block_num = np.array((1, 1))
    # pointer
    fp_ = 0
    for lvl in range(pyramid_levels):
        rs = np.linspace(0, h, current_block_num[0] + 1, dtype=np.int32)
        cs = np.linspace(0, w, current_block_num[1] + 1, dtype=np.int32)

        for row in range(current_block_num[0]):
            for col in range(current_block_num[1]):
                # TODO: is there a better way without .copy()? we cannot change shape for non-contiguous array...
                cm_part = cm[rs[row]:rs[row+1], cs[col]:cs[col+1]].copy()
                cm_part.shape = (cm_part.shape[0]*cm_part.shape[1], )
                hist_, _ = np.histogram(cm_part, bins=histogram_num_bins, density=histogram_density)
                features[fp_:fp_+histogram_num_bins] = np.asarray(hist_, dtype=t_)

                fp_ += histogram_num_bins

        # divide into smaller blocks
        current_block_num = np.multiply(current_block_num, np.array(block_division))

    return features


if __name__ == '__main__':
    import cPickle as pickle
    from scipy.misc import imread

    # __mat2pkl('data', 'w2c')

    im = imread('data/car.jpg')

    f = feature_descriptor(im)
    f2 = feature_descriptor(im, histogram_density=True)

    # load lookup table
    with open('data/w2c.pkl') as f:
        w2c = pickle.load(f)

    import time

    time1 = time.time()
    ColorNaming.im2c(im, out_type='probability_vector')
    print time.time() - time1

    time1 = time.time()
    ColorNaming.im2c(im, out_type='probability_vector')
    print time.time() - time1

    time1 = time.time()
    ColorNaming.im2c(im, out_type='probability_vector')
    print time.time() - time1
