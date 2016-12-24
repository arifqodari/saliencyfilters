from saliencyfilters import SaliencyFilters
from sys import argv
from skimage import io
import time


def main(input_image, output_image):
    start_time = time.time()

    sf = SaliencyFilters()
    image = io.imread(input_image)
    saliency = sf.compute_saliency(image)

    end_time = time.time() - start_time
    print 'time needed to compute saliency map', end_time

    io.imsave(output_image, (saliency * 255).astype('uint8'))


if __name__ == '__main__':
    main(argv[1], argv[2])
