saliencyfilters: A Python implementation of the Saliency Filters method
=======================================================================

saliencyfilters is a Python implementation of the Saliency Filters
method, which is an algorithm to estimate salient region on images.

References
----------

Perazzi, Federico, et al. "Saliency filters: Contrast based filtering
for salient region detection." Computer Vision and Pattern Recognition
(CVPR), 2012 IEEE Conference on. IEEE, 2012.

Required Packages
-----------------

-  numpy
-  scipy
-  scikit-image

Installation
------------

::

    pip install -r requirements.txt
    python setup.py install

Usage
-----

Sample code is provided in ``demo.py``.

::

    python demo.py sample.jpg output.jpg

Notes
-----

Our implementation does not use Permutohedral Lattice embedding as
stated in the paper. Hence, the code requires :math:`O(N^2)` operations.
The results are pretty much similar but a little bit slower in terms of
running time.
