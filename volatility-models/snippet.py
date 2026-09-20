"""extra, from the write-up at
https://davidariasfinance.com/scripts/volatility-models/
"""

WARNING:tensorflow: TensorFlow GPU support is not available on native Windows
for TensorFlow >= 2.11. Even if CUDA/cuDNN are installed, GPU will not be used.
Please use WSL2 or the TensorFlow-DirectML plugin.

>>> tf.test.is_built_with_cuda()
False
>>> tf.config.list_physical_devices()
['CPU']          # the 4050 is invisible by design, not by fault
