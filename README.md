# Maternal-ECG-Guided-Net
Acquiring a reliable and accurate non-invasive fetal
electrocardiogram has a number of significant challenges
historically in both data acquisition and attenuation of the
interfering maternal cardiogram. These factors encompass
maternal physical/physiological parameters, hardware sensitivity,
and the effectiveness of signal processing algorithms in separating
the maternal and fetal electrocardiograms. In this paper, we defer
the discussion of how maternal parameters and hardware
sensitivity affect the extraction of fetal electrocardiogram and
instead focus on the evaluation of signal processing algorithms.
Here, we propose a learning-based method based on the explicit
integration of maternal electrocardiogram acquired from the
chest as guidance for transabdominal fetal electrocardiogram
signal extraction. The results demonstrate that incorporating the
maternal electrocardiogram signal as input for training the neural
network outperforms the network solely trained using information
from the abdominal electrocardiogram. This indicates that
leveraging the precise information from the maternal QRS
(MQRS) complex serves as a suitable prior for effectively
attenuating maternal electrocardiogram from the abdominal
electrocardiogram.

<p align="center"><img src="/images/meg-net.svg" width="60%">
