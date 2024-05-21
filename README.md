# Maternal-ECG-Guided-Net (MEG-Net)
Historically, acquiring a reliable and accurate non-invasive fetal electrocardiogram has
several significant challenges in both data acquisition and attenuation of maternal
signals. These barriers include maternal physical/physiological parameters, hardware
sensitivity, and the effectiveness of signal processing algorithms in separating maternal
and fetal electrocardiograms. In this paper, we focus on the evaluation of signalprocessing
algorithms. Here, we propose a learning-based method based on the
integration of maternal electrocardiogram acquired as guidance for transabdominal
fetal electrocardiogram signal extraction. The results demonstrate that incorporating
the maternal electrocardiogram signal as input for training the neural network
outperforms the network solely trained using information from the abdominal
electrocardiogram. This indicates that leveraging the maternal electrocardiogram
serves as a suitable prior for effectively attenuating maternal electrocardiogram from
the abdominal electrocardiogram.

<p align="center"><img src="/images/meg-net.svg" width="40%">
