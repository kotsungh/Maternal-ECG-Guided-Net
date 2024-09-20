# Maternal-ECG-Guided-Net (MEG-Net)
Acquiring a reliable and accurate non-invasive fetal electrocardiogram has a number of significant challenges in both data acquisition and attenuation of the maternal signals. These barriers include maternal physical/physiological parameters, hardware sensitivity, and the effectiveness of signal processing algorithms in separating the maternal and fetal electrocardiograms. In this paper, we focus on the evaluation of signal processing algorithms. Here, we propose a learning-based method based on the integration of maternal electrocardiogram acquired from the chest as guidance for transabdominal fetal electrocardiogram signal extraction. The results demonstrate that incorporating the maternal electrocardiogram signal as input for training the neural network outperforms the network solely trained using information from the abdominal electrocardiogram. This indicates that leveraging the maternal QRS (MQRS) complex serves as a suitable prior for effectively attenuating maternal electrocardiogram from the abdominal electrocardiogram.

K.-T. Hsu, T. Nguyen, A. Krishnan, R. Govindan, and R. Shekhar, “Maternal ECG-guided neural network for improved fetal electrocardiogram extraction.” Biomedical Signal Processing and Control, Sep. 12, 2024. https://doi.org/10.1016/j.bspc.2024.106793.
