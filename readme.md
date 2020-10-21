# Butterfly-Net: Spatial-Temporal Architecture for Medical Image Segmentation

by
Tetiana Klymenko*,
Seong Tae Kim*,
Kirsten Lauber†,
Christopher Kurz†,
Guillaume Landry†,
Nassir Navab*,
Shadi Albarqouni*‡


> *Computer Aided Medical Procedures, Technical University of Munich, Munich, Germany  
>†Medical Center of the University of Munich, Munich, Germany  
>‡Helmholtz AI, Helmholtz Center Munich, Neuherberg, Germany  

## Abstract

> Radiation therapy tries to maximize the effect of radiation on the tumor and minimize its influence on adjacent tissues.
However, it highly depends on the accuracy of the tumor segmentation on the planning CT or MRI images. Tumor contouring today is carried out exclusively with significant contribution of medical specialists, which stands as a high time-consuming process, prone to inter%-/intra
-observer variation that can affect the reliability of the outcome. 
Existing methods for automatic tumor segmentation can reduce the influence of these factors, but are not completely reliable and leave a lot of room for improvement. In this work, we exploit spatio-temporal information from the longitudinal CT scans %of mice 
to improve the deep neural network for tumor segmentation. For this purpose, we devise a novel volumetric spatio-temporal memory network, Butterfly-Net, which stores the previous scan information and reads for the segmentation at the target time point. Moreover, the effect of clinical factors is investigated in the framework of our volumetric spatio-temporal memory network. 
Experimental results on our longitudinal CT scans show that our model could effectively utilize temporal information and clinical factors for tumor segmentation.