# Encoder Decoder with multi-head Attention layers for multi-channel Climate Downscaling (EDA)
An Encoder-Decoder model of Climate Downscaling of ERA5 </br>

training data: ERA5 in Taiwan region with the shape of (14,9)</br> 
auxiliary data: ERA5 10 meter high wind field vector u,v near Taiwan region with the shape of (14,9) and low-resolution(downsampled) label data</br>
label data: TCCIP grid daily precipitation with the shape of (70,45) (resized)</br>

# Model Architecture
Encoder involves Multi-head Attention Layer and Fully Connected Layer (FC). </br>
Decoder involves Upsampling Layer with Efficiency Sub-pixel method (of ESPCN) and convolutional layers. </br>
![image](https://github.com/AugChiang/EnDeAux_Climate_Downscaling/blob/main/arch2.jpg)

# Model Prediction

Prediction of precipitation on 2019.08.15 </br>
left-top: ERA5 Reanalysis Data of Precipitation </br>
left-bottom: TCCIP gridded observation </br>
right-top: MAE error heatmap </br>
right-bottom: model prediction </br>
![image](https://github.com/AugChiang/EnRe3_Climate_Downscaling/blob/main/res/20190815.png)

# Metrics
Evaluation on metrics of pixel-wise error: mean absolute error (MAE) and root mean square error (RMSE). </br>
And relationship: Pearson Correletaion (Corr) and Structural Similarity Index (SSIM). </br>
![image](https://github.com/AugChiang/EnRe3_Climate_Downscaling/blob/main/res/metric.png)

Compared to traditional statistical downscaling method, BCSD (Bias-Corrected Spatial Disaggregation): </br>
![image](https://github.com/AugChiang/EnRe3_Climate_Downscaling/blob/main/res/bcsd_metric.png)

# Environment
tensorflow==2.14.0+nv23.11 </br>
numpy==1.24.4 </br>
pandas==1.5.3 </br>
(last updated: 2024/02/21) </br>
