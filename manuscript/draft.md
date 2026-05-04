ManuscriptAbstract 
Solar irradiance forecasting is crucial in the transition towards solar energy, specifically in the process of maximizing irradiance input and making solar technologies more efficient at capturing the sun’s energy. However, the process of training deep-learning models to forecast irradiance is limited by the lack of high quality data. Existing datasets are either low in frequency, resolution, or lack pairings with ground truth measurements. In this work, we present an upgraded version of the sky imager dataset from the Solar Radiation Research Laboratory Baseline Measurement System (SRRL BMS) that addresses all three aforementioned limitations. The SRRL BMS dataset itself possesses high resolution images, with each image containing 1536 × 1536 pixels. Additionally, the dataset contains high quality ground truth measurements of irradiance, such as Direct Normal Irradiance (DNI), Diffuse Horizontal Irradiance (DHI), and Global Horizontal Irradiance (GHI). It also contains various other meteorological measurements to help train forecasting models such as sun angles, temperature, wind speed and direction, pressure, relative humidity, etc. 

However, a major limitation with this current dataset is that it only saves an image once every 10 minutes, which misaligns with the rest of the measurements that are stored every minute. Beyond just a mismatch in the data types, the low sampling frequency of the images adds complications to model training, as rapid cloud movements that occur on sub-minute scales are unobserved. In contrast, our dataset stores sky images by the minute, taken by the same camera used in the original dataset. This upgraded sky imagery dataset can be used to train irradiance forecast models, using the images in conjunction with ground truth measurements for both inputs and target predictions. 
1. Introduction
Solar energy is recognized to hold tremendous potential for the transition to green energy when compared to other major forms of energy generation. Hydropower is limited in scope due to having to be situated in a region with abundant natural rainfall, and most of these sites have already been developed [1]. Geothermal energy production is risky and limited because of induced seismic injections and high costs associated with the initial drilling process [2]. Wind energy generation suffers from remoteness, since urban sites have too many structures that obstruct and reduce wind speeds [3]. Thus, solar energy emerges as one of the most flexible and accessible forms of energy. Solar irradiance is available pretty much globally, including densely populated regions [4], and solar panels can easily integrate with urban rooftops to capture irradiance. 

A notable challenge with using solar energy, however,  is variability. Solar irradiance can fluctuate rapidly depending on atmospheric conditions such as cloud coverage, precipitation, temperature, and seasonal changes in sun angle. This variability makes it hard to predict how much solar energy can be extracted [5, 6]. Thus, in order to maintain grid stability and optimize solar energy generation, forecasting models need to be highly accurate in predicting irradiance values. Achieving this elite accuracy requires forecasting models to be trained on high temporal resolution datasets [7], with these datasets ideally containing sky images paired with ground truth measurements [8]. 

Many existing sky image datasets are limited by one of the following categories: frequency, resolution, insufficient ground truth measurements. The ideal dataset to be used to train irradiance forecasting models should have minute resolution images, a reasonably high resolution, and should have ground truth measurements that don't just have irradiance values but also other atmospheric attributes such as the sun angle, cloud coverage, precipitation, and temperature. An existing dataset that almost fits in perfectly with such attributes is the SRRL ASI-16 Sky Imager Gallery, as it has high resolution images (1536 × 1536) and can be paired with ground truth measurements from the SRRL Irradiance Inc. RSP v2 data files. However, a major problem with this data is that while the ground truth measurements are recorded by the minute, the sky images are saved only once every ten minutes, misaligning with the rest of the data. 

This paper aims to create a novel dataset that achieves high frequency, high resolution, and is paired with accurate ground truth measurements. [Explain following sections]
2. Methodology [make it brief, <1 page]
2.1 Image Scraping
Despite the fact that the SRRL ASI-16 Sky Imager Gallery only saves a sky image once every 10 minutes, the camera itself takes and temporarily saves a picture once every minute. This presents an opportunity to scrape the minute-resolution images and save them before the next picture is taken and the current one is deleted. 

Script #1: NrelScraper.py
[cite ghub repo + cite jetstream] -> metadata -> TDR as zip
Scripts freely avail here [ghub]
This script scrapes the images from the NREL ASI-16 EKO camera once every minute. It imports the request library to send out requests and receive the images from the data access site. 





3. Usage
How to access the data
The dataset used in this study is hosted on Hugging Face, a widely used platform for sharing machine learning datasets and models. 

First, visit the Hugging Face dataset at https://huggingface.co/datasets/knl2366/NREL_Sky_Imagery. From there, there are a variety of options to explore the dataset. The default view has all the sky photos listed for a preview. For a more organized and systematic view, navigate to the files and versions tab, where the folder hierarchy is shown (also called a tree). The folder structure in place is as follows: year/month/day. The files themselves are named based on the time they were retrieved with the following format: year/month/day-hour/second/millisecond. For instance, a file named “20261101-114633” would have been downloaded from NREL on November 1st, 2026, at precisely 11:46:33 MT. To manually download these files, just click on them and click on the download option. The next few sections will be on how to download them programmatically. 

Script to download data + timeframe
HFScraper.py 

This is the script you will need to download all the data automatically. In order to run this script a few conditions will have to be met. First, you need to have Python version 3.12 or later. You need to download huggingface_hub by running the following command in your terminal of choice: pip install huggingface_hub. To do anything with Hugging Face’s interface, you also need to get an API token, which can be made from the “Access Tokens” page. From there, paste your API token in the script where it says “HF_TOKEN.” Run the script and it will download all the files. 

Script to check dataset validity
HFValidData.py 
4. Analysis [Show completeness (heatmap)]


10 vs 1 picture
Tabular analysis
Simple model -> image every minute == more accuracy
WIP
Times with missing images
Script for pytorch; import data
5. Conclusion

6. Demo



 
Citations

1 Solar incred important > hydro bc hydro = already dev 

B. Kroposki et al., "Achieving a 100% Renewable Grid: Operating Electric Power Systems with Extremely High Levels of Variable Renewable Energy," in IEEE Power and Energy Magazine, vol. 15, no. 2, pp. 61-73, March-April 2017, doi: 10.1109/MPE.2016.2637122 

2 Geothermal sucks → high costs + seismic activities

Alberto Boretti, Enhanced geothermal systems: Potential, challenges, and a realistic path to integration in a sustainable energy future, Next Energy, Volume 8, 2025, 100332, ISSN 2949-821X, https://doi.org/10.1016/j.nxener.2025.100332. (https://www.sciencedirect.com/science/article/pii/S2949821X2500095X) 

3 Wind sucks → remoteness

R.K. Reja, Ruhul Amin, Zinat Tasneem, Md. Firoj Ali, Md. Robiul Islam, Dip Kumar Saha, Faisal Rahman Badal, Md. Hafiz Ahamed, Sumaya Ishrat Moyeen, Sajal Kumar Das, A review of the evaluation of urban wind resources: challenges and perspectives, Energy and Buildings, Volume 257, 2022, 111781, ISSN 0378-7788, https://doi.org/10.1016/j.enbuild.2021.111781.
(https://www.sciencedirect.com/science/article/pii/S0378778821010653) 

4 Solar is accessible

Remus Prăvălie, Cristian Patriche, Georgeta Bandoc, Spatial assessment of solar energy potential at global scale. A geographical approach, Journal of Cleaner Production, Volume 209, 2019, Pages 692-721, ISSN 0959-6526, https://doi.org/10.1016/j.jclepro.2018.10.239. (https://www.sciencedirect.com/science/article/pii/S0959652618332657) 

5 Solar power crucial to grid stability → through forecasting (seconds, days, weeks) + new approaches every year, thus need good data

J. Antonanzas, N. Osorio, R. Escobar, R. Urraca, F.J. Martinez-de-Pison, F. Antonanzas-Torres, Review of photovoltaic power forecasting, Solar Energy, Volume 136, 2016, Pages 78-111, ISSN 0038-092X, https://doi.org/10.1016/j.solener.2016.06.069. (https://www.sciencedirect.com/science/article/pii/S0038092X1630250X)

6 Variability necessitates high accuracy forecast models

N.Y. Hendrikx, K. Barhmi, L.R. Visser, T.A. de Bruin, M. Pó, A.A. Salah, W.G.J.H.M. van Sark,
All sky imaging-based short-term solar irradiance forecasting with Long Short-Term Memory networks, Solar Energy, Volume 272, 2024, 112463, ISSN 0038-092X, https://doi.org/10.1016/j.solener.2024.112463. (https://www.sciencedirect.com/science/article/pii/S0038092X24001579) 

7 High frequency == key

T. A. Siddiqui, S. Bharadwaj and S. Kalyanaraman, "A Deep Learning Approach to Solar-Irradiance Forecasting in Sky-Videos," 2019 IEEE Winter Conference on Applications of Computer Vision (WACV), Waikoloa, HI, USA, 2019, pp. 2166-2174, doi: 10.1109/WACV.2019.00234. 

8 Ground based images key esp if paired with ground measurements from sensors

Yuhao Nie, Quentin Paletta, Andea Scott, Luis Martin Pomares, Guillaume Arbod, Sgouris Sgouridis, Joan Lasenby, Adam Brandt, Sky image-based solar forecasting using deep learning with heterogeneous multi-location data: Dataset fusion versus transfer learning, Applied Energy, Volume 369, 2024, 123467, ISSN 0306-2619, https://doi.org/10.1016/j.apenergy.2024.123467.
(https://www.sciencedirect.com/science/article/pii/S030626192400850X) 

