---
title: 'Practical Machine Learning: Project'
author: "JoeriW"
date: "Saturday, May 23, 2015"
output: html_document
---

##1. Introduction

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, our goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset). 


The goal of this project is to create a machine learning algorithm that tries to predict the fashion in which participants to the experiment performed a barbell lift. The 5 possible fashions are:

A. exactly according to specification

B. throwing the elbows to the front 

C. lifting the dumbell only halfway 

D. lowering the dumbell only halfway 

E. throwing the hips to the front 


##2. Preparatory phase

load required packages:


```r
library(caret)
```

```
## Warning: package 'caret' was built under R version 3.1.3
```

```
## Loading required package: lattice
## Loading required package: ggplot2
## Need help? Try the ggplot2 mailing list: http://groups.google.com/group/ggplot2.
```

```r
library(gbm)
```

```
## Warning: package 'gbm' was built under R version 3.1.3
```

```
## Loading required package: survival
## Loading required package: splines
## 
## Attaching package: 'survival'
## 
## The following object is masked from 'package:caret':
## 
##     cluster
## 
## Loading required package: parallel
## Loaded gbm 2.1.1
```

```r
library(knitr)
```

Set the seed in order to allow reproducibility


```r
set.seed(666)
```


##3. Loading the data

download the data and store in a training and testing variable:


```r
if(!file.exists("pml_training.csv")){
        download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv","pml_training.csv")
}
if(!file.exists("pml_testing.csv")){
        download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv","pml_testing.csv")
}
training <- read.csv("pml_training.csv")
testing <- read.csv("pml_testing.csv")
```

##4. Pre-processing

Look at the dimension of the training data set.


```r
dim(training)
```

```
## [1] 19622   160
```

When looking at the dimensions of the data set, we notice an extensive amount of variables. Variables that exhibit zero or near zero variance will not contribute to our model and therefore can be removed.


```r
nzvCols <- nearZeroVar(training)
training <- training[,-nzvCols]
testing <- testing[,-nzvCols]
```

Look at the structure of our training data set


```r
summary(training) 
```

```
##        X            user_name    raw_timestamp_part_1 raw_timestamp_part_2
##  Min.   :    1   adelmo  :3892   Min.   :1.322e+09    Min.   :   294      
##  1st Qu.: 4906   carlitos:3112   1st Qu.:1.323e+09    1st Qu.:252912      
##  Median : 9812   charles :3536   Median :1.323e+09    Median :496380      
##  Mean   : 9812   eurico  :3070   Mean   :1.323e+09    Mean   :500656      
##  3rd Qu.:14717   jeremy  :3402   3rd Qu.:1.323e+09    3rd Qu.:751891      
##  Max.   :19622   pedro   :2610   Max.   :1.323e+09    Max.   :998801      
##                                                                           
##           cvtd_timestamp    num_window      roll_belt     
##  28/11/2011 14:14: 1498   Min.   :  1.0   Min.   :-28.90  
##  05/12/2011 11:24: 1497   1st Qu.:222.0   1st Qu.:  1.10  
##  30/11/2011 17:11: 1440   Median :424.0   Median :113.00  
##  05/12/2011 11:25: 1425   Mean   :430.6   Mean   : 64.41  
##  02/12/2011 14:57: 1380   3rd Qu.:644.0   3rd Qu.:123.00  
##  02/12/2011 13:34: 1375   Max.   :864.0   Max.   :162.00  
##  (Other)         :11007                                   
##    pitch_belt          yaw_belt       total_accel_belt max_roll_belt    
##  Min.   :-55.8000   Min.   :-180.00   Min.   : 0.00    Min.   :-94.300  
##  1st Qu.:  1.7600   1st Qu.: -88.30   1st Qu.: 3.00    1st Qu.:-88.000  
##  Median :  5.2800   Median : -13.00   Median :17.00    Median : -5.100  
##  Mean   :  0.3053   Mean   : -11.21   Mean   :11.31    Mean   : -6.667  
##  3rd Qu.: 14.9000   3rd Qu.:  12.90   3rd Qu.:18.00    3rd Qu.: 18.500  
##  Max.   : 60.3000   Max.   : 179.00   Max.   :29.00    Max.   :180.000  
##                                                        NA's   :19216    
##  max_picth_belt  min_roll_belt     min_pitch_belt  amplitude_roll_belt
##  Min.   : 3.00   Min.   :-180.00   Min.   : 0.00   Min.   :  0.000    
##  1st Qu.: 5.00   1st Qu.: -88.40   1st Qu.: 3.00   1st Qu.:  0.300    
##  Median :18.00   Median :  -7.85   Median :16.00   Median :  1.000    
##  Mean   :12.92   Mean   : -10.44   Mean   :10.76   Mean   :  3.769    
##  3rd Qu.:19.00   3rd Qu.:   9.05   3rd Qu.:17.00   3rd Qu.:  2.083    
##  Max.   :30.00   Max.   : 173.00   Max.   :23.00   Max.   :360.000    
##  NA's   :19216   NA's   :19216     NA's   :19216   NA's   :19216      
##  amplitude_pitch_belt var_total_accel_belt avg_roll_belt   
##  Min.   : 0.000       Min.   : 0.000       Min.   :-27.40  
##  1st Qu.: 1.000       1st Qu.: 0.100       1st Qu.:  1.10  
##  Median : 1.000       Median : 0.200       Median :116.35  
##  Mean   : 2.167       Mean   : 0.926       Mean   : 68.06  
##  3rd Qu.: 2.000       3rd Qu.: 0.300       3rd Qu.:123.38  
##  Max.   :12.000       Max.   :16.500       Max.   :157.40  
##  NA's   :19216        NA's   :19216        NA's   :19216   
##  stddev_roll_belt var_roll_belt     avg_pitch_belt    stddev_pitch_belt
##  Min.   : 0.000   Min.   :  0.000   Min.   :-51.400   Min.   :0.000    
##  1st Qu.: 0.200   1st Qu.:  0.000   1st Qu.:  2.025   1st Qu.:0.200    
##  Median : 0.400   Median :  0.100   Median :  5.200   Median :0.400    
##  Mean   : 1.337   Mean   :  7.699   Mean   :  0.520   Mean   :0.603    
##  3rd Qu.: 0.700   3rd Qu.:  0.500   3rd Qu.: 15.775   3rd Qu.:0.700    
##  Max.   :14.200   Max.   :200.700   Max.   : 59.700   Max.   :4.000    
##  NA's   :19216    NA's   :19216     NA's   :19216     NA's   :19216    
##  var_pitch_belt    avg_yaw_belt      stddev_yaw_belt    var_yaw_belt      
##  Min.   : 0.000   Min.   :-138.300   Min.   :  0.000   Min.   :    0.000  
##  1st Qu.: 0.000   1st Qu.: -88.175   1st Qu.:  0.100   1st Qu.:    0.010  
##  Median : 0.100   Median :  -6.550   Median :  0.300   Median :    0.090  
##  Mean   : 0.766   Mean   :  -8.831   Mean   :  1.341   Mean   :  107.487  
##  3rd Qu.: 0.500   3rd Qu.:  14.125   3rd Qu.:  0.700   3rd Qu.:    0.475  
##  Max.   :16.200   Max.   : 173.500   Max.   :176.600   Max.   :31183.240  
##  NA's   :19216    NA's   :19216      NA's   :19216     NA's   :19216      
##   gyros_belt_x        gyros_belt_y       gyros_belt_z    
##  Min.   :-1.040000   Min.   :-0.64000   Min.   :-1.4600  
##  1st Qu.:-0.030000   1st Qu.: 0.00000   1st Qu.:-0.2000  
##  Median : 0.030000   Median : 0.02000   Median :-0.1000  
##  Mean   :-0.005592   Mean   : 0.03959   Mean   :-0.1305  
##  3rd Qu.: 0.110000   3rd Qu.: 0.11000   3rd Qu.:-0.0200  
##  Max.   : 2.220000   Max.   : 0.64000   Max.   : 1.6200  
##                                                          
##   accel_belt_x       accel_belt_y     accel_belt_z     magnet_belt_x  
##  Min.   :-120.000   Min.   :-69.00   Min.   :-275.00   Min.   :-52.0  
##  1st Qu.: -21.000   1st Qu.:  3.00   1st Qu.:-162.00   1st Qu.:  9.0  
##  Median : -15.000   Median : 35.00   Median :-152.00   Median : 35.0  
##  Mean   :  -5.595   Mean   : 30.15   Mean   : -72.59   Mean   : 55.6  
##  3rd Qu.:  -5.000   3rd Qu.: 61.00   3rd Qu.:  27.00   3rd Qu.: 59.0  
##  Max.   :  85.000   Max.   :164.00   Max.   : 105.00   Max.   :485.0  
##                                                                       
##  magnet_belt_y   magnet_belt_z       roll_arm         pitch_arm      
##  Min.   :354.0   Min.   :-623.0   Min.   :-180.00   Min.   :-88.800  
##  1st Qu.:581.0   1st Qu.:-375.0   1st Qu.: -31.77   1st Qu.:-25.900  
##  Median :601.0   Median :-320.0   Median :   0.00   Median :  0.000  
##  Mean   :593.7   Mean   :-345.5   Mean   :  17.83   Mean   : -4.612  
##  3rd Qu.:610.0   3rd Qu.:-306.0   3rd Qu.:  77.30   3rd Qu.: 11.200  
##  Max.   :673.0   Max.   : 293.0   Max.   : 180.00   Max.   : 88.500  
##                                                                      
##     yaw_arm          total_accel_arm var_accel_arm     gyros_arm_x      
##  Min.   :-180.0000   Min.   : 1.00   Min.   :  0.00   Min.   :-6.37000  
##  1st Qu.: -43.1000   1st Qu.:17.00   1st Qu.:  9.03   1st Qu.:-1.33000  
##  Median :   0.0000   Median :27.00   Median : 40.61   Median : 0.08000  
##  Mean   :  -0.6188   Mean   :25.51   Mean   : 53.23   Mean   : 0.04277  
##  3rd Qu.:  45.8750   3rd Qu.:33.00   3rd Qu.: 75.62   3rd Qu.: 1.57000  
##  Max.   : 180.0000   Max.   :66.00   Max.   :331.70   Max.   : 4.87000  
##                                      NA's   :19216                      
##   gyros_arm_y       gyros_arm_z       accel_arm_x       accel_arm_y    
##  Min.   :-3.4400   Min.   :-2.3300   Min.   :-404.00   Min.   :-318.0  
##  1st Qu.:-0.8000   1st Qu.:-0.0700   1st Qu.:-242.00   1st Qu.: -54.0  
##  Median :-0.2400   Median : 0.2300   Median : -44.00   Median :  14.0  
##  Mean   :-0.2571   Mean   : 0.2695   Mean   : -60.24   Mean   :  32.6  
##  3rd Qu.: 0.1400   3rd Qu.: 0.7200   3rd Qu.:  84.00   3rd Qu.: 139.0  
##  Max.   : 2.8400   Max.   : 3.0200   Max.   : 437.00   Max.   : 308.0  
##                                                                        
##   accel_arm_z       magnet_arm_x     magnet_arm_y     magnet_arm_z   
##  Min.   :-636.00   Min.   :-584.0   Min.   :-392.0   Min.   :-597.0  
##  1st Qu.:-143.00   1st Qu.:-300.0   1st Qu.:  -9.0   1st Qu.: 131.2  
##  Median : -47.00   Median : 289.0   Median : 202.0   Median : 444.0  
##  Mean   : -71.25   Mean   : 191.7   Mean   : 156.6   Mean   : 306.5  
##  3rd Qu.:  23.00   3rd Qu.: 637.0   3rd Qu.: 323.0   3rd Qu.: 545.0  
##  Max.   : 292.00   Max.   : 782.0   Max.   : 583.0   Max.   : 694.0  
##                                                                      
##  max_picth_arm       max_yaw_arm     min_yaw_arm    amplitude_yaw_arm
##  Min.   :-173.000   Min.   : 4.00   Min.   : 1.00   Min.   : 0.00    
##  1st Qu.:  -1.975   1st Qu.:29.00   1st Qu.: 8.00   1st Qu.:13.00    
##  Median :  23.250   Median :34.00   Median :13.00   Median :22.00    
##  Mean   :  35.751   Mean   :35.46   Mean   :14.66   Mean   :20.79    
##  3rd Qu.:  95.975   3rd Qu.:41.00   3rd Qu.:19.00   3rd Qu.:28.75    
##  Max.   : 180.000   Max.   :65.00   Max.   :38.00   Max.   :52.00    
##  NA's   :19216      NA's   :19216   NA's   :19216   NA's   :19216    
##  roll_dumbbell     pitch_dumbbell     yaw_dumbbell      max_roll_dumbbell
##  Min.   :-153.71   Min.   :-149.59   Min.   :-150.871   Min.   :-70.10   
##  1st Qu.: -18.49   1st Qu.: -40.89   1st Qu.: -77.644   1st Qu.:-27.15   
##  Median :  48.17   Median : -20.96   Median :  -3.324   Median : 14.85   
##  Mean   :  23.84   Mean   : -10.78   Mean   :   1.674   Mean   : 13.76   
##  3rd Qu.:  67.61   3rd Qu.:  17.50   3rd Qu.:  79.643   3rd Qu.: 50.58   
##  Max.   : 153.55   Max.   : 149.40   Max.   : 154.952   Max.   :137.00   
##                                                         NA's   :19216    
##  max_picth_dumbbell min_roll_dumbbell min_pitch_dumbbell
##  Min.   :-112.90    Min.   :-149.60   Min.   :-147.00   
##  1st Qu.: -66.70    1st Qu.: -59.67   1st Qu.: -91.80   
##  Median :  40.05    Median : -43.55   Median : -66.15   
##  Mean   :  32.75    Mean   : -41.24   Mean   : -33.18   
##  3rd Qu.: 133.22    3rd Qu.: -25.20   3rd Qu.:  21.20   
##  Max.   : 155.00    Max.   :  73.20   Max.   : 120.90   
##  NA's   :19216      NA's   :19216     NA's   :19216     
##  amplitude_roll_dumbbell amplitude_pitch_dumbbell total_accel_dumbbell
##  Min.   :  0.00          Min.   :  0.00           Min.   : 0.00       
##  1st Qu.: 14.97          1st Qu.: 17.06           1st Qu.: 4.00       
##  Median : 35.05          Median : 41.73           Median :10.00       
##  Mean   : 55.00          Mean   : 65.93           Mean   :13.72       
##  3rd Qu.: 81.04          3rd Qu.: 99.55           3rd Qu.:19.00       
##  Max.   :256.48          Max.   :273.59           Max.   :58.00       
##  NA's   :19216           NA's   :19216                                
##  var_accel_dumbbell avg_roll_dumbbell stddev_roll_dumbbell
##  Min.   :  0.000    Min.   :-128.96   Min.   :  0.000     
##  1st Qu.:  0.378    1st Qu.: -12.33   1st Qu.:  4.639     
##  Median :  1.000    Median :  48.23   Median : 12.204     
##  Mean   :  4.388    Mean   :  23.86   Mean   : 20.761     
##  3rd Qu.:  3.434    3rd Qu.:  64.37   3rd Qu.: 26.356     
##  Max.   :230.428    Max.   : 125.99   Max.   :123.778     
##  NA's   :19216      NA's   :19216     NA's   :19216       
##  var_roll_dumbbell  avg_pitch_dumbbell stddev_pitch_dumbbell
##  Min.   :    0.00   Min.   :-70.73     Min.   : 0.000       
##  1st Qu.:   21.52   1st Qu.:-42.00     1st Qu.: 3.482       
##  Median :  148.95   Median :-19.91     Median : 8.089       
##  Mean   : 1020.27   Mean   :-12.33     Mean   :13.147       
##  3rd Qu.:  694.65   3rd Qu.: 13.21     3rd Qu.:19.238       
##  Max.   :15321.01   Max.   : 94.28     Max.   :82.680       
##  NA's   :19216      NA's   :19216      NA's   :19216        
##  var_pitch_dumbbell avg_yaw_dumbbell   stddev_yaw_dumbbell
##  Min.   :   0.00    Min.   :-117.950   Min.   :  0.000    
##  1st Qu.:  12.12    1st Qu.: -76.696   1st Qu.:  3.885    
##  Median :  65.44    Median :  -4.505   Median : 10.264    
##  Mean   : 350.31    Mean   :   0.202   Mean   : 16.647    
##  3rd Qu.: 370.11    3rd Qu.:  71.234   3rd Qu.: 24.674    
##  Max.   :6836.02    Max.   : 134.905   Max.   :107.088    
##  NA's   :19216      NA's   :19216      NA's   :19216      
##  var_yaw_dumbbell   gyros_dumbbell_x    gyros_dumbbell_y  
##  Min.   :    0.00   Min.   :-204.0000   Min.   :-2.10000  
##  1st Qu.:   15.09   1st Qu.:  -0.0300   1st Qu.:-0.14000  
##  Median :  105.35   Median :   0.1300   Median : 0.03000  
##  Mean   :  589.84   Mean   :   0.1611   Mean   : 0.04606  
##  3rd Qu.:  608.79   3rd Qu.:   0.3500   3rd Qu.: 0.21000  
##  Max.   :11467.91   Max.   :   2.2200   Max.   :52.00000  
##  NA's   :19216                                            
##  gyros_dumbbell_z  accel_dumbbell_x  accel_dumbbell_y  accel_dumbbell_z 
##  Min.   : -2.380   Min.   :-419.00   Min.   :-189.00   Min.   :-334.00  
##  1st Qu.: -0.310   1st Qu.: -50.00   1st Qu.:  -8.00   1st Qu.:-142.00  
##  Median : -0.130   Median :  -8.00   Median :  41.50   Median :  -1.00  
##  Mean   : -0.129   Mean   : -28.62   Mean   :  52.63   Mean   : -38.32  
##  3rd Qu.:  0.030   3rd Qu.:  11.00   3rd Qu.: 111.00   3rd Qu.:  38.00  
##  Max.   :317.000   Max.   : 235.00   Max.   : 315.00   Max.   : 318.00  
##                                                                         
##  magnet_dumbbell_x magnet_dumbbell_y magnet_dumbbell_z  roll_forearm      
##  Min.   :-643.0    Min.   :-3600     Min.   :-262.00   Min.   :-180.0000  
##  1st Qu.:-535.0    1st Qu.:  231     1st Qu.: -45.00   1st Qu.:  -0.7375  
##  Median :-479.0    Median :  311     Median :  13.00   Median :  21.7000  
##  Mean   :-328.5    Mean   :  221     Mean   :  46.05   Mean   :  33.8265  
##  3rd Qu.:-304.0    3rd Qu.:  390     3rd Qu.:  95.00   3rd Qu.: 140.0000  
##  Max.   : 592.0    Max.   :  633     Max.   : 452.00   Max.   : 180.0000  
##                                                                           
##  pitch_forearm     yaw_forearm      max_picth_forearm min_pitch_forearm
##  Min.   :-72.50   Min.   :-180.00   Min.   :-151.00   Min.   :-180.00  
##  1st Qu.:  0.00   1st Qu.: -68.60   1st Qu.:   0.00   1st Qu.:-175.00  
##  Median :  9.24   Median :   0.00   Median : 113.00   Median : -61.00  
##  Mean   : 10.71   Mean   :  19.21   Mean   :  81.49   Mean   : -57.57  
##  3rd Qu.: 28.40   3rd Qu.: 110.00   3rd Qu.: 174.75   3rd Qu.:   0.00  
##  Max.   : 89.80   Max.   : 180.00   Max.   : 180.00   Max.   : 167.00  
##                                     NA's   :19216     NA's   :19216    
##  amplitude_pitch_forearm total_accel_forearm var_accel_forearm
##  Min.   :  0.0           Min.   :  0.00      Min.   :  0.000  
##  1st Qu.:  2.0           1st Qu.: 29.00      1st Qu.:  6.759  
##  Median : 83.7           Median : 36.00      Median : 21.165  
##  Mean   :139.1           Mean   : 34.72      Mean   : 33.502  
##  3rd Qu.:350.0           3rd Qu.: 41.00      3rd Qu.: 51.240  
##  Max.   :360.0           Max.   :108.00      Max.   :172.606  
##  NA's   :19216                               NA's   :19216    
##  gyros_forearm_x   gyros_forearm_y     gyros_forearm_z   
##  Min.   :-22.000   Min.   : -7.02000   Min.   : -8.0900  
##  1st Qu.: -0.220   1st Qu.: -1.46000   1st Qu.: -0.1800  
##  Median :  0.050   Median :  0.03000   Median :  0.0800  
##  Mean   :  0.158   Mean   :  0.07517   Mean   :  0.1512  
##  3rd Qu.:  0.560   3rd Qu.:  1.62000   3rd Qu.:  0.4900  
##  Max.   :  3.970   Max.   :311.00000   Max.   :231.0000  
##                                                          
##  accel_forearm_x   accel_forearm_y  accel_forearm_z   magnet_forearm_x 
##  Min.   :-498.00   Min.   :-632.0   Min.   :-446.00   Min.   :-1280.0  
##  1st Qu.:-178.00   1st Qu.:  57.0   1st Qu.:-182.00   1st Qu.: -616.0  
##  Median : -57.00   Median : 201.0   Median : -39.00   Median : -378.0  
##  Mean   : -61.65   Mean   : 163.7   Mean   : -55.29   Mean   : -312.6  
##  3rd Qu.:  76.00   3rd Qu.: 312.0   3rd Qu.:  26.00   3rd Qu.:  -73.0  
##  Max.   : 477.00   Max.   : 923.0   Max.   : 291.00   Max.   :  672.0  
##                                                                        
##  magnet_forearm_y magnet_forearm_z classe  
##  Min.   :-896.0   Min.   :-973.0   A:5580  
##  1st Qu.:   2.0   1st Qu.: 191.0   B:3797  
##  Median : 591.0   Median : 511.0   C:3422  
##  Mean   : 380.1   Mean   : 393.6   D:3216  
##  3rd Qu.: 737.0   3rd Qu.: 653.0   E:3607  
##  Max.   :1480.0   Max.   :1090.0           
## 
```

Some variables seem to have NA values. Additionally, each variable that has NA values, has exactly 19216 of them, indicating that for these variables measurement are very incomplete and therefore should be removed for both training and test set.


```r
training <- training[,colSums(is.na(training)*1)==0]
testing <- testing[,colSums(is.na(training)*1)==0]
```

The first 6 columns seems to be identifier and timestamp data and presumably don't have a lot of added value in our model and therefore will
be removed as well.


```r
training <- training[,-c(1:6)]
testing <- testing[,-c(1:6)]
```

##5. Data Partitioning

In general accuracy on the training set is optimistic as the model will be somewhat overfitted to specific feautures of the data. A better estimate
comes from an independent set. Therefore we split the training data into training data and test data. The test data will be used to estimate the
out-of-sample error.


```r
split <- createDataPartition(training$classe,p=0.6,list=F)
trainSet <- training[split,]
testSet <- training[-split,]
```

##6. Model fitting

A **boosting** algorithm will be implemented as this is (together with other algorithms like random forest) one of the more accurate models.

### 6.1 Boosting

In order to validate our model a k-fold cross validation procedure will be applied. Although there are no strict rules regarding the size of k, 10-fold is a commonly used. However, this will slow down the procedure too much. Therefore we set the number to 5.


```r
modFitBoosting <- train(classe~.
                        ,method="gbm"
                        ,data=trainSet
                        ,verbose=FALSE
                        ,trControl = trainControl(method="cv",number=5))
```

```
## Loading required package: plyr
```
#### 6.1.2 in-sample error

Test the accuracy of the training set in order to obtain an in-sample error rate.


```r
trainPredBoosting <- predict(modFitBoosting,trainSet)
confusionMatrix(trainPredBoosting,trainSet$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 3318   47    0    1    3
##          B   21 2191   38    5   25
##          C    7   39 1995   50    9
##          D    1    2   17 1866   24
##          E    1    0    4    8 2104
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9744          
##                  95% CI : (0.9713, 0.9771)
##     No Information Rate : 0.2843          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9676          
##  Mcnemar's Test P-Value : 3.684e-11       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9910   0.9614   0.9713   0.9668   0.9718
## Specificity            0.9939   0.9906   0.9892   0.9955   0.9986
## Pos Pred Value         0.9849   0.9610   0.9500   0.9770   0.9939
## Neg Pred Value         0.9964   0.9907   0.9939   0.9935   0.9937
## Prevalence             0.2843   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2818   0.1861   0.1694   0.1585   0.1787
## Detection Prevalence   0.2861   0.1936   0.1783   0.1622   0.1798
## Balanced Accuracy      0.9925   0.9760   0.9802   0.9812   0.9852
```

The accurancy of the model on our training set is **0.9744**, meaning an in-sample error rate of **2.56%**.

#### 6.1.3 out-of-sample error

As the model was fitted it on the training set it is logical the in-sample error rate will be lower than the out-of-sample error rate. This is due to what is called overfitting. Consequently we expect the out-of-sample error rate to be somewhat higher than 2.45%


```r
testPredBoosting <- predict(modFitBoosting,testSet)
confusionMatrix(testPredBoosting,testSet$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2192   46    0    0    6
##          B   28 1419   57    3   26
##          C    8   51 1287   37   14
##          D    3    0   20 1232   16
##          E    1    2    4   14 1380
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9572          
##                  95% CI : (0.9525, 0.9615)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9458          
##  Mcnemar's Test P-Value : 5.726e-08       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9821   0.9348   0.9408   0.9580   0.9570
## Specificity            0.9907   0.9820   0.9830   0.9941   0.9967
## Pos Pred Value         0.9768   0.9256   0.9213   0.9693   0.9850
## Neg Pred Value         0.9929   0.9843   0.9874   0.9918   0.9904
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2794   0.1809   0.1640   0.1570   0.1759
## Detection Prevalence   0.2860   0.1954   0.1781   0.1620   0.1786
## Balanced Accuracy      0.9864   0.9584   0.9619   0.9760   0.9769
```

Indeed, the accuracy of our test set is **0.9572**, meaning and out-of-sample error rate of **4.28%**.

##7. Prediction

Our obtained learning algorithm is now applied to the 20 test cases. The outcome will be 20 predictions of the fashion (see introduction) in which participants performed barbell lifts


```r
answersBoosting <- predict(modFitBoosting,testing)
answersBoosting
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

Text files are created in order to submitt to the Coursera website.


```r
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}
pml_write_files(answersBoosting)
```


