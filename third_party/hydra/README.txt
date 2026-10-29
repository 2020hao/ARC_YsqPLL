This shared code and data reproduces the results obtained in our paper:
"Real-time series arc fault detection and
appliances classification in AC networks
based on competing convolutional kernels"
by
FRANCESCO FERRACUTI, RICCARDO FELICETTI, LUCA CAVANINI, PATRICK
SCHWEITZER, AND ANDREA MONTERIÙ

################
"visualize.m"
################
Plots data from pre-saved training and testing results.
Returns the plots shown in Section IV of the paper.
To perform training from scratch, use "main.m"


################
"main.m"
################
To extract features and train classifiers from scratch, delete the folder
"Results" and run "main.m"
Features extraction requires a Supported Python Implementation.
Please refer to:
https://it.mathworks.com/help/matlab/matlab_external/install-supported-python-implementation.html
to install and configure Python.
Full exploration can take more than one day, depending on the hardware.
Consider parallelizing tasks (see code lines 175-179 in main.m) and/or
reducing the size of the hyperparameter region to be explored.