# keras-transfer-learning-inception-resnetv2
## Requirements
You will need 2 files from the official keras repo: https://github.com/fchollet/keras :

keras/applications/inception_resnet_v2.py
keras/applications/imagenet_utils.py
Place(replace) them in your keras/applications folder (usually at "/Library/Python/2.7/site-packages/keras" but it could be different.

You have to replace these two files so you can easy use the inception resnet v2 model.

You will also need the dataset, u can find the one I used for this example at 
https://mega.nz/#!Bo42wLKK!yyR7gXJSFOSwy-W7R-TVbSfBmict5TdUvTuiSvEOZSE

As you can see the dataset follow this structure:
Dataset  
 ├── cats  
 ├── dogs  
 ├── horses  
 └── humans  

You can use the sample dataset from the link or you can also change it but u have to follow this structure.

## Quick start
Run evaluate.py
If it's the first time it will train the last layer of the model with the dataset you choose.
Otherwise it will just load the model previously saved.
