### Supplement To Applications of Deep Learning to Decorated Ceramic Typology and Classification: A Case Study Using Tusayan White Ware from Northeast Arizona by Leszek Pawlowicz and Christian Downum

This repository contains the following files:

Supplemental.pdf - A digital supplement to the paper, contain a longer exposition of both the creation of the training dataset and the training process.

ResNet50_processor_subfolders.py, VGG_16_processor_subfolders.py - Programs used to train CNN models to classify Tusayan White Ware sherds using both ResNet50 and VGG16 models pre-trained on the ImageNet dataset.

Ensemble_Labeled_Classification_Tester_Output.py - Program used to evaluate sherds from the test dataset using ensemble classification.

Data_structure.txt - Brief description of proper dataset folder structure necessary to train and evaluate CNN models.

requirements.txt - List of Python packages/versions required to run the above programs.

Python_environment_instructions.txt - Brief instructions on how to create Python environment, and install packages using the requirements.txt, necessary for running the Python programs as well as the Jupyter notebook (see below).

TWW_data.xlsx - Spreadsheet containing human sherd classifications, consensus classification, CNN model top type prediction, and per-sherd CNN model confidences for every  type

The original image data is too large to upload to GitHub. It has instead been uploaded to a separate location, and can be downloaded using this link:

https://1drv.ms/u/s!ApEEvWWEJITPhrYFVMaTNTLeigCFNA?e=EoNGIG


It contains both the original image data, and a Jupyter notebook that allows the user to "test-drive" the use of CNN models in classifying sherds, creating saliency heat maps, and finding the closest matching sherds in appearance. Instructions for use are in the Jupyter notebook comments, and in the file Python_environment_instructions.txt.
