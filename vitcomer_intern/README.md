MMSeg Extension with ViT-CoMer and InternImage Models
This repository extends the functionality of mmseg-extension by incorporating ViT-CoMer and InternImage models.
Setup and Usage
Clone the repository:
bash
git clone https://github.com/chenller/mmseg-extension.git
cd mmseg-extension

Copy the provided train.py and test.py files into the tools folder.
Add the supplied configuration files to the config directory.
Modify the ade.py file to match the labels of your current project.
Run the training or testing scripts as needed.
Model Support
This extension adds support for two additional models:
ViT-CoMer
InternImage
Running Experiments
To train or test a model, use the following commands:
bash
# For training
python tools/train.py

# For testing
python tools/test.py

Note
Remember to adjust the ade.py file to match your project's specific label structure before running any experiments.
