# Near-Perfect Image Super-Resolution: Exploring Multiple Divergence Measures in Fully Convolutional GAN</b>

<h3> PyTorch Implementation for <b>SuRGe: Super Resolution Generator</b> </h3>

```
pip install -r requirements.txt
```
Download the pretrained model from [Google Drive](https://drive.google.com/file/d/1cd0RUOTbD30dqLbngZ0h7GsvqiZK4RZQ/view?usp=drive_link) and Put it in the <b>pretrained_models</b> folder.

Run this code to check the images in Demo Folder

```
python3 test.py
```

Input <b> Demo </b> Folder 
Output <b> Test_Results </b> Folder

<b> Generator </b>
![Generator](https://github.com/Thecoder1012/SuRGe/blob/main/assets/generator_updated.png)
<b> Discriminator </b>
![Discriminator](https://github.com/Thecoder1012/SuRGe/blob/main/assets/discriminator_main.jpg)

<b> Super Resolution Results </b>
![Results](https://github.com/Thecoder1012/SuRGe/blob/main/assets/image_SR.png)

<b> Visual Comparison </b>
![Results](https://github.com/Thecoder1012/SuRGe/blob/main/assets/Qualitative.jpg)

Go to the folder <b>Trainer</b> to train SuRGe. 

Please provide the dataset path in <i>train.py</i>

```
python3 train.py
```
