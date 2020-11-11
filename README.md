# DL-project-Makeup-Transfer
## Baseline Models

We choose baseline in two aspects. First is image-to-image method which contains Deep Image Analogy and CycleGAN. Second is the makeup transfer method called BeautyGAN.

### Deep Image Analogy
In DeepImageAnalogy folder, run the code:
```bash
python DeepImageAnalogy.py
```
Results will be saved in the `Results/` folder.  
Our examples:  
<img src="DIA1.png" width="800" height="200" align="bottom" />
<img src="DIA2.png" width="800" height="200" align="bottom" />

### CycleGAN   
In CycleGAN folder, train the model with:
```bash
python train.py --dataroot ./datasets/summer2winter_yosemite --name summer2winter_yosemite_cyclegan --model cycle_gan
``` 
Test the model with:
```bash
python test.py --dataroot ./datasets/summer2winter_yosemite --name summer2winter_yosemite_cyclegan --model cycle_gan
```   
Also there are pretrain models can be retrieved using code:
```bash
bash ./scripts/download_cyclegan_model.sh summer2winter_yosemite
``` 
To generate result, run the code:
```bash
python test.py --dataroot datasets/Try --name summer2winter_pretrained --model test --no_dropout
```
Results will be saved in the `results/` folder. We also use style_monet, style_cezanne, style_ukiyoe, style_vangogh.  
Our examples:  
<img src="CycleGAN.png" width="600" height="400" align="bottom" />

### BeautyGAN
There is a BeautyGAN pretrain model, model is saved in https://drive.google.com/file/d/1Y-XDe8DLGFiChCGkJ1s8jxQ8aQrE5SpK/view?usp=sharing  
Unzip it and put it in BeautyGAN folder. Run the code with 
```bash
python main.py --nomake_up 1.jpg
```
Our examples:  
![](BGAN1.jpg)
![](BGAN2.jpg)
