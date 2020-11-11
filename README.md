# DL-project-Makeup-Transfer
## Authors: Jingyan Dai(jingyand) Anyu Chen(anyuc)

We choose baseline in two aspects. First is image-to-image method which contains Deep Image Analogy and CycleGAN. Second is the makeup transfer method called BeautyGAN.

**Deep Image Analogy**   
In DeepImageAnalogy folder, run the code using 
```bash
python DeepImageAnalogy.py
```
Results will be saved in the `Results/` folder.  
Our examples:  
<img src="DIA1.png" width="800" height="200" align="bottom" />
<img src="DIA2.png" width="800" height="200" align="bottom" />

**CycleGAN**   
In CycleGAN folder, run the code `python train.py --dataroot ./datasets/summer2winter_yosemite --name summer2winter_yosemite_cyclegan --model cycle_gan` to train.  
run the code `python test.py --dataroot ./datasets/summer2winter_yosemite --name summer2winter_yosemite_cyclegan --model cycle_gan` to test.  
Also there are pretrain models can be retrieved code `bash ./scripts/download_cyclegan_model.sh summer2winter_yosemite`.  
To generate result, run the code `python test.py --dataroot datasets/Try --name summer2winter_pretrained --model test --no_dropout`.  
Results will be saved in the `results/` folder. We also use style_monet, style_cezanne, style_ukiyoe, style_vangogh.  
Our examples:  
<img src="CycleGAN.png" width="600" height="400" align="bottom" />

**BeautyGAN**  
In BeautyGAN folder, run the code  

Our examples:  
![](BGAN1.jpg)
![](BGAN2.jpg)
