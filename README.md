# Denoise Tools

### Usage

You can implement image denoising using the following commands
**NOTICE** The images fed to the model should have the same width & height, i.e., should be a square image.

##### Denoise Single Image

$ bash denoise.sh 'Input_image' 'Output_image'

##### Denoise a Folder

$ bash denoise_folder.sh 'Input Directory' 'Output Directory'

### Examples

##### Salt & Pepper noise

Before

<img src="/Users/LungHuiWu/Dropbox/Mac/Desktop/DIP/Final/blurred_sharp/noise/78_sp.png" alt="78_poisson" style="zoom:100%;" />

After

<img src="/Users/LungHuiWu/Dropbox/Mac/Desktop/DIP/Final/blurred_sharp/denoise/78_sp_denoise.png" alt="78_poisson" style="zoom:100%;" />

##### Poisson noise

Before

![99_poisson](file:///Users/LungHuiWu/Dropbox/Mac/Desktop/DIP/Final/blurred_sharp/noise/99_poisson.png?lastModify=1641396166)

After

![99_poisson](file:///Users/LungHuiWu/Dropbox/Mac/Desktop/DIP/Final/blurred_sharp/denoise/99_poisson_denoise.png?lastModify=1641396166)

##### Gaussian noise

Before

![519_gaussian](file:///Users/LungHuiWu/Dropbox/Mac/Desktop/DIP/Final/blurred_sharp/noise/519_gaussian.png?lastModify=1641396166)

After

![519_gaussian](file:///Users/LungHuiWu/Dropbox/Mac/Desktop/DIP/Final/blurred_sharp/denoise/519_gaussian_denoise.png?lastModify=1641396166)



