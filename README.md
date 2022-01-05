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

<img src="/example/78_sp.png" alt="78_sp" style="zoom:100%;" />

After

<img src="/example/78_sp_denoise.png" alt="78_sp" style="zoom:100%;" />

##### Poisson noise

Before

<img src="/example/99_poisson.png" alt="99_poisson" style="zoom:100%;" />

After

<img src="/example/99_poisson_denoise.png" alt="99_poisson" style="zoom:100%;" />

##### Gaussian noise

Before

<img src="/example/519_gaussian.png" alt="519_gaussian" style="zoom:100%;" />

After

<img src="/example/519_gaussian_denoise.png" alt="519_gaussian" style="zoom:100%;" />



