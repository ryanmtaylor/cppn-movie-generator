%run -i sampler.py
sampler = Sampler(z_dim = 8, scale = 10.0, net_size = 32)
z1 = sampler.generate_z()
z2 = sampler.generate_z()
sampler.save_anim_gif(z1, z2, 'output.gif', 120, 0.5, 1.0, 0.1, 1960, 2048)

%run -i sampler.py
sampler = Sampler(z_dim = 8, scale = 10.0, net_size = 32)
sampler.save_anim_mp4('giraffe3', 360)

%run -i sampler.py
sampler = Sampler(z_dim = 4, scale = 10.0, net_size = 16)
sampler.save_anim_mp4('giraffe4', 360)


pip install the following:

ipython
matplotlib
numpy
tensorflow
image
Pillow
jupyter
images2gif

####

docker pull saren/butterflow
