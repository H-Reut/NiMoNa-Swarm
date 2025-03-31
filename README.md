# NiMoNa-Swarm
This repo contains:
- the code written (saved in irregular intervals) <!--(total size: ~0.34 MiB)-->
- media produced by the code (animations and graphs) <!--(total size: ~1.84 GiB)-->
  
### Note on media
This repo is a complete dump of everything produced. Some animations may be glitchy, show empty domains $\Omega$, or be at a zoom level where nothing interesting can be seen. The lab report .pdf contains links to specific pieces of media to demonstrate certain points, but besides those, most animations are not very interesting. 
The media in the folder "100 2nd order rk4 morse extension predator prey" was not created by me.

### Note on code
Similarly, the code may crash or simply be unfinished. I didn't save every iteration of the code, since many times after creating one animation only some parameters needed to be changed to produce the next animation. If you want to run the code yourself, note the following:
- if there are multiple files in a folder, obviously the `main.py` file is to be launched which then imports the other files.
- I imported the following libraries:
  - NumPy
  - SciPy
  - Matplotlib
  - jsonpickle (this is rarely used, in most cases you should be fine by just removing the import)
- the created animations will be placed either in the directory from where the program is run, or in a directory specified by the variable `img_save_path` at the beginning (after the imports) of the `main.py` file. By default this directory is `H:/dump/`. The directory needs to exist before launching the code, otherwise the program will likely crash. The program can create temporary files which are usually deleted automatically, but be sure to check afterwards.
- If visualization is done with `visualization_ffmpeg.py`, [FFmpeg](https://ffmpeg.org/download.html) needs to be installed and added to the PATH environment variable. FFmpeg converts the frames (.png) to videos (.mov or .mp4).
- I used Python 3.11, 3.12, and 3.13 but any Python 3 version should work.
The code is 