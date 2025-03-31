# NiMoNa-Swarm
This repo contains:
- the code written (saved in irregular intervals) <!--(total size: ~0.34 MiB)-->
- media produced by the code (animations and graphs) <!--(total size: ~1.84 GiB)-->
  
### Note on media
This repo is an (almost) complete dump of everything produced. Some animations may be glitchy, show empty domains $\Omega$, or be at a zoom level where nothing interesting can be seen. The lab report .pdf contains links to specific pieces of media to demonstrate certain points, but besides those, most animations are not very interesting. 
The media in the folder "100 2nd order rk4 morse extension predator prey" was not created by me.

### Note on code
If you want to run the code yourself, note the following:
- if there are multiple files in a folder, obviously the `main.py` file is to be launched which then imports the other files.
- I imported the following libraries:
  - NumPy
  - SciPy
  - Matplotlib
  - jsonpickle (this is rarely used, in most cases you should be fine by just removing the import)
- the created animations will be placed either in the directory from where the program is run, or in a directory specified by the variable `img_save_path` which is declared at the beginning (after the imports) of the `main.py` file. 
  By default this directory is `H:/dump/`. The directory needs to exist before launching the code, otherwise the program will likely crash. 
  The program can create temporary files in that folder which are usually deleted automatically, but be sure to check and delete afterwards.
- If visualization is done with `visualization_ffmpeg.py`, [FFmpeg](https://ffmpeg.org/download.html) needs to be installed and added to the PATH environment variable. FFmpeg converts the frames (.png) to videos (.mov or .mp4).
- I used Python 3.11, 3.12, and 3.13 but any Python 3 version should work.
The code evolved over time. I didn't save every iteration of the code, since many times after creating one animation only some parameters needed to be changed to produce the next animation.
Depending on what I studied, some parts were added or removed (usually commented out). 
For example when improving the efficiency (runtime) of a function, many implementations of the same function would exist. But in the next version most of those would be removed again. So there is no one definitive version.

Here is a quick summary of the versions:  
0.0 - was my first implementation, without group input.  
0.1 -  
0.2 -  
1.0 - 1st order ODE model we derived together  
1.1 - 1st order ODE model  
1.2 - 1st order ODE model  
1.3 - 2nd order ODE model. Implementation of an alternative solver  
3.0 - 2nd order ODE model. Implementation of an alternative solver; comparison with our main solver for the ODE  
3.1 - many implementations of morse functions, goal was to find the fastest.  
4.0 -  
4.1 -  
5.0 -  
5.1 - runtime comparison of different implementation for morse function. runtimes are already calculated in 'runtimes.py'  
6.0 -  
6.1 - final iteration to create animations
