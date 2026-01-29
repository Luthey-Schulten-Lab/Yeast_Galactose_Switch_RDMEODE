# VMD Video Rendering

Figure 3 (geometry overview) and Supplementary Videos 1–4 were rendered using VMD 1.9.4 with the Lattice Microbes plugin. Alternatively, VMD 2.0 includes the LM plugin by default and can execute the same scripts.

## Prerequisites

Download the required trajectory files (`.lm`) from Zenodo before reproducing any figures or videos.

## Reproducing Figure 3 (Geometry Overview)

1. Load the first frame of any ER trajectory
2. Load the visualization state file
3. Render and save the image

## Reproducing Video 1 (Camera Flythrough)

1. Load the camera motion and representation change script
2. The script defines predefined camera positions and angles
3. Start video recording

## Reproducing Videos 2–4 (Trajectory Animations)

1. Load the corresponding trajectory file
2. Load the visualization state file
3. For standard resolution: use the built-in `movie_maker` plugin
4. For high resolution: load the custom rendering script and call the function from TkConsole—this renders frame-by-frame and uses FFmpeg to assemble the final video
