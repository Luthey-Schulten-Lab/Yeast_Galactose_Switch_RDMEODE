#set isoval 0.6
display depthcue off
display projection Orthographic
axes location Off
display shadows on                                                                            
display ambientocclusion on   
set env(VMDOPTIXWRITEALPHA) 1                                        
render aasamples TachyonLOptiXInternal 24                     
render aosamples TachyonLOptiXInternal 24

proc make_movie_tga {} {
    set num [molinfo top get numframes]
    # loop through the frames
    for {set i 1} {$i < $num} {incr i} {
        # go to the given frame
        set filename [format "%05d" $i].tga
        animate goto $i
        display update
        render TachyonLOptiXInternal $filename
    }
}
#for transparent
proc make_movie_png {} {
    set num [molinfo top get numframes]
    # Enable alpha channel output
    set env(VMDOPTIXWRITEALPHA) 1
    # Loop through the frames
    for {set i 0} {$i < $num} {incr i} {
        # Go to the given frame
        set filename [format "%05d.png" $i]
        animate goto $i
        display update
        # Render using TachyonLOptiXInternal with PNG format
        render TachyonLOptiXInternal $filename
    }
}
#ffmpeg -framerate 24 -pattern_type glob -i '*.png'   -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2"   -c:v libx264 -pix_fmt yuv420p -crf 18 S2_baseline.mp4
#ffmpeg -framerate 24 -pattern_type glob -i '*.png'   -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2"   -c:v libx264 -pix_fmt yuv420p -crf 18 S3_chromosome.mp4
#ffmpeg -framerate 24 -pattern_type glob -i '*.png'   -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2"   -c:v libx264 -pix_fmt yuv420p -crf 18 S4_ER.mp4
#ffmpeg -framerate 24 -pattern_type glob -i '*.png'   -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2"   -c:v libx264 -pix_fmt yuv420p -crf 18 S5_eff.mp4
