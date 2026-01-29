#set isoval 0.6
display depthcue off
display projection Orthographic
axes location Off
display shadows on                                                        #打开阴影                           
display ambientocclusion on                                            #打开ao
render aasamples TachyonLOptiXInternal 24                     #设置抗锯齿强度，越大越平滑，但是渲染就越慢
render aosamples TachyonLOptiXInternal 24

proc make_movie {} {
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
proc make_movie {} {
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

#ffmpeg -framerate 24 -pattern_type glob -i '*.tga'   -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2"   -c:v libx264 -pix_fmt yuv420p -crf 18 demo_movie_ER.mp4
