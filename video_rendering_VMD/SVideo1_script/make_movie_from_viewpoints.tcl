# Movie generation functions for viewchangerender viewpoints
# Based on video_code.tcl rendering approach
#
# Usage:
#   1. Source DIY_movie_list.tcl to load viewpoints
#   2. Source video_code.tcl to set up rendering settings
#   3. Source this file to load the functions
#   4. Call make_movie_from_viewpoints_simple or make_movie_from_viewpoints

# Procedure to render movie from viewchangerender viewpoints
# This version attempts to apply the stored representations for each viewpoint
proc make_movie_from_viewpoints {{format "png"} {fps 24}} {
    # Ensure viewpoints are loaded
    if {![info exists ::VCR::movieList]} {
        puts "Error: movieList not found. Please run viewchangerender_restore_my_state first."
        puts "  (This is done automatically when you source DIY_movie_list.tcl)"
        return
    }
    
    # Get the movie list
    set vpList $::VCR::movieList
    set numViewpoints [llength $vpList]
    
    puts "Rendering $numViewpoints viewpoints..."
    
    # Get the top molecule
    set topmol [molinfo top]
    set molname [molinfo $topmol get name]
    
    # Frame counter for output files
    set frameNum 0
    
    # Loop through each viewpoint
    foreach vpID $vpList {
        puts "Rendering viewpoint $vpID (frame $frameNum)..."
        
        # Apply viewpoint matrices individually
        if {[info exists ::VCR::viewpoints($vpID,0)]} {
            molinfo $topmol set center_matrix $::VCR::viewpoints($vpID,1)
            molinfo $topmol set rotate_matrix $::VCR::viewpoints($vpID,0)
            molinfo $topmol set scale_matrix $::VCR::viewpoints($vpID,2)
            molinfo $topmol set global_matrix $::VCR::viewpoints($vpID,3)
        }
        
        # Apply field of view if available
        if {[info exists ::VCR::viewpoints($vpID,4)]} {
            display fieldofview $::VCR::viewpoints($vpID,4)
        }
        
        # Apply representations if they exist
        if {[info exists ::VCR::representations($vpID,$molname)]} {
            set repList $::VCR::representations($vpID,$molname)
            # Clear existing representations
            set numReps [molinfo $topmol get numreps]
            for {set i [expr $numReps - 1]} {$i >= 0} {incr i -1} {
                mol delrep $i $topmol
            }
            # Apply new representations
            foreach repStr $repList {
                # Parse representation string (format: Type_param1_param2-selection-ColorID_X-Material)
                # This is a simplified parser - you may need to adjust based on your exact format
                set parts [split $repStr "-"]
                if {[llength $parts] >= 2} {
                    set repType [lindex $parts 0]
                    set selection [lindex $parts 1]
                    set colorID 1
                    set material "AOChalky"
                    
                    # Extract ColorID
                    foreach part $parts {
                        if {[string match "ColorID_*" $part]} {
                            set colorID [lindex [split $part "_"] 1]
                        }
                        if {[string match "Material*" $part] || [string match "Transparent" $part] || [string match "AOChalky" $part]} {
                            set material $part
                        }
                    }
                    
                    # Parse representation type and parameters
                    set repParts [split $repType "_"]
                    set repStyle [lindex $repParts 0]
                    set repParams [lrange $repParts 1 end]
                    
                    # Create representation
                    mol representation $repStyle {*}$repParams
                    mol color ColorID $colorID
                    mol selection $selection
                    mol material $material
                    mol addrep $topmol
                }
            }
        }
        
        # Update display
        display update
        
        # Render frame
        if {$format == "png"} {
            # Enable alpha channel for PNG
            set env(VMDOPTIXWRITEALPHA) 1
            set filename [format "viewpoint_%05d.png" $frameNum]
        } else {
            set filename [format "viewpoint_%05d.tga" $frameNum]
        }
        
        render TachyonLOptiXInternal $filename
        puts "  Saved: $filename"
        
        incr frameNum
    }
    
    puts "Done! Rendered $frameNum frames."
    puts "To create video, run:"
    puts "  ffmpeg -framerate $fps -pattern_type glob -i 'viewpoint_*.png' -vf \"pad=ceil(iw/2)*2:ceil(ih/2)*2\" -c:v libx264 -pix_fmt yuv420p -crf 18 movie_from_viewpoints.mp4"
}

# Alternative simpler version that just renders one frame per viewpoint
# without trying to parse complex representation strings
# This keeps your current representations and only changes the viewpoint
proc make_movie_from_viewpoints_simple {{format "png"} {fps 24}} {
    # Ensure viewpoints are loaded
    if {![info exists ::VCR::movieList]} {
        puts "Error: movieList not found. Please run viewchangerender_restore_my_state first."
        puts "  (This is done automatically when you source DIY_movie_list.tcl)"
        return
    }
    
    # Get the movie list
    set vpList $::VCR::movieList
    set numViewpoints [llength $vpList]
    
    puts "Rendering $numViewpoints viewpoints (simple mode - keeping current representations)..."
    
    # Get the top molecule
    set topmol [molinfo top]
    
    # Frame counter for output files
    set frameNum 0
    
    # Loop through each viewpoint
    foreach vpID $vpList {
        puts "Rendering viewpoint $vpID (frame $frameNum)..."
        
        # Apply viewpoint matrices individually
        if {[info exists ::VCR::viewpoints($vpID,0)]} {
            molinfo $topmol set center_matrix $::VCR::viewpoints($vpID,1)
            molinfo $topmol set rotate_matrix $::VCR::viewpoints($vpID,0)
            molinfo $topmol set scale_matrix $::VCR::viewpoints($vpID,2)
            molinfo $topmol set global_matrix $::VCR::viewpoints($vpID,3)
        }
        
        # Apply field of view if available
        if {[info exists ::VCR::viewpoints($vpID,4)]} {
            display fieldofview $::VCR::viewpoints($vpID,4)
        }
        
        # Update display
        display update
        
        # Render frame
        if {$format == "png"} {
            # Enable alpha channel for PNG
            set env(VMDOPTIXWRITEALPHA) 1
            set filename [format "viewpoint_%05d.png" $frameNum]
        } else {
            set filename [format "viewpoint_%05d.tga" $frameNum]
        }
        
        render TachyonLOptiXInternal $filename
        puts "  Saved: $filename"
        
        incr frameNum
    }
    
    puts "Done! Rendered $frameNum frames."
    puts "To create video, run:"
    puts "  ffmpeg -framerate $fps -pattern_type glob -i 'viewpoint_*.png' -vf \"pad=ceil(iw/2)*2:ceil(ih/2)*2\" -c:v libx264 -pix_fmt yuv420p -crf 18 movie_from_viewpoints.mp4"
}

