#!/bin/sh
# the next line restarts using wish \
\
# for UNIX: \
exec wish "$0" "$@" -colormap new

option add *background gray26

# global variables can also be set after the proc's
set AUTOREFRESH 0
set ZOOM 6
set TMPPATH result

set activations_side top    ;# set: top  or  left

# set wm title
set splittedname [split $TMPPATH /]          ;# split elems where "/"
set shortname [lindex $splittedname end]     ;# last element
wm title . $shortname

set AREA_LIST "0 1 2 3 4 5"
set DISPLAYACTS 1
set DISPLAYWEIGHTS 1

#if  {$argc == 0} {
#    puts "display activity files like $TMPPATH/obs_S_0.pgm, $TMPPATH/obs_R_1.pgm using:"
#    puts "./look.tcl a 0 1"
#    puts "display weight files like $TMPPATH/obs_w_1_0.pgm using:"
#    puts "./look.tcl w 1"
#    puts "display all such files using:"
#    puts "./look.tcl a w 0 1 2 3 4"
#    exit
#}

#foreach u $argv {
#    if  {[string is integer $u]} {
#        lappend AREA_LIST $u
#    }
#    if  {[string compare $u a] == 0} {
#        set DISPLAYACTS 1
#        puts "Displaying acts"
#    }
#    if  {[string compare $u w] == 0} {
#        set DISPLAYWEIGHTS 1
#        puts "Displaying weights"
#    }
#}
#if  {[expr $DISPLAYACTS + $DISPLAYWEIGHTS] == 0} {
#    set DISPLAYACTS 1
#    set DISPLAYWEIGHTS 1
#    puts "Displaying acts and areas"
#}
puts "AREA_LIST: $AREA_LIST"

proc show_pic {fram pic name} {
    global firsttime
    global AUTOREFRESH
    global ZOOM
    global activations_side

    set catcherror [catch {set pic [image create photo $pic -palette 256 -file $pic]}]
    if  {$catcherror == 0} {
        image create photo copy$pic
        copy$pic copy $pic -zoom $ZOOM
    }

    if  {[string compare fifo [file type $pic]] == 0} {
        set AUTOREFRESH 1
        set highS 0
        set lowS  0
    } else {
        set fp [open $pic r]
        seek $fp 0
        gets $fp line
        gets $fp line
        set highS [lindex [split [lindex [split $line :] 1] " "] 1]
        set lowS [lindex [split $line :] 2]
        close $fp
    }

    # cut "obs_" away
    set splittedname [split $name _]             ;# split elems where "_"
    set shortname [lreplace $splittedname 0 0]   ;# repl elem 0,0 with nothing
    set joinname [join $shortname _]             ;# join elems with "_"


  if  {$catcherror == 0} {
    if  {$firsttime} {
        label $fram.pics$name -image copy$pic
        label $fram.text$name -fg white -text "$joinname: [format %6.2f $lowS] .. [format %6.2f $highS]"
        if  {[string compare [string index $name 0] W]} {
            pack $fram.pics$name -side $activations_side
	} else {
            pack $fram.pics$name
	}
        pack $fram.text$name
    } else {
        $fram.pics$name configure -image copy$pic
        $fram.text$name configure -fg white -text "$joinname: [format %6.2f $lowS] .. [format %6.2f $highS]"
    }

    if  {$lowS == 0} {
        if  {$highS == 0} {
            $fram.text$name configure -fg grey
        }
    }
  }
}


proc make_pics {} {
    global AREA_LIST
    global picnames
    global TMPPATH
    global DISPLAYACTS
    global DISPLAYWEIGHTS

    foreach ar $AREA_LIST {

        set filelistact($ar) \
           [list obs_A_$ar obs_B_$ar obs_C_$ar obs_D_$ar obs_E_$ar obs_F_$ar obs_G_$ar obs_H_$ar obs_I_$ar obs_J_$ar \
                 obs_K_$ar obs_L_$ar obs_M_$ar obs_N_$ar obs_O_$ar obs_P_$ar obs_Q_$ar obs_R_$ar obs_S_$ar obs_T_$ar \
                 obs_U_$ar obs_V_$ar obs_W_$ar obs_X_$ar obs_Y_$ar obs_Z_$ar]

        set filelistweight($ar) \
           [list obs_W_${ar}_0  obs_W_${ar}_1  obs_W_${ar}_2  obs_W_${ar}_3  obs_W_${ar}_4  obs_W_${ar}_5  obs_W_${ar}_6  obs_W_${ar}_7  obs_W_${ar}_8  obs_W_${ar}_9\
                 obs_W_${ar}_10 obs_W_${ar}_11 obs_W_${ar}_12 obs_W_${ar}_13 obs_W_${ar}_14 obs_W_${ar}_15 obs_W_${ar}_16 obs_W_${ar}_17 obs_W_${ar}_18 obs_W_${ar}_19\
                 obs_V_${ar}_0  obs_V_${ar}_1  obs_V_${ar}_2  obs_V_${ar}_3  obs_V_${ar}_4  obs_V_${ar}_5  obs_V_${ar}_6  obs_V_${ar}_7  obs_V_${ar}_8  obs_V_${ar}_9\
                 obs_V_${ar}_10 obs_V_${ar}_11 obs_V_${ar}_12 obs_V_${ar}_13 obs_V_${ar}_14 obs_V_${ar}_15 obs_V_${ar}_16 obs_V_${ar}_17 obs_V_${ar}_18 obs_V_${ar}_19]

        if {$DISPLAYACTS} {

            foreach u $filelistact($ar) {

                if  {[file exists $TMPPATH/$u.pgm]} {

                    show_pic .fr$ar $TMPPATH/${u}.pgm $u

                    set picnames($TMPPATH/${u}.pgm) \
                        [list .fr$ar [file mtime $TMPPATH/${u}.pgm]]
                }
            }
        }


        if {$DISPLAYWEIGHTS} {

            foreach u $filelistweight($ar) {

                if  {[file exists $TMPPATH/$u.pgm]} {

                    show_pic .fr$ar $TMPPATH/${u}.pgm $u

                    set picnames($TMPPATH/${u}.pgm) \
                        [list .fr$ar [file mtime $TMPPATH/${u}.pgm]]
                }
            }
        }
    }
}


proc doforever {} {
    global picnames
    global TMPPATH

    set update 0
    foreach u [array names picnames] {
        if {[lindex $picnames($u) 1] != [file mtime $u]} {
             set update 1
        }
    }
    if  {$update} {
        make_pics
    }

    after 10 doforever  ;# number is in msec
}



# Top level frames

foreach ar $AREA_LIST {
    frame .fr$ar
    pack .fr$ar -side left -fill both
}

frame .right
pack .right -side left -fill both


# Bitmap images

set firsttime 1
make_pics
set firsttime 0

bind . <Button-1> {make_pics}
bind . <Return>   {make_pics}
bind . <Button-3> {destroy .}
bind . <q>        {destroy .}
set ct 1


if  {$AUTOREFRESH} {
    doforever
}
