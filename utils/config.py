from configparser import ConfigParser

config = ConfigParser()

config["paths"] = {
    "ffmpeg_path" : "/home/rxelmer/Documents/turso/appl_local/ffmpeg/bin/ffmpeg",
    "latex_path" : "/home/rxelmer/Documents/turso/appl_local/tex-basic/texlive/2023/bin/x86_64-linux:",
    "bulkpath" : "/home/rxelmer/Documents/turso/bulks/sim26/"
}

# Define what animations are to be produced:
# Each animation has to be in the from of a list, e.g: ["<animation_type>", "<variable>", "<component>", "<animation_spesific>"]

# <animation_type>: 2D, fourier, sf, kurtosis or triple:
#   - 2D: 2 dimensional heat map of specified variable.
#   - fourier: fourier transform of given variable.
#   - sf: structure function of given variable.
#   - kurtosis: kurtosis of given variable.

# <variable>: "B", "v", "J", "rho"

# <component>: "x", "y", "z", or "total" for vector variable and "pass" for scalar variable.

# <animation_spesific>:
#   - 2D and tirple: "unit" or "unitless"

#   - fourier:
#       - ["x", <0-1>] -> x (or y) states the direction along which you want the slice to be done.
#         <0-1> states the y (or x) coordinate where you want the slice to be done (0-1 -> y_min - y_max).
# 
#       - ["diag", <1 or 2>] -> if you want fourier in diagonal direction. Enter 1 if you want SW->NE.
#                   Enter 2 if you want NW->SW.
#
#       - ["trace", <0-1>, <0-1>] -> If you want trace PSD.
#          First number for x slice y-coord and second number for y slice x-coord.
#
#       - ["trace_diag"] -> for trace PSD for diag directions.
#
#   - sf: a list like [2,4,6...] which states the dl in cells for structure function.
#
#   - kurtosis: again a list like [2,4,6...].

config["settings"] = {
    "start_frame" : 0,
    "end_frame" : 100,

    "output_dir" : "TurbulenceBoxPlots/sim26_anim/sim26",

    "animations" : [
                ("fourier", "B", "perp", ["2D"])
            ],

    "filetype" : ".mp4"
}

with open(".TurbulenceBoxVisualizer.ini", "w") as file:
    config.write(file)

""" ("sf", "B", "x", [2,4,8,16,32,64,128,256,512]),("sf", "B", "y", [2,4,8,16,32,64,128,256,512]),
                ("kurtosis", "B", "x", [2,4,8,16,32,64,128,256,512]),("kurtosis", "B", "y", [2,4,8,16,32,64,128,256,512]) ("rms", "B", "magnitude", "unit"),("rms", "v", "magnitude", "unit"),"""