from configparser import ConfigParser

config = ConfigParser()

config["paths"] = {
    "ffmpeg_path" : "/home/elmer/turso/appl/ffmpeg/bin/ffmpeg",
    "latex_path" : "/home/elmer/turso/appl/tex-basic/texlive/2023/bin/x86_64-linux:",
    "bulkpath" : "/home/elmer/turso/bulks_mount_home/test/"
}

# Define what animations are to be produced:
# Each animation has to be in the from of a list, e.g: ["<animation_type>", "<variable>", "<component>", "<animation_spesific>"]
#
# <animation_type>:
#   - 2D: 2 dimensional heat map of specified variable.
#   - triple: 2 dimensional heat map of all components of a vector variable.
#   - fourier: fourier transform of given variable.
#   - sf: structure function of given variable.
#   - kurtosis: kurtosis of given variable.
#   - diagnostics: figure of allotment of different variables.
#   - reconnection: 2D heat map of B_perp or J_z with A_z isolines on top and x and o points.
#
# <variable>: "B", "v", "J", "rho".
#
# <component>: "x", "y", "z", or "magnitude" for vector variable and "pass" for scalar variable.
#
# <animation_spesific>:
#   - 2D: "unit" or "unitless"
#
#   - triple: "unit" or "unitless"
#
#   - fourier:
#       - ["1D"] for 1 dimensional PSD
#       - ["2D"] for 2 dimensional PSD
#       - ["window"] for a snapshot of the 1D psd to check slope
#
#   - sf: a list like [2,4,6...] which states the dl in cells for structure function.
#
#   - kurtosis: again a list like [2,4,6...].

config["settings"] = {
    "start_frame" : 0,
    "end_frame" :22,

    "output_dir" : "Animations/velocity_test8/",

    "animations" : [
	                #("sf", "B", "y", [32,512]),
                    #("kurtosis", "B", "x", [32,64,128,256,512,1024]),
                    #("franci", "", "", [""]),
                    #("sigma", "", "", ["fourier"])
                    #("fourier", "B", "perp", ["1D"]),
                    ("fourier", "B", "perp", ["window"]),
                    #("2D", "J", "z", "unit"),
                    ("triple", "B", "pass", "unit"),
                    #("triple", "B", "pass", "unitless"),
                    #("reconnection", "J", "z", "unit"),
                    #("2D", "J", "z", ["unit"]),
            ],

    "filetype" : ".mp4"
}

with open(".TurbulenceBoxVisualizer.ini", "w") as file:
    config.write(file)

""" ("sf", "B", "x", [2,4,8,16,32,64,128,256,512]),("sf", "B", "y", [2,4,8,16,32,64,128,256,512]),
                ("kurtosis", "B", "x", [2,4,8,16,32,64,128,256,512]),("kurtosis", "B", "y", [2,4,8,16,32,64,128,256,512]) ("rms", "B", "magnitude", "unit"),("rms", "v", "magnitude", "unit"),"""
