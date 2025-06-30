import sys

# Dictionary for translating instructions for VlsvReader
translate = {"B":("vg_b_vol", 1e-9, "nT"),
                "v":("proton/vg_v", 1e3, "km/s"),
                "J":("vg_j", 1e-9, "\frac{nA}{m^2}"),
                "rho":("proton/vg_rho", 1e6, r"$\frac{n}{m^3}$")}

# Defining AnimationSpecs object and checking instructions
class AnimationSpecs():
    def __init__(
        self, animation_type, variable, component, 
        animation_specific, name, bulkpath
        ):
        if animation_type not in ["2D", "triple", "fourier", "sd", "kurtosis"]:
            print("animation_type defined incorrectly")
            print(name)
            sys.exit(1)

        if variable not in ["B", "v", "J", "rho"]:
            print("variable defined incorrectly")
            print(name)
            sys.exit(1)

        if name[-3:] not in ["mp4"]:
            print("filetype defined incorrectly")
            print(name)
            sys.exit(1)

        if animation_type == "2D" or animation_type == "triple":
            if animation_specific == "unitless":
                self.unitless = True
                self.memory_space_norm = ""
            else:
                self.unitless = False

        if animation_type == "fourier":
            if animation_specific[0] == "x" or animation_specific[0] == "y":
                self.fourier_type = "princpile"
                self.fourier_direc = animation_specific[0]
                self.fourier_loc = animation_specific[1]

            elif animation_specific[0] == "diag":
                self.fourier_type = "diag"
                self.fourier_direc = animation_specific[1]

            elif animation_specific[0] == "trace":
                self.fourier_type = "trace"
                self.fourier_loc_x = animation_specific[1]
                self.fourier_loc_y = animation_specific[2]

            elif animation_specific == "trace_diag":
                self.fourier_type = "trace_diag"

            else:
                print("fourier spec defined incorrectly")
                print(name)
                sys.exit(1)

        self.animation_type = animation_type

        self.variable = translate[variable][0]
        self.variable_name = variable

        self.component = component

        self.unit = translate[variable][1]
        self.unit_name = translate[variable][2]

        self.bulkpath = bulkpath

        self.name = name
        self.memory_space = ""
        self.shape = 0
        self.dtype = ""