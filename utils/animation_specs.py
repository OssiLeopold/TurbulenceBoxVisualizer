import sys

# Dictionary for translating instructions for VlsvReader
translate = {"B":("vg_b_vol", 1e-9, "nT"),
                "v":("proton/vg_v", 1e3, "km/s"),
                "J":("vg_j", 1e-9, r"$\frac{nA}{m^2}$"),
                "rho":("proton/vg_rho", 1e6, r"$\frac{n}{m^3}$")}

# Defining AnimationSpecs object and checking instructions
class AnimationSpecs():
    def __init__(
        self, animation_type, variable, component, 
        animation_specific, bulkpath, filetype
        ):
        if animation_type not in ["2D", "triple", "fourier", "sf", "kurtosis", "rms"]:
            print("animation_type defined incorrectly")
            sys.exit(1)

        if variable not in ["B", "v", "J", "rho", "bv"]:
            print("variable defined incorrectly")
            sys.exit(1)

        if filetype not in [".mp4"]:
            print("filetype defined incorrectly")
            sys.exit(1)

        if animation_type == "2D" or animation_type == "triple":
            if animation_specific == "unitless":
                self.unitless = True
                self.memory_space_norm = ""
            else:
                self.unitless = False
        
        self.fourier_direc = ""
        if animation_type == "fourier":
            if animation_specific[0] == "x" or animation_specific[0] == "y":
                self.fourier_type = "principle"
                self.fourier_direc = animation_specific[0]
                self.fourier_loc = animation_specific[1]

            elif animation_specific[0] == "diag":
                self.fourier_type = "diag"
                self.fourier_direc = animation_specific[1]

            elif animation_specific[0] == "trace":
                self.fourier_type = "trace"
                self.fourier_loc_x = animation_specific[1]
                self.fourier_loc_y = animation_specific[2]

            elif animation_specific[0] == "trace_diag":
                self.fourier_type = "trace_diag"

            elif animation_specific[0] == "1D":
                self.fourier_type = "1D"

            elif animation_specific[0] == "2D":
                self.fourier_type = "2D"

            else:
                print("fourier spec defined incorrectly")
                sys.exit(1)

        if animation_type in ["sf","kurtosis"]:
            self.delta_ls = animation_specific

        self.animation_type = animation_type

        if variable == "bv":
            self.variable = ["vg_b_vol", "proton/vg_v"]
        else:
            self.variable = translate[variable][0]
            
            self.unit = translate[variable][1]
            self.unit_name = translate[variable][2]

        self.variable_name = variable
        self.component = component

        self.bulkpath = bulkpath

        self.name = ""
        self.memory_space = {}
        self.shape = {}
        self.dtype = ""
        
        self.time = ""
        self.time_shape = 0
        self.time_dtype = ""




















"""                                               ......                                             
                                            .....'''''''.....                                       
                                        ...''''''''''',,;;::;'.                                     
                                      ...',,,,,,,;;:ldxkkkxool;.                                    
                                    .,;'.'cxkdlldxxxxxkOOxdkOO00x'                                  
                                   .:oo:'.,oxdO0OKNNWXkxkdxK00NMNx;;,'.                             
                                  ;odkOOxddkxxKNXNMMMNOk0kxOO000Ok0K00Od;.                          
                                 ,ok000000000kk0XNNNKOk00000000kdO00000KKx'                         
                                .lxO0000000000OkOOOOOO00kkkOOOkdxO000000KKk'                        
                               .:dk0000000000OxxkOO00000kxdddxxkkk0000000KKd.                       
                               'oxO000000000000OkkkkkkkkkkkkkkkxooO0000000K0;                       
                               :dk0000000000000000000OOOkkkkkOOOOkxk0000000Ko.                      
                              .lxk000000000000000000000000000000Oxxk0000000Kk'                      
                              'oxO00000000000000000000000000000kdk000000000KO;                      
                              ,dxO0000000000000000000000000000koxO000000000K0:                      
                              ;dxO0000000000000000000000000000OkkkkO000000000c                      
                              :dxO00000000000000OOO000000000000000kx000000000c                      
                              :dxO00000000000000xdkO00000000000000OxO00000000c                      
                             .:dxO00000000000000kooxO0000Oddk0000kdk000000000c                      
                           .:oloxO000000000000000kddddxxddddddxxxxk0000000000c                      
                         ..:xOxook000000000000000000OkkkO0000OO0000000000000Od:.                    
                     .':oxkxdxOOkkO00000000000000000000000000000000000000000k0Xo..                  
                  .;oxO00KKKOxxKWNKOOO000000000000000000000000000000000000Ok0WN0K0xc.               
               ':dkO0KXNWWMWWXOkKWMWKOkO000000000000000000000000000000000OkKWN0XMMMWXkl'            
            .:dkO0KXWMMMMMMMMMWX0OKWMWX0OkO0000000000000000000000000000OkONWX0XWMMMMMMMXk:.         
          'lkOO0KNWMMMMMMMMMMMMMWN00KNMMWX0OOO00000000000000000000000OkOXWWK0NMMMMMMMMMMMWKo'       
        ,okOOOKNWMMMMMMMMMMMMMMMMMMNK00KWMMWXK0OOOOO000000000000000kk0XWMN0KWMMMNNMMMMMMMMMWXxc:.   
      'lxkkkkOXWMMMMMMMMMMMMMMMMMMMMMWX000XWMMMWXK000OOOOO000000OkOKNMMNK0XWMMMMK0WMMMMMMMMMMW0K0,  
   .':dxk000000KKKKXWMMMMMMMMMMMMMMMMMMMWXKKKKNWMMMMWNXKK0kxxkOxdk0KWWX0XWMMMMMMKkXMMMMMMMMMMW0KNx:.
  ;dkOO0XNNWWWMMWNXKKKXWMMMMMMMMMMMMMMMMMMMWNXKKKKXNWMMMNKKK0OdxXWXOO0KWMMMMMMMMNkONMMMMMMMMMX0XX0Xd
 'dkOKXNNNNNNNWWMMMMWNKKXWMMMMMMMMMMMMMMMMMMMMMWNKKKKKKK0KMMMNOkXMMWNWMMMMMMMMMMMXkONMMMMMMMN0KWXNWd
.ckOkkkxxxxxxkOO0KXNMMMNKKNMMMMWNWMMMMMMMMMMMMMMMMMMWNXKXWMMW0kO0NMMMMMMMMMMMMMMMMXkkXWMMMWKOKWMMW0;
ckxoooodxkO000000OkO0XWMWX0NMWXKXWMMMMMMMMMMMMMMMMMMMMMMMMMMMWWWX0NMMMMMMMMMMMMMMMMNOx0XKOOOXMMWX0x;
,clddddxO00000000000Ok0NMWK0KKKNMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMWK0NMMMMMMMMMMMMMMMMNOddxOKNWNKOkOO;
 .ldddxO00000000000000OOXX0kkXWMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMWK0NMMMMMMMMMMMMMMMMW0ooddxkkkO0Kx.
  :dddk000000000000000KOOKkkNMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMNKkOWMMMMMMMMMMMMMMMMW0lldkO00000l 
  'odxO0000000000000000KOkkKMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMXOOkxOKXNWMMMMMMMMMMMMW0dxO00000k' 
   :dxO0000000000000000K0xOWMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMWWXkkKNWMMMMMMMMMMMMMMW0dx0000Oc  
   .cdk00000000000000000KOONMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMWK0NMMMMMMMMMMMMMMMMW0dk000d.  
    .cxO000000000000000000OKMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMX0NMMMMMMMMMMMMMMMMWOx00k;   
     .ck00000000000000000KOONMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMX0NMMMMMMMMMMMMMMMMXkk0l.   
      .ck00000000000000000KO0WMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMWX0k0WMMMMMMMMMMMMMMMW0k0l.   
        ;x0000000000000000K0OXMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMWKO00XMMMMMMMMMMMMMMMMKkOO,   
        'ldO000000000000000KOKMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMK0WMMMMMMMMMMMMMMMXkOO;   
      .,coodO0000Odx0000000KO0MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMK0WMMMMMMMMMMMMMMMKkOkc.  
     .lkOOkdoxkOOdlx0000000KOKMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMX0NMMMMMMMMMMMMMMWOkkxOc  
      ':ooloxxoloocokO000000OKMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMK0WMMMMMMMMMMMMMWKxO0Ox,  
         .:k0kooOkocdkxk000OONMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMNKKOKMMMMMMMMMMWNWWOoxOOo'   
          ,xOdlx0OolO0Oxk0Ok0WMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMXOkkXMMMMMMMMWX0Ox:.....     
           ,cc:lxxllk00OOxod0XNNXNWMMMMMMMWWMMMMMMMMMMMMMMMMMMMMMMMMMMMMN0KWMMMMMMN0dc,.            
            .'.';,'.;clc,...,:cooldkKNWWMMWK0KNWMMMMMMMMMMMMMMMMMMMMMMWKOKWWWNX0xo:.                
            .'''...                ..;coxkOkxddxO0KXNWWWWWWWWWWNNXK0Okdoooc:;,....'. ..,.           
             .'''''.....                 ..;;,.....',::cccccc:codddddxxxxo'      .,'.';,.           
              .''''''''''......           .,'.                .lxkkkxxkkxl.  ....',,,,,.            
               .''''''',,,,,,''''..........''.                .'clooollc;'..'',,,,,,,,.             
                ..'''''',,,,,,,,,,,,,,,,''',''.................'''','''',,,,,,,,,,,,'.              
                 ..'''''',,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,.                
                   .'''''',,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,'..                 
                    .''''',,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,'.                    
                     .''''',,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,'.                     
                     .'''',,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,'''',,,,,,,,,,,,,,.                      
                     .'''',,,,,,,,,,,,,,,,,,,,,,,,,,,''''''''',,,,,,,,,,,,,,,.                      
                    .''''',,,,,,,,,,,,,,,,,,,,,,,,,,,'..''',,,,,,,,,,,,,,,,,;.                      
                    .'''',,,,,,,,,,,,,,,,,,,,,,,,,,,,'''',,,,,,,,,,,,,,,,,,,;'                      
                    .'''',,,,,,,,,,,,,,,,,,,,,,,,,',,'''',,,,,,,,,,,,,,,,,,,;,                      
                   .''''',,,,,,,,,,,,,,,,,,,,,,,,,,;,''',,,,,,,,,,,,,,,,,,,,;;.                     
                   .'''',,,,,,,,,,,,,,,,,,,,,,,,,,,;,''',,,,,,,,,,,,,,,,,,,,,;.                     
                   .'''',,,,,,,,,,,,,,,,,,,,,,,,,,,;,''',,,,,,,,,,,,,,,,,,,,,;'                     
                  ..'''',,,,,,,,,,,,,,,,,,,,,,,,,,,;;'.',,,,,,,,,,,,,,,,,',,,;,.                    
                  ..'''',,,,,,,,,,,,,,,,,,,,,,,,,,',;'.'',,,,,,,,'''.........',.                    
                   .'''',,,,,,,,,,,,,,,'''......''',;'.'',,,,,''.................                   
                   .'''',,,,,,,,,,'''...............'..''''''.......................                
                   .'''',,,,,,'''....................................................               
                    ....''''........................................................'.              
                     ....................................'............................              
                     .....................................'...........................              
                     ...............................................................                
                        ...................................                                         
                              ........................                                              
                                       .......                                                       """