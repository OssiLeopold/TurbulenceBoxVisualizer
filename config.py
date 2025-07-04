from configparser import ConfigParser

config = ConfigParser()

config["local"] = {
    "ffmpeg_path" : "/home/rxelmer/Documents/turso/appl_local/ffmpeg/bin/ffmpeg",
    "latex_path" : "/home/rxelmer/Documents/turso/appl_local/tex-basic/texlive/2023/bin/x86_64-linux:",
    "bulkpath" : "/home/rxelmer/Documents/turso/bulks/sim22/"
}

config["turso"] = {
    "ffmpeg_path" : "/wrk-vakka/group/spacephysics/proj/appl/ffmpeg/bin/ffmpeg",
    "latex_path" : "/wrk-vakka/group/spacephysics/proj/appl/tex-basic/texlive/2023/bin/x86_64-linux:",
    "bulkpath" : "/wrk-vakka/group/spacephysics/turbulence/turbulence_dev/bulks/sim22/"
}