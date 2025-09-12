import ffmpeg
import os
import wave
import os_toolkit as ost
import pandas as pd



input_path = r"C:\C_Video_Python\Learn German\Learn German with Nico\Splitted Audio\Learn German with Nico_A1"

paths_df = pd.DataFrame(ost.get_full_filename(input_path))

