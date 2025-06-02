
import os
import subprocess

# Get all PNG images in the current folder
image_files = sorted([f for f in os.listdir("./output_history") if f.endswith(".png")])

# Create file_list.txt
with open("file_list.txt", "w") as f:
    for img in image_files:
        f.write(f"file './output_history/{img}'\n")

print("Created file_list.txt with sorted images.")

# Run ffmpeg to make the video
ffmpeg_cmd = [
    "ffmpeg", "-r", "8", "-f", "concat", "-safe", "0", "-i", "file_list.txt",
    "-vf", "scale=256:256:flags=neighbor", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
    "output.mp4"
]

print("Running FFmpeg...")
subprocess.run(ffmpeg_cmd)

print("Video successfully created: output.mp4")