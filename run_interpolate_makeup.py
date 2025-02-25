import subprocess

# Define the command as a list
command = [
    'python', 'interpolate_makeup.py',
    '--align_face',
    '--style', 'makeup',
    '--name', 'makeup',
    '--content', './data/test/003767.png',
    '--content2', './data/test/083311.png',
    '--makeup_name_1', 'refined_makeup_code.npy',
    '--style_id', '0',
    '--makeup_name_2', 'refined_makeup_code.npy',
    '--style_id2', '0'
]

# Print the command for verification
print("Running interpolate_makeup with command:")
print(" ".join(command))

# Execute the command
subprocess.run(command)
