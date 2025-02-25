import subprocess

style = "makeup"
style_id = 0  # Set the specific style_id you want to use
img = "003767.png"  # Set the specific image

if style_id == 0:
    command = [
        'python', 'makeup_transfer.py',  
        '--style', style,  
        '--name', 'makeup',  
        '--style_id', str(style_id),
        '--exstyle_name', 'refined_makeup_code.npy',   
        '--content', './data/test/' + img,  
        '--weight', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', 
        '--output_path', './output/makeup/',  
        '--align_face'
    ]
    print("Running style_transfer with style_id:", style_id) 
    subprocess.run(command)
