import gradio as gr
import cv2
import numpy as np

# Function to convert 2x3 affine matrix to 3x3 for matrix multiplication
def to_3x3(affine_matrix):
    return np.vstack([affine_matrix, [0, 0, 1]])

# Function to apply transformations based on user inputs
def apply_transform(image, scale, rotation, translation_x, translation_y, flip_horizontal):

    # Convert the image from PIL format to a NumPy array
    image = np.array(image)
    # Pad the image to avoid boundary issues
    pad_size = min(image.shape[0], image.shape[1]) // 2
    image_new = np.zeros((pad_size*2+image.shape[0], pad_size*2+image.shape[1], 3), dtype=np.uint8) + np.array((255,255,255), dtype=np.uint8).reshape(1,1,3)
    image_empyt=np.zeros((pad_size*2+image.shape[0], pad_size*2+image.shape[1], 3), dtype=np.uint8) + np.array((255,255,255), dtype=np.uint8).reshape(1,1,3)

    ### FILL: Apply Composition Transform 
    # Note: for scale and rotation, implement them around the center of the image （围绕图像中心进行放缩和旋转）
    h = int(scale * image.shape[0])
    w = int(scale * image.shape[1])
    image_s = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            image_s[i, j] = image[int(i//scale), int(j//scale)]

    image_new[(image_new.shape[0]-h)//2:(image_new.shape[0]-h)//2+h, \
              (image_new.shape[1]-w)//2:(image_new.shape[1]-w)//2+w] = image_s
 

    #R = np.array([[np.cos(rotation), -np.sin(rotation), 0], [np.sin(rotation), np.cos(rotation), 0], [0, 0, 1]])
    #T = np.array([[1, 0, translation_x], [0, 1, translation_y], [0, 0, 1]])
    the = np.pi * rotation / 180
    M = np.array([[np.cos(the), -np.sin(the), translation_y], \
                  [np.sin(the), np.cos(the), translation_x], \
                    [0, 0, 1]])
    image_M=image_empyt
    x, y = h //2 , w // 2
    center_x, center_y = (image_new.shape[0]-h) // 2 + x, (image_new.shape[0]-h) // 2 + y
    for i in range(-x, h-x):
        for j in range(-y, w - y):
            mulcor = np.array([[i], [j], [1]])
            mulcor = np.dot(M, mulcor)
            x0, y0 = int(mulcor[0, 0]+center_x), int(mulcor[1, 0]+center_y)
            if x0 < image_new.shape[0] and x0 >= 0 and y0 < image_new.shape[1] and y0 >= 0: 
                image_M[x0, y0] = image_new[center_x + i, center_y + j]

    image_new=image_M

    if flip_horizontal == 1:
        image_new = np.flip(image_new, axis=1)




    transformed_image = np.array(image_new)
    return transformed_image

# Gradio Interface
def interactive_transform():
    with gr.Blocks() as demo:
        gr.Markdown("## Image Transformation Playground")
        
        # Define the layout
        with gr.Row():
            # Left: Image input and sliders
            with gr.Column():
                image_input = gr.Image(type="pil", label="Upload Image")

                scale = gr.Slider(minimum=0.1, maximum=2.0, step=0.1, value=1.0, label="Scale")
                rotation = gr.Slider(minimum=-180, maximum=180, step=1, value=0, label="Rotation (degrees)")
                translation_x = gr.Slider(minimum=-300, maximum=300, step=10, value=0, label="Translation X")
                translation_y = gr.Slider(minimum=-300, maximum=300, step=10, value=0, label="Translation Y")
                flip_horizontal = gr.Checkbox(label="Flip Horizontal")
            
            # Right: Output image
            image_output = gr.Image(label="Transformed Image")
        
        # Automatically update the output when any slider or checkbox is changed
        inputs = [
            image_input, scale, rotation, 
            translation_x, translation_y, 
            flip_horizontal
        ]

        # Link inputs to the transformation function
        image_input.change(apply_transform, inputs, image_output)
        scale.change(apply_transform, inputs, image_output)
        rotation.change(apply_transform, inputs, image_output)
        translation_x.change(apply_transform, inputs, image_output)
        translation_y.change(apply_transform, inputs, image_output)
        flip_horizontal.change(apply_transform, inputs, image_output)

    return demo

# Launch the Gradio interface
interactive_transform().launch()
