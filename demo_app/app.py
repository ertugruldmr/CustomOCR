import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import StringLookup
import gradio as gr

# File Paths
max_length = 5
model_path = "ocr_model"

ocr_model = tf.keras.models.load_model(model_path)

char_to_num_voc = pickle.load(open("char_to_num", 'rb'))
num_to_char_voc = pickle.load(open("num_to_char", 'rb'))

char_to_num = StringLookup(vocabulary=char_to_num_voc, mask_token=None)
num_to_char = StringLookup(vocabulary=num_to_char_voc, mask_token=None, invert=True)

def encode_img(image_path, img_height=50, img_width=200):
    
    # Load the image file using a file I/O operation
    img = tf.io.read_file(image_path)

    # Decode the image file as a tensor using tf.image.decode_image()
    img = tf.image.decode_image(img, channels=1)

    # Convert the image tensor to float32 data type and rescale the pixel values to the range [0, 1]
    img = tf.image.convert_image_dtype(img, tf.float32)
    
    # 4. Resize to the desired size
    img = tf.image.resize(img, [img_height, img_width])
    
    # 5. Transpose the image because we want the time
    # dimension to correspond to the width of the image.
    img = tf.transpose(img, perm=[1, 0, 2])
    
    return img

def decode_batch_predictions(pred):

    # creating convenient data
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    
    # Use greedy search. For complex tasks, you can use beam search
    decode_params = {"y_pred":pred, "input_length":input_len, "greedy":True}
    results_all = tf.keras.backend.ctc_decode(**decode_params)
    results = results_all[0][0][:, :max_length]
    
    # converting the results into text format
    output_text = [
        tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8") 
        for res in results
    ]

    return output_text[0]

def extract_text(image_path):

  # loading the image
  img = encode_img(image_path)

  # Adjusting the sahpe to predict
  batch_image = tf.expand_dims(img, 0)

  # prediction
  pred = ocr_model.predict(batch_image)
  
  # decoding the predictions (y pred)
  pred_text = decode_batch_predictions(pred)
  
  return pred_text

with gr.Blocks() as demo:
    
    # creating the components
    img_path = gr.Image(type="filepath", source="upload")
    extract_btn = gr.Button("Extract text form image")
    output = gr.Textbox()

    # connecting the button functions
    extract_btn.click(extract_text, inputs=img_path, outputs=output)

    # setting the exmples
    gr.Examples("Examples", img_path)

# Launching the demo
if __name__ == "__main__":
    demo.launch()
