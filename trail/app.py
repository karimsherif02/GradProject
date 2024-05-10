from flask import Flask, request, jsonify, send_file, render_template
from werkzeug.utils import secure_filename
import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
import random
import os
import tempfile

app = Flask(__name__)
UPLOAD_FOLDER = 'C:/Users/user/OneDrive/Desktop/trail/uploads'
UPLOAD_FOLDER_VIDEO = 'C:/Users/user/OneDrive/Desktop/trail/uploads'
UPLOAD_FOLDER_ENCRYPT = 'C:/Users/user/OneDrive/Desktop/trail/uploads'
UPLOAD_FOLDER_DECRYPT = 'C:/Users/user/OneDrive/Desktop/trail/uploads'
UPLOAD_FOLDER_IMAGE = 'C:/Users/user/OneDrive/Desktop/trail/uploads'
UPLOAD_FOLDER_IMAGE_de = 'C:/Users/user/OneDrive/Desktop/trail/uploads'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

app.config['UPLOAD_FOLDER_IMAGE'] = UPLOAD_FOLDER_IMAGE
app.config['UPLOAD_FOLDER_IMAGE_de'] = UPLOAD_FOLDER_IMAGE_de

app.config['UPLOAD_FOLDER_VIDEO'] = UPLOAD_FOLDER_VIDEO
app.config['UPLOAD_FOLDER_ENCRYPT'] = UPLOAD_FOLDER_ENCRYPT
app.config['UPLOAD_FOLDER_DECRYPT'] = UPLOAD_FOLDER_DECRYPT


# Function to hide secret text within cover text
def hide_text(cover_text, secret_text):
    zero_width_secret = encode_message("", secret_text)
    stego_text = cover_text[:len(secret_text)] + zero_width_secret + cover_text[len(secret_text):]
    return stego_text

# Function to save stego text to a new file
def save_stego_text(stego_text, output_file):
    output_file = output_file.strip('"')  # Remove any extra quotes
    with open(output_file, 'wb') as file:
        file.write(stego_text.encode())

# Function to encode the message using zero-width steganography
def encode_message(original_text, secret_message):
    binary_secret = ''.join(format(ord(char), '08b') for char in secret_message)
    encoded_text = original_text

    for bit in binary_secret:
        if bit == '0':
            encoded_text += '​'  # Zero-width space
        else:
            encoded_text += '‌'  # Zero-width non-joiner

    return encoded_text

# Function to convert zero-width characters back to text
def zero_width_to_text(zero_width_text):
    text = ''
    # Split the zero-width text into chunks of 8 characters
    zero_width_chunks = [zero_width_text[i:i+8] for i in range(0, len(zero_width_text), 8)]
    for chunk in zero_width_chunks:
        # Convert each zero-width chunk to its corresponding character
        char = chr(int(chunk.replace('​', '0').replace('‌', '1'), 2))
        text += char
    return text

# Function to decode a message encoded using zero-width steganography
def decode_message(encoded_text):
    binary_secret = ''
    for char in encoded_text:
        if char == '​':
            binary_secret += '0'
        elif char == '‌':
            binary_secret += '1'

    # Split the binary string into 8-bit chunks
    binary_chunks = [binary_secret[i:i+8] for i in range(0, len(binary_secret), 8)]

    # Convert each binary chunk to its corresponding character
    decoded_message = ''.join([chr(int(chunk, 2)) for chunk in binary_chunks])
    return decoded_message

# Function to extract hidden text from stego text
def extract_text(stego_text):
    zero_width_text = ''
    for char in stego_text:
        if char == '​' or char == '‌':  # Check for zero-width characters
            zero_width_text += char
    extracted_text = decode_message(zero_width_text)
    return extracted_text

# Function to decrypt video
def decrypt_video(encrypted_video_path):
    cap = cv2.VideoCapture(encrypted_video_path)
    if not cap.isOpened():
        return None

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        return None

    locss = []
    st = ""
    height, width, _ = frames[0].shape

    if not height or not width:
        return None

    random.seed(42)
    indices = list(range(height * width * 3))
    random.shuffle(indices)

    for file_n, bit_index in enumerate(range(0, 160)):
        pixel_index = indices[bit_index]
        i_pixel = pixel_index // (width * 3)
        j_pixel = (pixel_index % (width * 3)) // 3
        color_channel = pixel_index % 3
        locss.append([bit_index % len(frames), i_pixel, j_pixel, color_channel])

    for i, loc in enumerate(locss):
        pixel_value = frames[loc[0]][loc[1]][loc[2]][loc[3]]
        st += format(pixel_value, '08b')[-1]

    hidden_data = [st[i:i+8] for i in range(0, len(st), 8)]
    reconstructed_data = bytearray([int(chunk, 2) for chunk in hidden_data])

    for file_n, bit_index in enumerate(range(160, int(reconstructed_data.split()[0]) + 160)):
        pixel_index = indices[bit_index]
        i_pixel = pixel_index // (width * 3)
        j_pixel = (pixel_index % (width * 3)) // 3
        color_channel = pixel_index % 3
        locss.append([bit_index % len(frames), i_pixel, j_pixel, color_channel])

    st = ""
    for i, loc in enumerate(locss[160:]):
        pixel_value = frames[loc[0]][loc[1]][loc[2]][loc[3]]
        st += format(pixel_value, '08b')[-1]

    hidden_data = [st[i:i+8] for i in range(0, len(st), 8)]
    reconstructed_data = bytearray([int(chunk, 2) for chunk in hidden_data])

    extracted_file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'extracted_file.txt')
    with open(extracted_file_path, "wb") as f:
        f.write(reconstructed_data)

    return extracted_file_path

@app.route('/') #index (main page)
def index():
    return render_template('index2.html')

@app.route('/intro_to_stego') #intro page from index page
def intro_to_stego():
    return render_template('intro_to_stego.html')

@app.route('/about_us') #about page from index page
def about_us():
    return render_template('about_us.html')
#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
@app.route('/home2') # renders home2.html
def home2():
    return render_template('home2.html')

@app.route('/image_stego') # render image pg from home2
def image_stego():
    return render_template('image_stego.html')

@app.route('/video_stego') # render video pg from home2
def video_stego():
    return render_template('video_stego.html')

@app.route('/decrypt_image') 
def decrypt_image():
    return render_template('decrypt_image.html')


@app.route('/encrypt') #hide data for text
def encrypt_form():
    return render_template('encrypt_form.html')

@app.route('/decrypt') #extract data for text
def decrypt_form():
    return render_template('decrypt_form.html')

@app.route('/start_stego', methods=['POST'])
def start_stego():
    return render_template('home2.html')

@app.route('/encrypt', methods=['POST']) # text 
def encrypt():
    cover_text = request.files['cover_text'].read().decode('utf-8')
    secret_text = request.files['secret_text'].read().decode('utf-8')
    output_file = request.form['output_file']  # Get the output file path from the form data

    stego_text = hide_text(cover_text, secret_text)
    save_stego_text(stego_text, os.path.join(app.config['UPLOAD_FOLDER_ENCRYPT'], output_file))

    return jsonify({"message": "Stego text saved successfully."})

@app.route('/decrypt', methods=['POST']) # text
def decrypt():
    stego_text = request.files['stego_text'].read().decode('utf-8')
    extracted_text = extract_text(stego_text)
    
    # Save decoded text to a temporary file
    temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
    temp_file.write(extracted_text)
    temp_file.close()
    
    # Send the temporary file as a downloadable attachment
    return send_file(temp_file.name, as_attachment=True, download_name='decoded_text.txt')

# Video encryption functionality
@app.route('/video_encrypt', methods=['GET', 'POST'])
def video_encrypt():
    if request.method == 'POST':
        # Get uploaded files
        video_file = request.files['video']
        file_to_hide = request.files['fileToHide']

        # Save uploaded files
        video_path = os.path.join(app.config['UPLOAD_FOLDER_VIDEO'], video_file.filename)
        video_file.save(video_path)
        file_to_hide_path = os.path.join(app.config['UPLOAD_FOLDER_VIDEO'], file_to_hide.filename)
        file_to_hide.save(file_to_hide_path)

        # Read the file to hide
        with open(file_to_hide_path, "rb") as file:
            file_data = file.read()

        binary_data = ''.join(format(i, '08b') for i in file_data)

        # Video processing
        cap = cv2.VideoCapture(video_path)
        fourcc = cv2.VideoWriter_fourcc(*'FFV1')
        out_path = os.path.join(app.config['UPLOAD_FOLDER_VIDEO'], 'encrypted_video.avi')
        out = cv2.VideoWriter(out_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (int(cap.get(3)), int(cap.get(4))))
        frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)

        cap.release()

        print(f"Number of frames read: {len(frames)}")

        random.seed(42)

        locs = []

        file_data0 = f"{len(binary_data)}                    "
        binary_data0 = ''.join(format(ord(i), '08b') for i in file_data0[:20])

        height, width, _ = np.array(frames[0]).shape
        max_bytes = height * width * 3 // 8
        if len(binary_data) > max_bytes:
            raise ValueError("File size is too large to be stored in the image.")
        random.seed(42)
        indices = list(range(height * width * 3))
        random.shuffle(indices)
        chunk_size = 1
        for file_n, bit_index in enumerate(range(0, len(binary_data) + 160, chunk_size)):
            if bit_index < 160:
                chunk = binary_data0[file_n:file_n + chunk_size]
            else:
                file_ind = file_n - 160
                chunk = binary_data[file_ind:file_ind + chunk_size]
            for i, bit in enumerate(chunk):
                pixel_index = indices[bit_index + i]
                i_pixel = pixel_index // (width * 3)
                j_pixel = (pixel_index % (width * 3)) // 3
                color_channel = pixel_index % 3
                locs.append([i_pixel, j_pixel, color_channel])
                pixel_value = frames[bit_index % np.array(frames).shape[0]][i_pixel][j_pixel][color_channel]
                pixel_bin = format(pixel_value, '08b')
                pixel_bin = pixel_bin[:-1] + bit
                new_pixel_value = int(pixel_bin, 2)
                frames[bit_index % len(frames)][i_pixel][j_pixel][color_channel] = new_pixel_value

        for frame in frames:
            out.write(frame.astype(np.uint8))

        out.release()

        print(f"Number of frames written to encrypted video: {len(frames)}")

        return jsonify({'message': 'Video processed successfully', 'success': True, 'videoUrl': '/download/encrypted_video'})

    return render_template('video_encrypt.html')

# Video decryption functionality
@app.route('/video_decrypt', methods=['GET', 'POST'])
def video_decrypt():
    if request.method == 'POST':
        # Get uploaded encrypted video file
        encrypted_video_file = request.files['encrypted_video']

        # Define the upload folder for decryption
        UPLOAD_FOLDER_DECRYPT = 'C:/Users/user/OneDrive/Desktop/trail/uploads'#C:\xampp\htdocs\trail\uploads
        app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER_DECRYPT
        
        # Save uploaded encrypted video
        encrypted_video_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(encrypted_video_file.filename))
        encrypted_video_file.save(encrypted_video_path)

        # Decrypt the video
        decrypted_file_path = decrypt_video(encrypted_video_path)

        if decrypted_file_path:
            # Send the decrypted file as a downloadable attachment
            return send_file(decrypted_file_path, as_attachment=True, download_name='decrypted_video.txt')
        else:
            return jsonify({'error': 'Failed to decrypt video.'}), 500

    return render_template('video_decrypt.html')


# Function to hide file in image using RBS steganography
def hide_file_in_image(image_path, file_to_hide_path):
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        return jsonify({'error': "Couldn't read the image."}), 500

    # Read file to hide
    with open(file_to_hide_path, 'rb') as file:
        file_data = file.read()

    # Convert file data to binary
    binary_data = ''.join(format(byte, '08b') for byte in file_data)
    file_data0 = f"{len(binary_data)}                    "
    binary_data0 = ''.join(format(ord(char), '08b') for char in file_data0[:20])

    height, width, _ = image.shape
    max_bytes = height * width * 3 // 8

    if len(binary_data) > max_bytes:
        return jsonify({'error': "File size is too large to be stored in the image."}), 400

    random.seed(42)
    indices = list(range(height * width * 3))
    random.shuffle(indices)

    # Hide data in the image
    locs = []
    chunk_size = 1
    for file_n, bit_index in enumerate(range(0, len(binary_data)+160, chunk_size)):
        if bit_index < 160:
            chunk = binary_data0[file_n:file_n + chunk_size]
        else:
            file_ind = file_n - 160
            chunk = binary_data[file_ind:file_ind + chunk_size]
        for i, bit in enumerate(chunk):
            pixel_index = indices[bit_index + i]
            i_pixel = pixel_index // (width * 3)
            j_pixel = (pixel_index % (width * 3)) // 3
            color_channel = pixel_index % 3
            locs.append([i_pixel, j_pixel, color_channel])
            pixel_value = image[i_pixel, j_pixel, color_channel]
            pixel_bin = format(pixel_value, '08b')
            pixel_bin = pixel_bin[:-1] + bit
            new_pixel_value = int(pixel_bin, 2)
            image[i_pixel, j_pixel, color_channel] = new_pixel_value

    # Save the modified image
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'output_image.png')
    cv2.imwrite(output_path, image)

    return output_path

# Route to handle file hiding in an image
@app.route('/hide_file_in_image', methods=['POST'])
def hide_file_in_image_route():
    # Remove existing files in the directory
    for i in os.listdir(app.config['UPLOAD_FOLDER']):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], i)
        if os.path.isfile(file_path) and i != 'output_image.png':
            try:
                os.remove(file_path)
            except Exception as e:
                return jsonify({'error': f"Error deleting file '{i}': {e}"}), 500

    # Receive image and file from the request
    image_file = request.files.get('image')
    file_to_hide = request.files.get('file')

    if not image_file or not file_to_hide:
        return jsonify({'error': 'Image or file not provided.'}), 400

    # Save received files
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'received_image.png')
    image_file.save(image_path)
    file_to_hide_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file_to_hide.filename))
    file_to_hide.save(file_to_hide_path)

    # Hide file in the image
    output_path = hide_file_in_image(image_path, file_to_hide_path)

    # Remove temporary files
    os.remove(image_path)
    os.remove(file_to_hide_path)

    return jsonify({'message': 'File hidden successfully in the image.', 'output_path': output_path}), 200

@app.route('/encrypt_image', methods=['GET', 'POST'])
def encrypt_image():
    if request.method == 'POST':
        # Get uploaded files
        image_file = request.files['image']
        file_to_hide = request.files['file']

        # Save uploaded files
        image_path = os.path.join(app.config['UPLOAD_FOLDER_IMAGE'], image_file.filename)
        image_file.save(image_path)
        file_to_hide_path = os.path.join(app.config['UPLOAD_FOLDER_IMAGE'], file_to_hide.filename)
        file_to_hide.save(file_to_hide_path)

        # Hide file in the image
        output_path = hide_file_in_image(image_path, file_to_hide_path)

        return send_file(output_path, as_attachment=True, attachment_filename='encrypted_image.png')

    return render_template('encrypt_image.html')

#######################################################################################################################

def browse_file():
    root = tk.Tk()
    root.withdraw()  
    file_path = filedialog.askopenfilename()  
    return file_path

def decode_hidden_file(image_path, hidden_len=True, hd_ln=160):
    image = cv2.imread(image_path)
    if image is None:
        return None
    
    hidden_data = bytearray()
    st = ""
    height, width, _ = image.shape
    indices = list(range(height * width * 3))
    random.seed(42)
    random.shuffle(indices)
    
    if hidden_len:
        start_p = 0
        end_p = 160
    else:
        start_p = 160
        end_p = hd_ln
    
    locss = []
    chunk_size = 1
    
    for file_n, bit_index in enumerate(range(start_p, start_p + end_p, chunk_size)):
        pixel_index = indices[bit_index]
        i_pixel = pixel_index // (width * 3)
        j_pixel = (pixel_index % (width * 3)) // 3
        color_channel = pixel_index % 3
        locss.append([i_pixel, j_pixel, color_channel])

    for i, loc in enumerate(locss):
        pixel_value = image[loc[0], loc[1], loc[2]]
        hidden_bit = int(format(pixel_value, '08b')[-1])
        st = st + format(pixel_value, '08b')[-1]
    
    return st

@app.route('/extract', methods=['POST'])
def extract_hidden_file():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file sent'}), 400

    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Save the uploaded image file to the desired folder
    image_path = os.path.join('C:\\Users\\user\\OneDrive\\Desktop\\t\\uploads', image_file.filename)
    image_file.save(image_path)

    hidden_data = decode_hidden_file(image_path, True)
    if hidden_data is not None:
        hidden_data = [hidden_data[i:i+8] for i in range(0, len(hidden_data), 8)]
        reconstructed_data = bytearray([int(chunk, 2) for chunk in hidden_data])

        hidden_data_len = reconstructed_data.split()[0]
        hidden_data = decode_hidden_file(image_path, False, int(hidden_data_len))
        if hidden_data is not None:
            hidden_data = [hidden_data[i:i+8] for i in range(0, len(hidden_data), 8)]
            reconstructed_data = bytearray([int(chunk, 2) for chunk in hidden_data])

            output_file_path = os.path.join('C:\\Users\\user\\OneDrive\\Desktop\\trail\\downloads', 'extracted_file.txt')
            with open(output_file_path, "wb") as f:
                f.write(reconstructed_data)

            # Return the path to the extracted file
            return jsonify({'message': 'File extracted successfully!', 'file_path': '/download/extracted_file.txt'}), 200
        else:
            return jsonify({'error': 'Failed to extract hidden data'}), 500
    else:
        return jsonify({'error': 'Failed to extract hidden data'}), 500

@app.route('/download/<file_name>')
def download_files(file_name):
    file_path = os.path.join('C:\\Users\\user\\OneDrive\\Desktop\\trail\\downloads', file_name)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    else:
        return jsonify({'error': 'File not found'}), 404

#######################################################################################################################
# Add a route to download the encrypted video
@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER_VIDEO'], filename), as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
