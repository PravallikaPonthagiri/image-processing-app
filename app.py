import os
import cv2
import numpy as np
from flask import Flask, render_template, request, send_file
from PIL import Image
from rembg import remove

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join("static", "uploads")
OUTPUT_FOLDER = os.path.join("static", "outputs")

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/process", methods=["POST"])
def process_image():
    file = request.files["image"]
    option = request.form["option"]

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    img = cv2.imread(filepath)

    if option == "clear":
        img = cv2.detailEnhance(img, sigma_s=10, sigma_r=0.15)

    elif option == "sharpen":
        kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
        img = cv2.filter2D(img, -1, kernel)

    elif option == "edge":
        img = cv2.Canny(img, 100, 200)

    elif option == "brightness":
        img = cv2.convertScaleAbs(img, alpha=1.2, beta=30)

    elif option == "background":
        input_image = Image.open(filepath)
        output_image = remove(input_image)
        output_path = os.path.join(OUTPUT_FOLDER, "output.png")
        output_image.save(output_path)
        return send_file(output_path, as_attachment=True)
    elif option == "contrast":
     img = cv2.convertScaleAbs(img, alpha=1.5, beta=0)

    elif option == "blur":
     img = cv2.GaussianBlur(img, (15,15), 0)

    elif option == "grayscale":
     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    elif option == "rotate":
     img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    elif option == "deblur":
    # Sharpening kernel
     kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
     img = cv2.filter2D(img, -1, kernel)

    # Extra clarity
     img = cv2.convertScaleAbs(img, alpha=1.3, beta=10)

    elif option == "resize":
     img = cv2.resize(img, (500, 500))

    elif option == "crop":
     h, w = img.shape[:2]
     start_x = w // 4
     start_y = h // 4
     end_x = start_x + w // 2
     end_y = start_y + h // 2
     img = img[start_y:end_y, start_x:end_x]

    elif option == "erase":
     mask = np.zeros(img.shape[:2], np.uint8)
     mask[100:300, 100:300] = 255  # Example area
     img = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)


    elif option == "resize":
     width = int(request.form["width"])
     height = int(request.form["height"])
     img = cv2.resize(img, (width, height))

    elif option == "crop":
     x = int(request.form["x"])
     y = int(request.form["y"])
     cw = int(request.form["crop_width"])
     ch = int(request.form["crop_height"])
     img = img[y:y+ch, x:x+cw]


    elif option == "erase":
     ex = int(request.form["ex"])
     ey = int(request.form["ey"])
     ew = int(request.form["erase_width"])
     eh = int(request.form["erase_height"])
     mask = np.zeros(img.shape[:2], np.uint8)
     mask[ey:ey+eh, ex:ex+ew] = 255
     img = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)

    output_path = os.path.join(OUTPUT_FOLDER, "output.jpg")
    cv2.imwrite(output_path, img)

    return send_file(output_path, as_attachment=True)


if __name__ == "__main__":
  port = int(os.environ.get("PORT", 10000))
  app.run(host="0.0.0.0", port=port)