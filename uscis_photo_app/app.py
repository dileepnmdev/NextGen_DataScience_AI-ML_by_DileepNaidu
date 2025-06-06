from flask import Flask, request, render_template, send_file
from io import BytesIO
from PIL import Image, ImageOps

app = Flask(__name__)

TARGET_SIZE = (600, 600)  # 2x2 inches at 300 DPI


def process_image(file_stream):
    image = Image.open(file_stream).convert('RGB')
    image = ImageOps.exif_transpose(image)
    img_ratio = image.width / image.height
    target_ratio = TARGET_SIZE[0] / TARGET_SIZE[1]

    if img_ratio > target_ratio:
        new_height = TARGET_SIZE[1]
        new_width = int(new_height * img_ratio)
    else:
        new_width = TARGET_SIZE[0]
        new_height = int(new_width / img_ratio)

    resized = image.resize((new_width, new_height), Image.LANCZOS)
    background = Image.new('RGB', TARGET_SIZE, (255, 255, 255))
    offset = ((TARGET_SIZE[0] - new_width) // 2, (TARGET_SIZE[1] - new_height) // 2)
    background.paste(resized, offset)
    return background


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST' and 'photo' in request.files:
        processed = process_image(request.files['photo'])
        buf = BytesIO()
        processed.save(buf, format='JPEG')
        buf.seek(0)
        return send_file(buf, mimetype='image/jpeg', as_attachment=True,
                         download_name='uscis_photo.jpg')
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
