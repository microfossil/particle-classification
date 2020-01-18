from flask import Flask, request, redirect, flash, render_template
from PIL import Image
from io import BytesIO
import numpy as np
from miso.data.datasource import DataSource
from miso.save.freezing import load_from_xml
from skimage.transform import resize

app = Flask(__name__)


@app.route('/')
def index():
    return "Hello, World!"

def GET_page():
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Classification</h1>
    <p>Upload a file to have it classified</p>
    <form method=post enctype=multipart/form-data>
        <input type=file name=file>
        <input type=submit value=Upload>
    </form>
    '''

def load_image(filename, img_size):
    if img_size[2] == 3:
        im = Image.open(filename)
    else:
        im = Image.open(filename).convert('L')
    im = np.asarray(im, dtype=np.float)
    if im.ndim == 2:
        im = np.expand_dims(im, -1)
        if img_size[2] == 3:
            im = np.repeat(im, repeats=3, axis=-1)
    im = DataSource.make_image_square(im)
    im = resize(im, (img_size[0], img_size[1]), order=1)
    im = np.divide(im, 255)
    return im

@app.route('/file', methods=['GET'])
def classify_file():
    if request.method == 'GET':
        # check if the post request has the file part
        print(request.args)
        if 'filename' not in request.args:
            return GET_page()
        filename = request.args['filename']

        # im =
        # # if user does not select file, browser also
        # # submit a empty part without filename
        # if file.filename == '':
        #     flash('No selected file')
        #     return redirect(request.url)
        # if file and allowed_file(file.filename):
        #     # filename = secure_filename(file.filename)
        #     filename = file.filename
        #     im = Image.open(BytesIO(file.read())).convert('L')
        #     im = np.asarray(im, dtype=np.float)  # numpy array
        #     im = resize(im, [width, width], order=1)  # resize using linear interpolation
        #     im = np.divide(im, 255)  # divide to put in range [0 - 1]
        #     im = im[np.newaxis, :, :, np.newaxis]
        #     probs = session.run(output, feed_dict={input: im})
        #     cls = np.argmax(probs)
        #     response_format = request.args.get('format', default='html')
        #     if response_format == "raw":
        #         return labels[cls]
        #     else:
        #         return render_template("classification_result.html", image=filename, cls=cls)



if __name__ == '__main__':
    print("CLASSIFICATION SERVER")
    print("---------------------")
    session, input, output, img_size, cls_labels = load_from_xml("/Users/chaos/Desktop/pollen_20190829-150110/model/network_info.xml")

    im = load_image("/Users/chaos/Documents/Development/Bourel_pollen_pictures/fossil/aj/aj-f006_B.tif", img_size)
    result = session.run(output, feed_dict={input: im[np.newaxis,:,:,:]})

    print(cls_labels)
    print(img_size)
    print(result)
    print("{{\"code\":{}, \"score\":{}}}".format(cls_labels[np.argmax(result)], np.max(result)))

    app.run(debug=False)
