from flask import Flask, request, flash
from PIL import Image
import numpy as np
from miso.archive.datasource import DataSource
from miso.deploy.saving import load_from_xml
from skimage.transform import resize
import sys
import pandas as pd

app = Flask(__name__)

@app.route('/')
def index():
    return '''
    <!doctype html>
    <title>MISO Server</title>
    <h1>MISO Server</h1>
    <h2>Classification</h2>
    <p>Use the image or file end points to classify an image</p>
    <p>Use the info end point to obtain the current network details</p>
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


@app.route('/count', methods=['GET'])
def counts():
    if request.method == 'GET':
        if 'format' in request.args and request.args['format'] == 'csv':
            return app_df.to_csv()
        else:
            return app_df.to_json()


@app.route('/file', methods=['GET'])
def classify_file():
    global app_df
    if request.method == 'GET':
        # check if the post request has the file part
        print(request.args)
        if 'filename' not in request.args:
            return '''
            <!doctype html>
            <title>Import file</title>
            <h1>Import and classify a particle image</h1>
            <p>Enter the full file path to import the image and classify it</p>
            <form method="GET">
                    File path:</br>
            <input type=text name=filename>
            </br>
                    Sample:</br>
            <input type=text name=sample>
            </br>
                    Index 1:</br>
            <input type=text name=index1>
            </br>
                    Index 2:</br>
            <input type=text name=index2>
            </br>
                    Resolution (pixels/mm):</br>
            <input type=text name=resolution>
                    </br>
            <input type=submit value=Upload>
            </form>'''
        else:
            filename = request.args['filename']
            sample = request.args['sample']
            index1 = request.args['index1']
            index2 = request.args['index2']
            resolution = request.args['resolution']
            print(filename)
            print(sample)
            print(index1)
            print(index2)
            print(resolution)

            if filename == "":
                flash('{"error":"No filename was entered"}')
                return
            try:
                im = load_image(filename, app_img_size)
                result = app_session.run(app_output, feed_dict={app_input: im[np.newaxis, :, :, :]})
                idx = np.argmax(result)
                code = app_cls_labels[idx]
                score = np.max(result)
                result_str = "{{\"code\":\"{}\", \"score\":{}}}".format(code, score)
                print(app_df.index.contains(sample))
                if len(app_df) == 0:
                    app_df = app_df.append(pd.Series([sample] + [0] * len(app_cls_labels), index=app_df.columns), ignore_index=True)
                    app_df.set_index('sample', inplace=True)
                elif app_df.index.contains(sample) is False:
                    app_df = app_df.append(pd.Series([0] * len(app_cls_labels), name=sample, index=app_df.columns), ignore_index=False)
                print(app_df)
                app_df.loc[sample][idx] = app_df.loc[sample][idx] + 1
                print("Image {} classified.\nCode: {}, score: {}".format(filename, code, score))
                return result_str
            except Exception as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                print("Exception on line: {}".format(exc_tb.tb_lineno))
                print(e)
                result_str = "{{\"error\":\"{}\"}}".format(e)
                return result_str


def test():
    session, input, output, img_size, cls_labels = load_from_xml(r"C:\Users\rossm\Documents\Data\TrainedNetworks\OB\OB\model\network_info.xml")

    im = load_image(r"C:\Users\rossm\Documents\Data\Foraminifera\EndlessForams\border_removed\endless_forams_20190914_165343\Beella_digitata\00026-Beella_digitata.jpg", img_size)
    result = session.run(output, feed_dict={input: im[np.newaxis, :, :, :]})

    print(cls_labels)
    print(img_size)
    print(result)
    print("{{\"code\":{}, \"score\":{}}}".format(cls_labels[np.argmax(result)], np.max(result)))
    return session, input, output, img_size, cls_labels


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Error: You must specify a network description xml.")
        # app_session, app_input, app_output, app_img_size, app_cls_labels = test()
    else:
        app_session, app_input, app_output, app_img_size, app_cls_labels = load_from_xml(sys.argv[1])
        app_df = pd.DataFrame(columns=['sample'] + app_cls_labels)
        print("CLASSIFICATION SERVER")
        print("---------------------")
        print("Labels:")
        print(app_cls_labels)
        app.run(debug=False, port=5555)


