import argparse
from flask import Flask, request, flash
import numpy as np
from miso.data.image_utils import load_image
from miso.deploy.saving import load_from_xml
import sys
import pandas as pd
import tensorflow as tf

app = Flask(__name__)


@app.route('/')
def index():
    return '''
    <!doctype html>
    <title>MISO Classification Server</title>
    <h1>MISO Classification Server</h1>
    <h2>Classification</h2>
    <p>Use the <a href="/file">file</a> end point to classify an image</p>
    <p>Use the <a href="/count">count</a> end point to get the class counts</p>
    '''


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
        print("Classification request:")
        if 'filename' not in request.args:
            print("- no filename, serving webpage instead...")
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
            print(f" - filename:   {filename}")
            print(f" - sample:     {sample}")
            print(f" - index1:     {index1}")
            print(f" - index2:     {index2}")
            print(f" - resolution: {resolution}")

            if filename == "":
                flash('{"error":"No filename was entered"}')
                return
            try:
                # Load image
                if app_img_size[2] == 1:
                    img_type = 'rgm'
                else:
                    img_type = 'greyscale'
                im = load_image(filename, app_img_size, img_type)

                # Classify
                result = app_model(tf.convert_to_tensor(im[np.newaxis, :, :, :], dtype=tf.float32)).numpy()[0]

                # Predictions
                idx = np.argmax(result)
                code = app_cls_labels[idx]
                score = np.max(result)

                # Format response
                result_str = "{{\"code\":\"{}\", \"score\":{}}}".format(code, score)
                if len(app_df) == 0:
                    app_df = app_df.append(pd.Series([sample] + [0] * len(app_cls_labels), index=app_df.columns), ignore_index=True)
                    app_df.set_index('sample', inplace=True)
                elif app_df.index.str.contains(sample) is False:
                    app_df = app_df.append(pd.Series([0] * len(app_cls_labels), name=sample, index=app_df.columns), ignore_index=False)
                app_df.loc[sample][idx] = app_df.loc[sample][idx] + 1
                print("Results\n - code: {}\n - score: {}".format(filename, code, score))
                return result_str
            except Exception as e:
                # Erros
                exc_type, exc_obj, exc_tb = sys.exc_info()
                print("Exception on line: {}".format(exc_tb.tb_lineno))
                print(e)
                result_str = "{{\"error\":\"{}\"}}".format(e)
                return result_str


if __name__ == '__main__':
    parser = argparse.ArgumentParser("MISO Classification Server")
    parser.add_argument("-i", "--info", required=True, help="CNN network information file")
    parser.add_argument("-p", "--port", required=True, help="Server port")
    args = parser.parse_args()

    app_model, app_img_size, _, app_cls_labels = load_from_xml(args.info)
    app_df = pd.DataFrame(columns=['sample'] + app_cls_labels)
    print("MISO Classification Server - port {}".format(args.port))
    print("--------------------------")
    print("Labels:")
    print(app_cls_labels)
    app.run(debug=False, port=args.port)


