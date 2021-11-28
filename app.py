from flask import Flask, render_template , Response ,url_for, redirect
from video_detector import *

app = Flask(__name__)
VIDEO = VideoStreaming()


@app.route("/")
def home():
    return render_template('index.html')

@app.route("/video_feed")
def video_feed():
    return Response(VIDEO.show() ,  mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)
