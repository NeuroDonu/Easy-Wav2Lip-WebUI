from flask import Flask, request, redirect, url_for, render_template
import configparser
import os
import subprocess

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        video = request.files['video']
        audio = request.files['audio']
        quality = request.form.get('quality')

        video.save(os.path.join('uploads', video.filename))
        audio.save(os.path.join('uploads', audio.filename))

        config = configparser.ConfigParser()
        config.read('config.ini')
        config.set('OPTIONS', 'video_file', os.path.join('uploads', video.filename))
        config.set('OPTIONS', 'vocal_file', os.path.join('uploads', audio.filename))
        config.set('OPTIONS', 'quality', quality)

        with open('config.ini', 'w') as configfile:
            config.write(configfile)

        subprocess.run(["python", "run.py"])

        output_file = os.path.join('outputs', video.filename)
        os.rename(os.path.join('uploads', video.filename), output_file)

        return render_template('index.html', output_file=output_file)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)