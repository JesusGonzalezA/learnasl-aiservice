import os

from flask import Flask, request, jsonify
from dotenv import load_dotenv
from werkzeug.utils import secure_filename

from model.predict import predict
from prediction import Prediction
from utils.validate_video_filename import validate_video_filename
from utils.token_required_decorator import token_required

load_dotenv()
app = Flask(__name__)


@app.route('/validate', methods=['POST'])
@token_required
def validate_video():
    if request.files:
        video = request.files['video']
        if not validate_video_filename(video.filename):
            return jsonify({
                'message': 'Invalid video or empty'
            }), 409

        filename = os.path.join(os.getenv('UPLOAD_FOLDER'), secure_filename(video.filename))
        video.save(filename)
        difficulty = request.form.get('difficulty')
        out_labels, out_probs = predict(filename, difficulty)

        dict = []
        for label, prob in zip(out_labels, out_probs):
            dict.append(Prediction(label, prob))

        return jsonify([e.serialize() for e in dict])
    else:
        return jsonify({
            'message': 'You should pass a video'
        }), 409


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port='443')
