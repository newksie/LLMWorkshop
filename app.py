from flask import Flask, render_template, request, jsonify
from models import db, PromptSubmission
from sqlalchemy import desc
import os
from utils import CometEvaluator
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///submissions.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize the database
db.init_app(app)

with app.app_context():
    db.create_all()

# Initialize COMET evaluator once at startup
comet_evaluator = CometEvaluator()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    data = request.get_json()
    name = data.get('name', '').strip()
    source_text = data.get('source_text', '').strip()
    system_output = data.get('system_output', '').strip()
    reference_translation = data.get('reference_translation', '').strip()

    if not name or not source_text or not system_output or not reference_translation:
        return jsonify({'error': 'All fields are required.'}), 400

    try:
        # Compute COMET score
        score = comet_evaluator.evaluate(source_text, system_output, reference_translation)

        submission = PromptSubmission(
            name=name,
            source_text=source_text,
            system_output=system_output,
            reference_translation=reference_translation,
            score=score
        )
        db.session.add(submission)
        db.session.commit()

        return jsonify({'message': 'Submission successful!', 'score': score}), 200
    except Exception as e:
        return jsonify({'error': f'An error occurred during evaluation: {str(e)}'}), 500

@app.route('/leaderboard', methods=['GET'])
def leaderboard():
    top_submissions = PromptSubmission.query.order_by(desc(PromptSubmission.score)).limit(10).all()
    leaderboard_data = [
        {'name': submission.name, 'score': submission.score}
        for submission in top_submissions
    ]
    return jsonify(leaderboard_data), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))