from flask import Flask, render_template, request, jsonify
from models import db, PromptSubmission
from sqlalchemy import desc
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Configuration
DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///submissions.db')
app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URL
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize the database
db.init_app(app)

with app.app_context():
    db.create_all()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    data = request.get_json()
    name = data.get('name', '').strip()
    prompt = data.get('prompt', '').strip()

    if not name or not prompt:
        return jsonify({'error': 'Name and prompt are required.'}), 400

    score = len(prompt)  # Scoring by prompt length (number of characters)

    submission = PromptSubmission(name=name, prompt=prompt, score=score)
    db.session.add(submission)
    db.session.commit()

    return jsonify({'message': 'Submission successful!', 'score': score}), 200

@app.route('/leaderboard', methods=['GET'])
def leaderboard():
    top_submissions = PromptSubmission.query.order_by(desc(PromptSubmission.score)).limit(10).all()
    leaderboard_data = [
        {'name': submission.name, 'score': submission.score}
        for submission in top_submissions
    ]
    return jsonify(leaderboard_data), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))