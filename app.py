from flask import Flask, render_template, request, jsonify
from models import db, PromptSubmission
from sqlalchemy import desc
import os
from utils import BasicAPICall, AdvancedAPICall, SimilarityScore
from dotenv import load_dotenv
import requests

load_dotenv()

app = Flask(__name__)

# Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///submissions.db')
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
    # source_text = data.get('source_text', '').strip()
    system_output = data.get('system_output', '').strip()
    reference_translation = data.get('reference_translation', '').strip()

    if not name or not system_output or not reference_translation:
    # if not name or not source_text or not system_output or not reference_translation:

        return jsonify({'error': 'All fields are required.'}), 400
    
    try:
        # Compute embeddings score
        llm_output_text = BasicAPICall(system_output)
        
        score = SimilarityScore(reference_translation, llm_output_text)

        # score = len(reference_translation)

        submission = PromptSubmission(
            name=name,
            # source_text=source_text,
            system_output=system_output,
            reference_translation=reference_translation,
            llm_output=llm_output_text,
            score=score
        )
        db.session.add(submission)
        db.session.commit()

        return jsonify({'message': 'Submission successful!', 'score': score, 'LLM Output': llm_output_text}), 200

    except EnvironmentError as env_err:
           # Handle missing environment variables
           return jsonify({'error': str(env_err)}), 500

    except ConnectionError as conn_err:
        # Handle network-related errors
        return jsonify({'error': str(conn_err)}), 502

    except ValueError as val_err:
        # Handle invalid responses or parsing errors
        return jsonify({'error': str(val_err)}), 500

    except Exception as e:
        # Handle any other unforeseen errors
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500

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