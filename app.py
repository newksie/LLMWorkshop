from flask import Flask, render_template, request, jsonify
from models import db, PromptSubmission
from sqlalchemy import desc
import os
from utils import BasicAPICall, AdvancedAPICall, SimilarityScore
from dotenv import load_dotenv
import requests

load_dotenv()

app = Flask(__name__)

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
    system_prompt = data.get('system_prompt', '').strip()
    system_output = data.get('system_output', '').strip()
    # 'Gold standard' english
    reference_translation = "Last weekend, I was wandering around town and, by chance, I ran into my mum - we live far from each other and don't often see each other. Anyway, she gets cold easily and I was peckish so we went to a local pub to warm up, eat an afternoon snack and have a drink. In the pub/bar, the lights were blinding and mum was annoyed that the waiter was disrespectfully casual. So we went to another bar that a friend said served fancy cocktails. But, in fact, it was a complete mess - the drinks were sickly and everyone was really drunk. In the end, after all of that, we finished our catch-up at my house with a bottle of cheap wine and some crackers."
    if not name or not system_output or not reference_translation or not system_prompt:

        return jsonify({'error': 'All fields are required.'}), 400
    
    try:
        # Compute embeddings score
        llm_output = BasicAPICall(system_prompt, system_output)
        
        score = SimilarityScore(reference_translation, llm_output)

        submission = PromptSubmission(
            name=name,
            system_prompt=system_prompt,
            system_output=system_output,
            llm_output=llm_output,
            score=score
        )
        db.session.add(submission)
        db.session.commit()

        return jsonify({'message': 'Submission successful!', 'score': score, 'llm_output': llm_output}), 200

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
    top_submissions = PromptSubmission.query.order_by(desc(PromptSubmission.score)).limit(20).all()
    leaderboard_data = [
        {
            'name': submission.name,
            'score': submission.score,
            'system_prompt': submission.system_prompt,
            'system_output': submission.system_output
        }
        for submission in top_submissions
    ]
    return jsonify(leaderboard_data), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))