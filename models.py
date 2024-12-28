from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class PromptSubmission(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    prompt = db.Column(db.Text, nullable=False)
    score = db.Column(db.Integer, nullable=False)

    def __repr__(self):
        return f'<PromptSubmission {self.name}: {self.score}>'