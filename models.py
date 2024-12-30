from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class PromptSubmission(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    # source_text = db.Column(db.Text, nullable=False)
    system_output = db.Column(db.Text, nullable=False)
    reference_translation = db.Column(db.Text, nullable=False)
    score = db.Column(db.Float, nullable=False)

    def __repr__(self):
        return f'<PromptSubmission {self.name}: {self.score}>'