from flask import Blueprint, render_template
from flask_socketio import emit
from . import socketio

main = Blueprint("main", __name__)

leaderboard = []  # Example: [{"user": "Alice", "score": 95}]

@main.route("/")
def index():
    return render_template("index.html", leaderboard=leaderboard)

@socketio.on("submit_prompt")
def handle_prompt(data):
    user = data["user"]
    prompt = data["prompt"]

    # Simulate scoring (replace with OpenAI API logic)
    score = len(prompt)  # Example: length of the prompt as score  ## CHANGE TO COMET
    leaderboard.append({"user": user, "score": score})
    leaderboard.sort(key=lambda x: x["score"], reverse=True)

    # Broadcast the updated leaderboard
    emit("update_leaderboard", leaderboard, broadcast=True)