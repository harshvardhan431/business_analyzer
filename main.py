from flask import Flask, redirect, url_for
from config import Config
from database import db, bcrypt, login_manager
import google.generativeai as genai


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    # configure Gemini globally
    genai.configure(api_key=Config.GEMINI_API_KEY)

    # init extensions
    db.init_app(app)
    bcrypt.init_app(app)
    login_manager.init_app(app)

    # register blueprints
    from auth import auth_bp
    from chat import chat_bp
    app.register_blueprint(auth_bp)
    app.register_blueprint(chat_bp)

    with app.app_context():
        db.create_all()

    @app.route("/")
    def home():
        return redirect(url_for("auth.login"))

    return app

from rag.chroma_client import get_collection
col = get_collection()
print("Total chunks:", col.count())

if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)