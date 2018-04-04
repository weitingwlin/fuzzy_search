from flask import Flask, session
from config import Config

from flask_bootstrap import Bootstrap

app = Flask(__name__)
# app.config.from_object(Config)
app.secret_key = 'You Will Never Guess'
bootstrap = Bootstrap(app)

from app import routes
