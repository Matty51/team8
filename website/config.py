from os import environ, urandom

class Config(object):
    # IMPORTANT: overwrite development settings prior to deployment
    FLASK_APP = environ.get('FLASK_APP')
    FLASK_ENV = environ.get('FLASK_ENV')
    FLASK_RUN_PORT = environ.get('FLASK_RUN_PORT')
    SECRET_KEY = urandom(64)
    TESTING = False
    DEBUG = True