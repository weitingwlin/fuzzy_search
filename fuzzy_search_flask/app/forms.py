from flask_wtf import FlaskForm
from wtforms import StringField, BooleanField, SubmitField
from wtforms.validators import DataRequired

class AppForm(FlaskForm):
    searchstring = StringField('Search:',default='Apple', validators=[DataRequired()])
    fiction = BooleanField('Fiction', default=True)
    history = BooleanField('History')
    submit = SubmitField('Submit')

# class N_result(FlaskForm):
#     increment = SubmitField('increment')
