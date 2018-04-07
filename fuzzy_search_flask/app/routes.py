from app import app
from flask import render_template, flash, redirect, request, session
from app.forms import AppForm#, N_result
from app.search_app import search_app, suggestion_plot, plot_2D
from bokeh.embed import components
from bokeh.resources import CDN
from bokeh.plotting import figure
from bokeh.models.glyphs import Text



@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/app', methods=['GET', 'POST'])
def my_app():
    form = AppForm()
    Sup = ""
    session['n_out'] = 5
    if form.validate_on_submit():
        Sup = search_app(form.searchstring.data, 5)
    print(session['n_out'])
    p = suggestion_plot(form.searchstring.data)
    script, div = components(p)

    return render_template('app.html', form = form, strout = Sup, N_out = session['n_out'],
                        script=script, div=div,  bokeh_js=CDN.render_js())

@app.route('/app/add', methods=['GET', 'POST'])
def my_app_add():
    # p = suggestion_plot()
    # script, div = components(p)
    return redirect('/app')
