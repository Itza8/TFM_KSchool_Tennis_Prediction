import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

model = pickle.load(open('objects/model.pkl', 'rb'))
df_players = pickle.load(open('objects/data.pkl', 'rb'))
shap_explainer = pickle.load(open('objects/shap_explainer.pkl', 'rb'))


def validate_player_name(player_name: str) -> bool:
    """Checking the players name for avoid error pages in Flask
    """
    return player_name not in df_players['player_name'].unique()


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if (validate_player_name(request.form['player_1'])
            | validate_player_name(request.form['player_2'])):
        return render_template(
            'index.html',
            prediction_text=f'Check the player names!'
        )

    match_to_predict = (
            df_players[df_players['player_name'] == request.form['player_1']]
            .drop(columns='player_name')
            .iloc[0]
            - df_players[df_players['player_name'] == request.form['player_2']]
            .drop(columns='player_name')
            .iloc[0]
    )

    prediction = (
        model
        .predict_proba(pd.Series(float(request.form['round_num']),
                                 index=['round_num'])
                       .append(match_to_predict)
                       .to_frame()
                       .T)
    )

    player_1_probability = prediction[0, 1]

    winner, probability = (
        (request.form["player_1"], player_1_probability)
        if player_1_probability > .5
        else (request.form["player_2"], prediction[0, 0])
    )

    example_to_explain = (pd.Series(float(request.form['round_num']),
                                    index=['round_num'])
                          .append(match_to_predict)).to_dict()

    top_features = [feature
                    for feature, _ in sorted(example_to_explain.items(),
                                             key=lambda x: x[1],
                                             reverse=True)[:3]]

    return render_template(
        'index.html',
        prediction_text=f'Winner is {winner} '
                        f'(probability of {probability * 100:.2f}%) '
                        f'and top 3 features contributions are '
                        f'{", ".join(top_features)}!'
    )


if __name__ == "__main__":
    app.run(debug=True)
