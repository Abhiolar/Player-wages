import os
from sklearn.externals import joblib
from flask import Flask
from flask_restful import Resource, Api, reqparse
from flask_cors import CORS
import numpy as np
import pandas as pd



app = Flask(__name__)
CORS(app)
api = Api(app)




parser = reqparse.RequestParser()


parser.add_argument("0")
parser.add_argument("1")
parser.add_argument("2")
parser.add_argument("3")
parser.add_argument("4")
parser.add_argument("5")
parser.add_argument("6")
parser.add_argument("league_Bundesliga")
parser.add_argument("league_Championship")
parser.add_argument("league_Eredivisie")
parser.add_argument("league_La Liga")
parser.add_argument("league_Liga NOS")
parser.add_argument("league_Ligue 1")
parser.add_argument("league_Premier League")
parser.add_argument("league_Serie A")
parser.add_argument("position_Defender")
parser.add_argument("position_Forward")
parser.add_argument("position_Goalkeeper")
parser.add_argument("position_Midfielder")





















if os.path.isfile("finalized_model.sav"):
    model = joblib.load("finalized_model.sav")
else:
    raise FileNotFoundError
class Predict(Resource):
    def post(self):
        args = parser.parse_args()
        
        X = (
            pd.DataFrame(
                [
                    args["0"],
                    args["1"],
                    args["2"],
                    args["3"],
                    args["4"],
                    args["5"],
                    args["6"],
                    args["league_Bundesliga"],
                    args["league_Championship"],
                    args["league_Eredivisie"],
                    args["league_La Liga"],
                    args["league_Liga NOS"],
                    args["league_Ligue 1"],
                    args["league_Premier League"],
                    args["league_Serie A"],
                    args["position_Defender"],
                    args["position_Forward"],
                    args["position_Goalkeeper"],
                    args["position_Midfielder"]
                   
                    
                   
                   
                   
                   
                    
                   
                ]
            ).astype("float")
            
        )
        _y = model.predict(X.iloc[[1]])
        return {"class": _y}
api.add_resource(Predict, "/predict")
if __name__ == "__main__":
    app.run(debug=True)
    
    
    
    
    
    
    
    
    
    
    
    