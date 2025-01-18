from src.fifty_k.pipelines.prediction_pipeline import CustomClass,PredictionPipeline

from flask import Flask,render_template ,request,jsonify

app=Flask(__name__)

@app.route('/',methods=["GET","POST"])
def prediction_data():
    if request.method=="GET":
        return render_template("home.html")
    
    else:
        data= CustomClass(
            age= request.form.get("age"),
            workclass = request.form.get("workclass"),
            education_num = request.form.get("education_num"),
            marital_status = request.form.get("marital_status"),
            occupation = request.form.get("occupation"),
            relationship = request.form.get("relationship"),
            race = request.form.get("race"),
            sex = request.form.get("sex"),
            capital_gain = request.form.get("capital_gain"),
            capital_loss = request.form.get("capital_loss"),
            hours_per_week = request.form.get("hours_per_week"),
            native_country = request.form.get("native_country"),

            
        )
        
        final_data=data.get_data_as_dataframe()
        prediction_pipeline=PredictionPipeline()
        pred=prediction_pipeline.predict(final_data)
        
        result= pred
        
        if result == 0:
          return render_template("results.html", final_result = " 😭 Your Yearly Income is Less than 50k $" )

        elif result == 1:
           return render_template("results.html", final_result = " 😎 Your Yearly Income is More than 50k $" )




if __name__=='__main__':
    app.run(port=5002,debug=True)


