from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired, Email, EqualTo, Length
from wtforms import IntegerField, FloatField, SelectField
from wtforms.validators import DataRequired, NumberRange

# ----------------- Register Form -----------------
class RegisterForm(FlaskForm):
    username = StringField(
        "Username",
        validators=[DataRequired(), Length(min=3, max=25)]
    )
    email = StringField(
        "Email",
        validators=[DataRequired(), Email()]
    )
    password = PasswordField(
        "Password",
        validators=[DataRequired(), Length(min=6)]
    )
    confirm_password = PasswordField(
        "Confirm Password",
        validators=[DataRequired(), EqualTo("password", message="Passwords must match.")]
    )
    submit = SubmitField("Register")


# ----------------- Login Form -----------------
class LoginForm(FlaskForm):
    email = StringField(
        "Email",
        validators=[DataRequired(), Email()]
    )
    password = PasswordField(
        "Password",
        validators=[DataRequired()]
    )
    submit = SubmitField("Login")
 
class PredictForm(FlaskForm):
    # --- Diabetes fields (PIMA order) ---
    pregnancies = IntegerField('Pregnancies', validators=[DataRequired()])
    glucose = FloatField('Glucose', validators=[DataRequired()])
    blood_pressure = FloatField('Blood Pressure', validators=[DataRequired()])
    skin_thickness = FloatField('Skin Thickness', validators=[DataRequired()])
    insulin = FloatField('Insulin', validators=[DataRequired()])
    bmi = FloatField('BMI', validators=[DataRequired()])
    dpf = FloatField('Diabetes Pedigree Function', validators=[DataRequired()])
    age = IntegerField('Age', validators=[DataRequired()])

    # --- Heart fields ---
    sex = SelectField('Sex (0 or 1)', choices=[('0','0'),('1','1')], coerce=int, validators=[DataRequired()])
    cp = SelectField('Chest pain type (0-3)', choices=[('0','0'),('1','1'),('2','2'),('3','3')], coerce=int, validators=[DataRequired()])
    trestbps = FloatField('Resting blood pressure (trestbps)', validators=[DataRequired()])
    chol = FloatField('Cholesterol', validators=[DataRequired()])
    fbs = SelectField('Fasting blood sugar > 120 mg/dl (0/1)', choices=[('0','0'),('1','1')], coerce=int, validators=[DataRequired()])
    restecg = SelectField('Resting ECG (0-2)', choices=[('0','0'),('1','1'),('2','2')], coerce=int, validators=[DataRequired()])
    thalach = FloatField('Max heart rate achieved (thalach)', validators=[DataRequired()])
    exang = SelectField('Exercise induced angina (0/1)', choices=[('0','0'),('1','1')], coerce=int, validators=[DataRequired()])
    oldpeak = FloatField('ST depression (oldpeak)', validators=[DataRequired()])
    slope = SelectField('Slope (0-2)', choices=[('0','0'),('1','1'),('2','2')], coerce=int, validators=[DataRequired()])
    ca = SelectField('CA (0-4)', choices=[('0','0'),('1','1'),('2','2'),('3','3'),('4','4')], coerce=int, validators=[DataRequired()])
    thal = SelectField('Thal (0-3)', choices=[('0','0'),('1','1'),('2','2'),('3','3')], coerce=int, validators=[DataRequired()])

    # --- Kidney fields (order used in training) ---
    bp = FloatField('Blood pressure (bp)', validators=[DataRequired()])
    sg = FloatField('Specific gravity (sg)', validators=[DataRequired()])
    al = IntegerField('Albumin (al) - integer', validators=[DataRequired()])
    su = IntegerField('Sugar (su) - integer', validators=[DataRequired()])
    # rbc/pc/pcc/ba were label-encoded in training — we accept human labels and map them
    rbc = SelectField('RBC (normal/abnormal)', choices=[('normal','normal'),('abnormal','abnormal')], validators=[DataRequired()])
    pc = SelectField('Pus cell (normal/abnormal)', choices=[('normal','normal'),('abnormal','abnormal')], validators=[DataRequired()])
    pcc = SelectField('Pus cell clumps (notpresent/present)', choices=[('notpresent','notpresent'),('present','present')], validators=[DataRequired()])
    ba = SelectField('Bacteria (notpresent/present)', choices=[('notpresent','notpresent'),('present','present')], validators=[DataRequired()])
    bgr = FloatField('Blood glucose random (bgr)', validators=[DataRequired()])
    bu = FloatField('Blood urea (bu)', validators=[DataRequired()])

    submit = SubmitField('Predict')