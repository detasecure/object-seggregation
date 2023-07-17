import sqlite3
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
import uvicorn
import os
from file_handler import *
from trash_classifier import *
import base64
from pydantic import BaseModel

from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta

app = FastAPI()

app.mount("/static", StaticFiles(directory="./static"), name="static")

templates = Jinja2Templates(directory="templates")

SECRET_KEY = "YOUR_SECRET_KEY"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

class TokenData(BaseModel):
    username: Optional[str] = None

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def get_user(username: str):
    conn = sqlite3.connect('predictions.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
    user = cursor.fetchone()
    conn.close()
    return user

def authenticate_user(username: str, password: str):
    user = get_user(username)
    if not user:
        return False
    if not verify_password(password, user[3]):
        return False
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(access_token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(access_token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    user = get_user(fake_db, username=token_data.username)
    if user is None:
        raise credentials_exception
    return user


@app.get("/login", response_class=HTMLResponse)
def login_form(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/token", response_model=TokenData)
def login_for_access_token(request: Request, form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        return templates.TemplateResponse("login.html", {"request": request, "message": "Invalid Username or password"})

    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user[1]}, expires_delta=access_token_expires
    )
    return templates.TemplateResponse("home.html", {"request": request, "current_user": user[1]})

@app.get("/register", response_class=HTMLResponse)
def register_form(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})

@app.post("/register")
async def register(request: Request, username: str = Form(...), email: str = Form(...), password: str = Form(...)):
    # Connect to the SQLite database
    conn = sqlite3.connect('predictions.db')
    c = conn.cursor()

    # Check if the username or email already exists
    c.execute('SELECT * FROM users WHERE username = ? OR email = ?', (username, email))
    if c.fetchone() is not None:
        return templates.TemplateResponse("register.html", {"request": request, "error_message": "Username or email already exists"})

    # Hash the password
    hashed_password = get_password_hash(password)

    # Insert a new row into the users table
    c.execute('INSERT INTO users (username, email, password) VALUES (?, ?, ?)',
              (username, email, hashed_password))

    # Commit the changes and close the connection
    conn.commit()
    conn.close()
    return templates.TemplateResponse("login.html", {"request": request, "message": "User created successfully"})









@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})
    # return templates.TemplateResponse("home.html", {"request": request})

@app.post("/")
# async def upload(request: Request, access_token: str = Form(...), file: UploadFile = File(...), current_user: str = Depends(get_current_user)):
async def upload(request: Request, current_user: str = Form(...), file: UploadFile = File(...)):
    (result, filename, image_file_content) = save_file(file.filename, file.file)
    if result:
        prediction = predict_using_restnet_newly_trained(filename)

        prediction_restnet35 = predict_using_restnet_newly_trained(filename)

        if 'predicted' in prediction_restnet35.keys():
            prediction_restnet35_response = {
                "bin_type": bin_types[prediction['predicted']],
                "label": prediction['predicted'],
                "probability_score": float(prediction['probability'])
            }

            prediction_restnet50_response = predict_using_restnet50(filename)

            # Connect to the SQLite database
            conn = sqlite3.connect('predictions.db')
            c = conn.cursor()

            c.execute('''
                INSERT INTO predictions (image, filename, prediction_restnet35, prediction_restnet50, current_user)
                VALUES (?, ?, ?, ?, ?)
            ''', (image_file_content, filename, str(prediction_restnet35_response), str(prediction_restnet50_response), current_user))

            # Save (commit) the changes
            conn.commit()

            # Close the connection
            conn.close()

            return templates.TemplateResponse("home.html",
                                              {
                                                  "request": request,
                                                  "filename": os.path.basename(filename),
                                                  "prediction_restnet35_response": prediction_restnet35_response,
                                                  "prediction_restnet50_response": prediction_restnet50_response,
                                                  "current_user": current_user
                                              })
        else:
            return templates.TemplateResponse("home.html", {"request": request,
                                                            "error_message": 'An error has occurred processing file',
                                                            "current_user": current_user
                                                            })
    else:
        return templates.TemplateResponse("home.html", {"request": request, "error_message": filename, "current_user": current_user})

@app.get("/predictions", response_class=HTMLResponse)
# async def get_predictions(request: Request, current_user: str = Depends(get_current_user)):
async def get_predictions(request: Request):
    # Connect to the SQLite database
    conn = sqlite3.connect('predictions.db')
    conn.row_factory = sqlite3.Row  # This enables column access by name: row['column_name']
    c = conn.cursor()

    # Fetch all predictions from the database
    c.execute('SELECT * FROM predictions')
    rows = c.fetchall()

    # Close the connection
    conn.close()

    # Convert rows to dictionaries and encode image data
    predictions = []
    for row in rows:
        prediction = dict(row)
        prediction['image'] = base64.b64encode(prediction['image']).decode('utf-8')
        predictions.append(prediction)

    return templates.TemplateResponse("predictions.html", {"request": request, "predictions": predictions})



if __name__ == "__main__":
    host = os.getenv("HOST")
    port = 5000
    uvicorn.run(app, host=host, port=port)
