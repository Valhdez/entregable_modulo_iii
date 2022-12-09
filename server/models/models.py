from pydantic import BaseModel

class Titanic(BaseModel):
    pclass: int;
    name: object;
    sex: object;
    age: object;
    sibsp: int;
    parch: int;
    ticket: object;
    fare: object;
    cabin: object;
    embarked: object;
    boat: object;
    body: object;
    home_dest: object;
