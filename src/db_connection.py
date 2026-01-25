from sqlalchemy import create_engine

def get_engine():
    engine = create_engine(
        "mysql+mysqlconnector://root:Ajju0323@localhost/medical_db"
    )
    return engine