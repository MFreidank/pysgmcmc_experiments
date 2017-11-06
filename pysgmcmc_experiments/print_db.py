#!/usr/bin/python3
# -*- coding: iso-8859-15 -*-
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy_utils import database_exists, create_database

from database_schema import *


engine = create_engine("sqlite:///test.db")

assert database_exists(engine.url)

Base.metadata.bind = engine

session = sessionmaker(bind=engine, autoflush=True)()

q = session.query(Experiment)

for val in q.all():
    print(vars(val))
print()

q = session.query(TrialStatus)

for val in q.all():
    print(vars(val))

print()

q = session.query(StepsizeTrial)

for val in q.all():
    print(vars(val))

print()
