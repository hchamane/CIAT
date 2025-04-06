"""
========================================================
Cultural Impact Assessment Tool (CIAT) - Database Models
=========================================================

This module defines the database models used by the CIAT application.
It includes a base model class that provides common functionality for
all database models in the application.

Inspired by and adapted from:
    - https://github.com/levan92/logging-example
    - https://docs.sqlalchemy.org/en/20/orm/extensions/
    - https://flask-script.readthedocs.io/en/latest/
    - https://flask-sqlalchemy.palletsprojects.com/en/3.0.x/models/

Author: Hainadine Chamane
Version: 1.0.0
Date: February 2025
"""

from datetime import datetime
from . import db


class BaseModel:
    """
    Base model with common fields and methods for all database models.
    
    This abstract base class provides common attributes and methods that
    can be inherited by all database models in the application, including
    timestamps for creation and updates, as well as convenience methods
    for database operations.
    
    Attributes:
        id (int): Primary key for the model
        created_at (datetime): Timestamp when the record was created
        updated_at (datetime): Timestamp when the record was last updated
    """
    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def save(self):
        """
        Save the current model instance to the database.
        
        This method adds the current instance to the session and commits
        the changes to the database.
        """
        db.session.add(self)
        db.session.commit()

    def delete(self):
        """
        Delete the current model instance from the database.
        
        This method removes the current instance from the database and
        commits the changes.
        """
        db.session.delete(self)
        db.session.commit()

    @classmethod
    def get_by_id(cls, id):
        """
        Retrieve a model instance by its ID.
        
        Args:
            id (int): The primary key ID of the record to retrieve
            
        Returns:
            BaseModel: The model instance with the specified ID, or None if not found
        """
        return cls.query.get(id)
