"""Database module for MESQUAL framework.

This module provides database abstractions and implementations for caching
and storing dataset results in the MESQUAL energy modeling framework.

Classes:
    Database: Abstract base class defining the database interface.
    PickleDatabase: File-based database implementation using pickle format.
    SQLDatabase: SQL-based database implementation with automatic table management.
    MongoDatabase: MongoDB-based database implementation with geospatial support.

Example:
    Basic usage with pickle database:

    >>> from mesqual.databases import PickleDatabase
    >>> db = PickleDatabase("/path/to/cache")
    >>> # Use with dataset.fetch() for caching

    SQL database usage:

    >>> from mesqual.databases import SQLDatabase
    >>> db = SQLDatabase("sqlite:///cache.db")
    >>> # Automatic table creation and management

    MongoDB database usage:

    >>> from mesqual.databases import MongoDatabase
    >>> db = MongoDatabase("mongodb://localhost:27017/mesqual_cache")
    >>> # Automatic collection creation and GeoJSON support
"""

from mesqual.databases.database import Database
from mesqual.databases.pickle_db import PickleDatabase
from mesqual.databases.sql_db import SQLDatabase
from mesqual.databases.mongo_db import MongoDatabase

__all__ = [
    'Database',
    'PickleDatabase',
    'SQLDatabase',
    'MongoDatabase'
]
