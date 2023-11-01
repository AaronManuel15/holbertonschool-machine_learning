#!/usr/bin/env python3
"""Task 30. List all documents in Python"""
import pymongo


def list_all(mongo_collection):
    """lists all documents in a collection"""

    return list(mongo_collection.find({}))
