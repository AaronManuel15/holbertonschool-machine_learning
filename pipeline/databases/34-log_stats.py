#!/usr/bin/env python3
"""Task 34. Log stats"""
import pymongo


if __name__ == "__main__":
    client = pymongo.MongoClient('mongodb://127.0.0.1:27017')
    logs = client.logs.nginx.count_documents({})

    print(f"{logs} logs")
    print("Methods:")

    methods = ["GET", "POST", "PUT", "PATCH", "DELETE"]
    for method in methods:
        count = client.logs.nginx.count_documents({"method": method})
        print(f"\tmethod {method}: {count}")

    status_check = client.logs.nginx.count_documents({"method": "GET",
                                                      "path": "/status"})
    print(f"{status_check} status check")
