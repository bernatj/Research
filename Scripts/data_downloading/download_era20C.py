#!/usr/bin/env python
from ecmwfapi import ECMWFDataServer
server = ECMWFDataServer()

#2m temp
for i in range(1900,2010,1):
    server.retrieve({
        "class": "e2",
        "dataset": "era20c",
        "date": f"{i}-01-01/to/{i}-12-31",
        "expver": "1",
        "levtype": "sfc",
        "grid": "2.5/2.5",
        "param": "167.128",
        "stream": "oper",
        "time": "00:00:00/06:00:00/12:00:00/18:00:00",
        "type": "an",
        "format": "netcdf",
        "target": "/home/bernatj/Data/my_data/t2m-era20c-6h-"+str(i)+".nc"
        })

#mslp 
for i in range(1900,2010,1):
    server.retrieve({
        "class": "e2",
        "dataset": "era20c",
        "date": f"{i}-01-01/to/{i}-12-31",
        "expver": "1",
        "levtype": "sfc",
        "grid": "2.5/2.5",
        "param": "151.128",
        "stream": "oper",
        "time": "00:00:00/06:00:00/12:00:00/18:00:00",
        "type": "an",
        "format": "netcdf",
        "target": "/home/bernatj/Data/my_data/mslp-era20c-6h-"+str(i)+".nc"
        })