import os
import sys
import cv2
from pathlib import Path
from threading import Thread, Lock
import time
from pydantic import BaseModel
from queue import Queue
import queue
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List



class Singleton(type):
    _instances = {}
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._locked_call(*args, **kwargs)
        return cls._instances[cls]
    
    def _locked_call(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)

class FaceData():
    def __init__(self, id, embedding, bbox):
        self.id = id
        self.embedding = embedding
        self.bbox = bbox
    
    def __str__(self):
        return f'FaceData(id={self.id}, embedding={self.embedding}, bbox={self.bbox})'
    
    id: str = 'Unknown'
    embedding: str=None
    bbox: List[str] = None
    
class PathManager(metaclass=Singleton):
    def __init__(self):
        self.cwd = Path(__file__).parent
        self.root = self.cwd.parent
        self.model = self.root / f'data/model/'
        self.model_NN = self.model / f'NN/'
        self.embeddings = self.model / f'embedding.csv/'
        self.InsightFace_url = 'http://localhost:18081/'
        self.log_data = self.root / f'data/log/'
        self.today_frame_log = self.log_data / f'{time.strftime("%m_%d_%y", time.localtime())}'
        self.time_log = self.log_data / f'time_log_{time.strftime("%m_%d_%y", time.localtime())}.csv'
        self.info_log = self.log_data / f'personnel_info.csv'
        
    def _mkdir_if_required(self): 
        for attr in dir(self): 
            if not attr.startswith('_') and 'url' not in attr and not callable (getattr(self, attr)):
                path = getattr(self, attr) 
                if '.' not in Path (path).name: 
                    Path (path).mkdir(parents=True, exist_ok=True)
    
    def list_dir(self, dir_path):
        lst_item = [] 
        for item in dir_path.iterdir(): 
            lst_item.append(item) 
        return lst_item
    
    def walk_dir(self, dir_path):
        lst_item= [] 
        for item in dir_path.glob('**/*'): 
            lst_item.append(item)
        return lst_item
    
    def update_time_sensitive_path(self):
        self.time_log = self.time_log.parent / f'time_log_{time.strftime("%m_%d_%y", time.localtime())}.csv'
        self.today_frame_log = self.log_data / f'{time.strftime("%m_%d_%y", time.localtime())}'
        self._mkdir_if_required()
        

