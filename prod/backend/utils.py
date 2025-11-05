#!/usr/bin/env python3
"""
Utilities module for backend tests
"""

import logging
import sys
from typing import Any, Dict, List

def setup_logging(level=logging.INFO):
    """Setup basic logging configuration"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def validate_data_quality(data: Dict[str, Any]) -> bool:
    """Validate data quality for tests"""
    if not data:
        return False
    
    # Basic validation checks
    required_fields = ['timestamp', 'data']
    for field in required_fields:
        if field not in data:
            return False
    
    return True

def generate_data_hash(data: Any) -> str:
    """Generate hash for data integrity"""
    import hashlib
    import json
    
    if isinstance(data, dict):
        data_str = json.dumps(data, sort_keys=True)
    else:
        data_str = str(data)
    
    return hashlib.md5(data_str.encode()).hexdigest()