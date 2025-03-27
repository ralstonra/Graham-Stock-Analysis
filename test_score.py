import asyncio
from graham_data import calculate_graham_score_8, calculate_common_criteria, fetch_historical_data

def test_graham_score():
    score = calculate_graham_score