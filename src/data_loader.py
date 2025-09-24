import pandas as pd

def load_airline_data():
    """Load your manually created airline quarterly data"""
    return pd.read_csv('data/processed/airline_filling_data.csv')

def load_tsa_data():
    """Load your TSA throughput data"""
    return pd.read_csv('data/processed/TSA_throughput_data_Totals.csv')

def load_all_data():
    """Load everything and return as dict"""
    return {
        'airlines': load_airline_data(),
        'tsa': load_tsa_data()
    }

