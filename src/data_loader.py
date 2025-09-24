import pandas as pd
import os

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

# Add this test section (optional but helpful)
if __name__ == "__main__":
    print("Testing data loading...")
    
    try:
        data = load_all_data()
        print(f"✅ Airline data: {data['airlines'].shape}")
        print(f"✅ TSA data: {data['tsa'].shape}")
        print("All data loaded successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")

