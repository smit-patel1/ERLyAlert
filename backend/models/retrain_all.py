from train_hybrid_model import train_hybrid_model

counties = ["Cabarrus", "Caldwell", "Davidson", "Durham", "Mecklenburg", "Pitt"]

for county in counties:
    print(f"\n--- Training hybrid model for {county} ---")
    result = train_hybrid_model(county, days_ahead=7)
    print(f"Returned {len(result['forecast'])} forecast rows | MAE: {result['mae']:.2f}")
