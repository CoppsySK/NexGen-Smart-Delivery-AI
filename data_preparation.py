import pandas as pd

# ------------------------------------------
# STEP 1: Load all datasets
# ------------------------------------------
print("📥 Loading all datasets...")

orders = pd.read_csv("data/orders.csv")
delivery = pd.read_csv("data/delivery_performance.csv")
routes = pd.read_csv("data/routes_distance.csv")
vehicles = pd.read_csv("data/vehicle_fleet.csv")
warehouse = pd.read_csv("data/warehouse_inventory.csv")
feedback = pd.read_csv("data/customer_feedback.csv")
costs = pd.read_csv("data/cost_breakdown.csv")

print("✅ All 7 datasets loaded successfully!")
print()

# ------------------------------------------
# STEP 2: Standardize column names
# ------------------------------------------
def clean_cols(df):
    df.columns = df.columns.str.strip().str.replace(" ", "_").str.title()
    return df

for df in [orders, delivery, routes, vehicles, warehouse, feedback, costs]:
    clean_cols(df)

# ------------------------------------------
# STEP 3: Merge datasets
# ------------------------------------------
print("🔗 Merging datasets...")

# Merge orders + delivery
merged = pd.merge(orders, delivery, on="Order_Id", how="left")

# Merge with routes (Route_ID)
if "Route_Id" in orders.columns and "Route_Id" in routes.columns:
    merged = pd.merge(merged, routes, on="Route_Id", how="left")

# Merge with warehouse
if "Warehouse_Id" in orders.columns and "Warehouse_Id" in warehouse.columns:
    merged = pd.merge(merged, warehouse, on="Warehouse_Id", how="left")

# Merge with customer feedback (Order_ID)
if "Order_Id" in feedback.columns:
    merged = pd.merge(merged, feedback, on="Order_Id", how="left")

# Merge with cost breakdown (Order_ID)
if "Order_Id" in costs.columns:
    merged = pd.merge(merged, costs, on="Order_Id", how="left")

# Merge with vehicle info (optional if Vehicle_ID exists)
if "Vehicle_Id" in delivery.columns and "Vehicle_Id" in vehicles.columns:
    merged = pd.merge(merged, vehicles, on="Vehicle_Id", how="left")

print(f"✅ Merging complete! Final shape: {merged.shape}")
print()

# ------------------------------------------
# STEP 4: Feature Engineering
# ------------------------------------------
print("🧠 Generating business features...")

# Delivery delay (based on whichever columns exist)
if "Promised_Delivery_Times" in merged.columns and "Actual_Delivery_Times" in merged.columns:
    merged["Delay_Days"] = merged["Actual_Delivery_Times"] - merged["Promised_Delivery_Times"]
elif "Promised_Delivery_Days" in merged.columns and "Actual_Delivery_Days" in merged.columns:
    merged["Delay_Days"] = merged["Actual_Delivery_Days"] - merged["Promised_Delivery_Days"]
else:
    merged["Delay_Days"] = 0

# Binary delayed flag
merged["Is_Delayed"] = merged["Delay_Days"].apply(lambda x: 1 if x > 0 else 0)

# Total cost (sum all cost-related columns)
cost_columns = [col for col in merged.columns if "cost" in col.lower()]
if cost_columns:
    merged["Total_Cost"] = merged[cost_columns].sum(axis=1)
else:
    merged["Total_Cost"] = 0

# Customer rating category (optional)
if "Customer_Rating" in merged.columns:  # sometimes singular
    merged["Rating_Category"] = merged["Customer_Rating"].apply(
        lambda x: "High" if x >= 4 else "Medium" if x >= 2.5 else "Low"
    )
elif "Customer_Ratings" in merged.columns:  # plural version
    merged["Rating_Category"] = merged["Customer_Ratings"].apply(
        lambda x: "High" if x >= 4 else "Medium" if x >= 2.5 else "Low"
    )
else:
    merged["Rating_Category"] = "Unknown"

print("✅ Feature engineering completed!\n")

# ------------------------------------------
# STEP 5: Handle missing values
# ------------------------------------------
print("🧹 Handling missing values...")

# Fill numeric columns with mean
for col in merged.select_dtypes(include=["float64", "int64"]).columns:
    merged[col] = merged[col].fillna(merged[col].mean())

# Fill categorical columns with mode
for col in merged.select_dtypes(include=["object"]).columns:
    merged[col] = merged[col].fillna(merged[col].mode()[0])

print("✅ Missing values filled.\n")

# ------------------------------------------
# STEP 6: Save the clean merged dataset
# ------------------------------------------
merged.to_csv("data/merged.csv", index=False)
print("💾 Saved cleaned merged dataset to data/merged.csv\n")

# ------------------------------------------
# STEP 7: Quick sanity check
# ------------------------------------------
preview_cols = [c for c in ["Order_Id", "Carrier", "Priority", "Delay_Days", "Is_Delayed", "Total_Cost", "Rating_Category"] if c in merged.columns]
print("🔍 Sample preview:")
print(merged[preview_cols].head())