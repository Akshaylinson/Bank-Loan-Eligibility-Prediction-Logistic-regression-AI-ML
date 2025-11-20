import pandas as pd
import numpy as np
import random

# Read existing data to understand patterns
df = pd.read_csv('data/loan_data.csv')

# Analyze data ranges and patterns
income_min, income_max = df['income'].min(), df['income'].max()
credit_min, credit_max = df['credit_score'].min(), df['credit_score'].max()
loan_amount_min, loan_amount_max = df['loan_amount'].min(), df['loan_amount'].max()
loan_term_min, loan_term_max = df['loan_term'].min(), df['loan_term'].max()
age_min, age_max = df['age'].min(), df['age'].max()
debt_min, debt_max = df['existing_debt'].min(), df['existing_debt'].max()

employment_types = df['employment_type'].unique()

# Generate 100,000 new records
np.random.seed(42)
n_records = 100000

new_data = {
    'income': np.random.randint(income_min, income_max + 1, n_records),
    'credit_score': np.random.randint(credit_min, credit_max + 1, n_records),
    'employment_type': np.random.choice(employment_types, n_records),
    'loan_amount': np.random.randint(loan_amount_min, loan_amount_max + 1, n_records),
    'loan_term': np.random.randint(loan_term_min, loan_term_max + 1, n_records),
    'age': np.random.randint(age_min, age_max + 1, n_records),
    'existing_debt': np.random.randint(debt_min, debt_max + 1, n_records),
}

# Generate approval decisions based on realistic patterns
# Higher income, credit score -> higher approval probability
# Lower debt-to-income ratio -> higher approval probability
approvals = []
for i in range(n_records):
    income = new_data['income'][i]
    credit = new_data['credit_score'][i]
    debt = new_data['existing_debt'][i]
    loan_amt = new_data['loan_amount'][i]
    
    # Calculate approval probability based on financial factors
    debt_to_income = debt / income if income > 0 else 1
    loan_to_income = loan_amt / income if income > 0 else 1
    
    # Base probability from credit score (300-900 range)
    credit_prob = (credit - 300) / 600
    
    # Adjust for debt ratios
    if debt_to_income > 0.5:
        credit_prob *= 0.6
    if loan_to_income > 5:
        credit_prob *= 0.7
        
    # Add some randomness
    final_prob = max(0.1, min(0.9, credit_prob + np.random.normal(0, 0.1)))
    approvals.append(1 if np.random.random() < final_prob else 0)

new_data['approved'] = approvals

# Create new DataFrame
new_df = pd.DataFrame(new_data)

# Combine with existing data
combined_df = pd.concat([df, new_df], ignore_index=True)

# Save to new file
combined_df.to_csv('data/loan_data_extended.csv', index=False)
print(f"Generated {len(new_df)} new records")
print(f"Total records: {len(combined_df)}")
print(f"Approval rate in new data: {np.mean(approvals):.2%}")
