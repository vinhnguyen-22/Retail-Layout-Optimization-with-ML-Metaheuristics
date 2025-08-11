
The data processing flow and high-utility itemset mining model consist of the following steps:
## 1. Load Data
Retrieve transaction data from the database, including fields such as _BillId, CustomerId, Date, StoreId, Sku, Quantity, Sales, Cost, etc.,_ to collect comprehensive information about each shopping basket for customer behavior analysis.

## 2. Transform
Preprocess the data to reduce complexity and clean it for compatibility with the FP-Growth and HUIM models:

-	Aggregate SKUs into product groups (e.g., Level-2 categories such as Fresh Milk, Confectionery) to reduce the number of items to process.
-	Remove invalid or low-value transactions.
-	Calculate the utility (Sales) of each item (NH2 - product group level 2) per transaction.

## 3.	FP-Growth
Apply the FP-Growth algorithm to identify product group combinations that are frequently purchased together. The output includes frequent itemsets representing product groups with strong co-occurrence patterns within transactions.

## 4.	HUIM
Apply the EFIM algorithm to identify product combinations that contribute high revenue, regardless of how often they appear. Items in a combination do not need to occur frequently together, as long as the total utility exceeds a predefined threshold.
