# Models

## FP-Growth
FP-Growth is an algorithm used to **identify groups of products that are frequently purchased together, based on their occurrence frequency in transaction data**. It employs an FP-tree structure to efficiently organize the data and reduce processing time. FP-Growth is suitable for discovering common associations between products but does not consider transaction value, which means it may overlook infrequent combinations with high utility.

## EFIM
EFIM is an algorithm under the HUIM category, designed to **identify product combinations with high utility in transactions, such as those generating revenue above a defined threshold**. Unlike frequency-based methods, EFIM focuses on the actual value of items by leveraging optimization techniques such as utility-lists and the Transaction-Weighted Utility (TWU) measure. This algorithm is particularly suitable for retail applications that require evaluating the economic effectiveness of product combinations in large-scale datasets.

## Genetic Algorithm
Appling Genetic Algorithm (GA) in Store Layout Optimization: Genetic Algorithm (GA) is an optimization technique inspired by the principles of natural selection, well-suited for solving complex combinatorial problems with large search spaces and multiple constraints. When applied to store layout optimization, GA helps identify the most effective arrangement of product groups to maximize business value. The process typically involves the following steps:

-	**Initialization:** Generate a set of random layout configurations.
-	**Evaluation:** Assess the performance of each layout using a fitness function based on criteria such as total revenue and product group synergy.
-	**Generation:** Produce new layout options by combining and mutating the most effective configurations.
-	**Iteration:** Repeat the process across multiple generations to gradually improve solution quality.

## Greedy and Local Search

- **Greedy Search** is a strategy that selects the best option at each current step, with the hope that locally optimal choices will lead to a globally optimal solution. However, this method does not guarantee finding the best overall solution.

- **Local Search** starts with a feasible solution and gradually improves it by exploring neighboring solutions to find better ones. This approach is suitable for problems with a very large solution space, where exploring all possible options is impractical.
