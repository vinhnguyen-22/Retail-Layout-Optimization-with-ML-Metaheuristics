# 0. Project Overview
In modern retail, **optimizing store layout plays a vital role in enhancing the shopping experience and improving business performance.** A well-organized store layout not only helps customers easily find products but also encourages them to explore additional items, thereby boosting sales and maximizing operational efficiency.

This project leverages the power of data analytics by integrating customer transaction information with physical store layout data. Two data processing streams are executed in parallel:

- **Customer Transaction Analysis Stream:** Transaction data is processed and analyzed using the FP-Growth algorithm to extract association rules, identifying product groups that are frequently purchased together and generate high transaction value.

- **Store Layout Data Processing Stream:** The store layout design (in PowerPoint format) is converted into a CSV file containing information on product groups, coordinates, dimensions, and display areas.

The results from both streams are combined as input for two different optimization methods — **Simulated Annealing (SA) and Genetic Algorithm (GA) combined with Greedy & Local Search.** These methods generate alternative store layout configurations, which are then compared to determine the most optimal arrangement to boost sales, improve the shopping experience, and enhance operational efficiency.