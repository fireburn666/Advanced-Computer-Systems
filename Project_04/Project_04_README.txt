Project 04 README

This project implements a dictionary codec for compressing data with low cardinality 
and improving search/scan operations. The program includes multi-threaded dictionary encoding 
and SIMD (Single Instruction, Multiple Data) optimizations for querying, providing a significant
speed boost compared to baseline implementations.

Usage Instructions:
1. Open Command prompt
2. Navigate to the directory containing the project files
3. Compile the code: g++ -o dictionary_encoder ACS_project_04.cpp -std=c++17 -march=native -O3
4. Run the program: dictionary_encoder.exe

1. Dictionary Encoding

- Scans raw column data to build a dictionary of unique items
- Encodes data by replacing each item with its corresponding dictionary ID
- Stores the dictionary and encoded data in a single output file

2. Query 

- Checks whether a data item exists and returns the indices of all occurrences
- Finds all unique data matching the prefix and their indices
- Utilizes SIMD for efficient querying

3. Vanilla Scan

- Vanilla column search/scan without dictionary encoding to compare performance

4. Performace Optimization

- Multi-threading: Accelerates encoding by parallelizing tasks across CPU cores.
- SIMD: Enhances query speed by processing multiple data points in parallel.


