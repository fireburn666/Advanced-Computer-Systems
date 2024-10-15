README

Code is written in C++

Code:
Starts with all #includes
Global variables set
Multiple different functions to perform each experiment task
Argument parsing for easy repetition
Main function to perform desired calculations and output results

Download Project_02.cpp and complie using command prompt: 
g++ Project_02.cpp -mavx2 -mfma -o <output_file_name> 
where <output_file_name> can be any name that you want your file to be called when you want to run it
Afterwards, run your output_file_name or with any desired arguments afterwards

specific arguments:
size int
density float
density_A float
density_B float
threads int
simd
cache
all int

example:
g++ Project_02.cpp -mavx2 -mfma -o Project_02

Project_02 size 1000 density_A 0.05 density_B 0.5 all 12 