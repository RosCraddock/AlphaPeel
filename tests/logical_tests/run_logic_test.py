import os
import shutil

# test_phase code implementation

# Path to test_phase folder (and input files)
path = os.path.join("tests","logical_tests")
test_phase = os.path.join(path, "test_phase")


# Create an output folder in test_phase folder
output_path = os.path.join(test_phase, "Outputs")
if os.path.exists(output_path):
    shutil.rmtree(output_path)     # If output directory already exists, then remove.

os.mkdir(output_path)
output_name = output_path + "/output"

#Get the paths for the inputs
geno_input = test_phase + "/geno_file.txt"
ped_input = test_phase + "/ped_file.txt"

# Set up the arguments to run AlphaPeel
command = "AlphaPeel "
command += f"-genotypes {geno_input} "
command += f"-pedigree {ped_input} "
command += f"-out {output_name} "
command += "-phased_geno_prob"

# Run AlphaPeel and save outputs to output folder
os.system(command)
# Compare files with the expected
expected_output = test_phase + "/expected_phased_probs.txt"
expected_file = open(expected_output, "r")
expected_file = expected_file.read()

actual_output = test_phase + "/Outputs/output.phased_geno_prob.txt"
output_file = open(actual_output, "r")
output_file = output_file.read()

#assert whether they are equal and return result
assert expected_file == output_file


# TO DO: Adapt the above into class and def inline with the run_func_test.py:TestClass
