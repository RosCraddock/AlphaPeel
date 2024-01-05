import os
import shutil

class TestClass:
    path = os.path.join("tests","logical_tests")
    command = "AlphaPeel"
    test_cases = None
    input_file_depend_on_test_cases= None

    files_to_input = ["geno_file", "ped_file", "penetrance", "hap_file", "seq_file"]

    files_to_check = [] #TO DO

    def mk_output_dir(self):
        """
        Prepare a empty folder at the input path
        """
        if os.path.exists(self.output_path):
            shutil.rmtree(self.output_path)

        os.mkdir(self.output_path)


