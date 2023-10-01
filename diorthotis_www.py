import multiprocessing

import pexpect
import sys
import configparser
import os
import shutil
import subprocess
import time
import hashlib
import csv
import json

from multiprocessing import Pool
from decimal import Decimal
from glob import glob


class DiorthotisSettings:

    def create_executable_name(self, source_filename):
        source_filename = source_filename.split(".")[0] + "." + self.extension
        return "".join(list(map(lambda x:x.replace("_exename_", source_filename), self.execution_command)))

    def create_compilation_command(self, source_filename, executable_filename):
        return list(map(lambda x:x.replace("_sourcename_", source_filename).replace("_exename_", executable_filename),
                        self.compilation_command))

    def __init__(self):
        config = configparser.ConfigParser()
        os.chdir("/home/alexandros/diorthotis/")

        config.read("/home/alexandros/diorthotis/diorthotis.ini")
        self.cpus = int(multiprocessing.cpu_count())-1
        self.year = config["Diorthotis"]['Year']
        self.diorthotis_path = "/home/alexandros/diorthotis/" #config["Diorthotis"]['DiorthotisPath']
        self.timeout = int(config["Diorthotis"]['Timeout'])
        self.compilation_command = config["Diorthotis"]['CompileCommands'].split(",")
        self.execution_command = config["Diorthotis"]['ExecutionCommands'].split(",")
        self.extension = config["Diorthotis"]['ExecutableExtension']
        self.grading_scale = Decimal(config["Diorthotis"]['GradingScale'])

        if config.has_option("Diorthotis", 'CPUs'):
            self.cpus = int(config["Diorthotis"]['CPUs'])

class ExerciseSettings:

    def get_weights_sum(self, weight_list):
        return sum([sum(x) for x in weight_list])

    def fill_student_data(self, student_id, grade, comment="", graded=False, exception=False):
        existing_comments = ""
        if student_id in self.studentdata["ungraded"] and "comment" in self.studentdata["ungraded"][student_id]:
            existing_comments = self.studentdata["ungraded"][student_id]["comment"]
            del self.studentdata["ungraded"][student_id]

        if graded:
            self.studentdata["graded"][student_id] = {}
            self.studentdata["graded"][student_id]["grade"] = grade
            self.studentdata["graded"][student_id]["comment"] = existing_comments + comment
        else:
            self.studentdata["ungraded"][student_id] = {}
            self.studentdata["ungraded"][student_id]["comment"] = existing_comments + comment

        if exception:
            self.manual_review.append(student_id)

    def __init__(self, settings_filename, fullpath, results_path, input_path):

        self.diorthotis = DiorthotisSettings()

        # studentdata has two keys: graded and ungraded
        self.exercise_not_found = []
        self.studentdata = {"graded": {}, "ungraded": {}}
        self.skeleton_path = ""
        self.solutions_path = ""
        self.results_path = results_path
        self.input_path = input_path
        self.copy_detection = "none"

        # self.input_exercises = []
        self.filesToCheck = []
        self.id_paths = []
        self.ids = []
        self.manual_review = []
        # self.cmd_input = []
        # self.cmd_response = []
        self.base_dir = os.getcwd()
        self.ordered = True
        self.less_functions = []
        self.copiers_list = []
        self.outfile = ""
        self.outfile_keywords = []
        self.strings_to_count = {}
        self.working_dir = "working_directory"

        config = configparser.ConfigParser()
        config.read(settings_filename)
        print(config)
        #self.fullpath = config["Settings"]['Path']
        self.fullpath = fullpath
        if config.has_option("Settings", 'CopyDetection'):
            self.copy_detection = config["Settings"]['CopyDetection']
        if config.has_option("Settings", 'Filenames'):
            self.filename = config["Settings"]['Filenames']
        if config.has_option("Settings", 'SolutionsPath'):
            self.solutions_path = config["Settings"]['SolutionsPath']
        if config.has_option("Settings", 'SolutionKeyWords'):
            self.exercise_keywords = [x.split(",") for x in
                                      config["Settings"]['SolutionKeyWords'].replace("_linechange_", "\n").split(":")]
        #if config.has_option("Settings", 'InputPath'):
        #    self.input_path = config["Settings"]['InputPath']
        if config.has_option("Settings", 'InputExercises'):
            self.input_exercise = config["Settings"]['InputExercise']
        if config.has_option("Settings", 'CMDInput'):
            self.cmd_input = [x.split(",") for x in config["Settings"]['CMDInput'].replace("_linechange_", "\n").split(":")]

        if config.has_option("Settings", 'CMDResponse'):
            self.cmd_response = [x.split(",") for x in
                                 config["Settings"]['CMDResponse'].replace("_linechange_", "\n").lower().split(":")]

            if config.has_option("Settings", 'CMDWeights'):
                self.cmd_weights = [list(map(int,x.split(","))) for x in
                                     config["Settings"]['CMDWeights'].split(":")]

            else:
                self.cmd_weights = [[1]*len(x) for x in self.cmd_response]


        if config.has_option("Settings", 'Ordered'):
            self.ordered = config["Settings"]['Ordered']
        if config.has_option("Settings", 'MinFunctions'):
            self.min_accepted_functions = config["Settings"]['MinFunctions']
        else:
            self.min_accepted_functions = 0
        if config.has_option("Settings", 'OutFile'):
            self.outfile = config["Settings"]['OutFile']
        if config.has_option("Settings", 'OutFileKeyWords'):
            self.outfile_keywords = \
                [x.split(",") for x in config["Settings"]['OutFileKeyWords'].replace("_linechange_",
                                                                                             "\n").split(":")]
            if config.has_option("Settings", 'OutFileWeights'):
                self.outfile_weights = [list(map(int,x.split(","))) for x in
                                     config["Settings"]['OutFileWeights'].split(":")]
            else:
                self.outfile_weights = [[1]*len(x) for x in self.outfile_keywords]


        if config.has_option("Settings", 'Strings'):
            temp = [x.split(",") for x in config["Settings"]['Strings'].split(":")]
            for t in temp:
                self.strings_to_count[t[0]] = t[1]

        files_to_check = self.filename.split(":")
        # for fn in files_to_check:
        #    for k in self.studentdata.keys():
        #        self.studentdata[k][fn] = []
        self.id_paths = sorted(glob(self.fullpath + '/*/'))

        for i in self.id_paths:
            src_files = os.listdir(i)
            student = i.split("/")[-2]

            if self.filename in src_files:
                self.fill_student_data(
                    graded=False,
                    student_id=student,
                    comment="",
                    grade=0,
                    exception=False
                )
            else:
                self.exercise_not_found.append(student)

        print("++++++++++++"+self.working_dir+"++++++++++++++++")
        if os.path.exists(self.working_dir) and os.path.isdir(self.working_dir):
            shutil.rmtree(self.working_dir)



def print_compilation_results(results):
    count_correct_compile = 0
    count_error_compile = 0
    count_warn_compile = 0
    count_pointer_integer_error = 0
    count_file_not_found = 0
    for item in results:
        count_correct_compile += item[1][1]
        count_error_compile += item[1][2]
        count_warn_compile += item[1][3]
        count_pointer_integer_error += item[1][4]
        count_file_not_found += item[1][5]
    print("Correct compile:" + str(count_correct_compile) + " / Warn compile:"
          + str(count_warn_compile) + " / Error compile:" + str(count_error_compile) + " / Error pointer:" + str(
        count_pointer_integer_error))


def fill_all_student_data(results, settings):
    """
    results is a tuple with 3 parts
    results[0] is the id.
    results[1] is an array with function specific output
    results[2] is an array with output relative to grading:
    graded=item[2][0], student_id=item[0], exercise=settings.filename, comment=item[2][1], grade=item[2][2]
    """
    for item in results:
        if len(item[1]) > 5 and item[1][5] == 1:  # Filling only for those who have submitted this exercise
            pass
        else:
            if len(item[2]) > 3:
                settings.fill_student_data(graded=item[2][0], student_id=item[0],
                                           comment=item[2][1], grade=item[2][2], exception=item[2][3])
            else:
                settings.fill_student_data(graded=item[2][0], student_id=item[0],
                                           comment=item[2][1], grade=item[2][2])
    return settings


def find_item_name(path, student):
    filenames = []
    success = True
    with open(path, 'r') as file:
        try:
            data = file.read().splitlines()
        except:
            try:
                data = file.read().decode('utf-8').splitlines()
            except:
                # print("unknown format - need manually:"+student)
                success = False
    if success:
        for line in data:
            if "fopen" in line:
                # print(line, student)
                pos_quote = line.find('"')
                pos_quote += 1
                line = line[pos_quote:]
                pos_quote = line.find('"')
                filename = line[:pos_quote].strip()
                filenames.append(filename)
    return filenames


def copy_input(settings):
    print("Starting copying input")
    for student in settings.studentdata["ungraded"]:  # for each student in UNGRADED exercise
        exercise_path = settings.base_dir + "/" + settings.working_dir + "/" + settings.filename + "/" + student + "/"
        #src = settings.base_dir + "/" + settings.input_path + "/" + settings.filename + "/"
        src = settings.input_path
        original_input_names = os.listdir(src)

        for item in original_input_names:
            try:
                s = os.path.join(src, item)
                d = os.path.join(exercise_path, item)
                print(s, d)
                shutil.copy(s, d)
            except:
                print("Error in copying:" + student)
                settings.manual_review.append(student)


def main(argv):
    if len(argv) == 4:
        print("Running with " + argv[0] + ", " + argv[1] + ", " + argv[2]+" and "+argv[3])
        print("------- Initializing -------")
        settings = ExerciseSettings(argv[0], argv[1], argv[2], argv[3])
    else:
        print("usage: python3 diorthotis.py <exercise_settings_file.ini>")
        exit()
    """
    settings: Settings = Settings("ds_settings_a9f3.ini")
    """
    """
    while True:
        check_copies = input("Check for copies? Y or N:")
        if check_copies == "Y" or check_copies == "N":
            break
    """

    start = time.time()  # Start measuring time

    #if check_copies == "Y":
    #    print("------- Detecting copies")
    detect_copy(settings)

    print("------- Checking keyword usages")

    results = [check_keyword_usages(settings, student) for student in get_ungraded_students_list(settings)]
    settings = fill_all_student_data(results, settings)

    print("------- Compiling")
    # This is sequential
    # results = [my_compile(settings, student) for student in get_ungraded_students_list(settings)]

    # The following three lines are for parallel
    parallel_input = [(settings, student) for student in get_ungraded_students_list(settings)]
    with Pool(settings.diorthotis.cpus) as pool:
        results = pool.map(parallel_my_compile, parallel_input)  # Compilation

    stop = time.time()  # Stop measuring time
    print(stop - start)

    print_compilation_results(results)
    settings = fill_all_student_data(results, settings)

    if len(settings.input_path) > 0:
        copy_input(settings)

    ungraded_students = get_ungraded_students_list(settings)

    print("------- Executing")
    start = time.time()  # Start measuring time
    # results = [check_keywords_with_input(settings, student) for student in ungraded_students]


    parallel_input = [(settings, index) for index in ungraded_students]
    with Pool(settings.diorthotis.cpus) as pool:
        results = pool.map(parallel_check_keywords_with_input, parallel_input)  # Running
    stop = time.time()  # Stop measuring time
    print(stop - start)

    settings = fill_all_student_data(results, settings)
    ungraded_students = get_ungraded_students_list(settings)

    insert_grades_into_database(settings)
    print("------- Writing manual review list")
    write_list_to_file(settings, '_manually')


def write_to_csv(settings, exercise, data, optional_filename_description=''):
    with open(os.path.join(settings.results_path, exercise[:-2] + optional_filename_description + '.csv'),
              mode='w') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL, escapechar='\\')
        for d in data:
            filewriter.writerow(d)


def insert_grades_into_database(settings):
    csv_data = []
    for student in settings.studentdata["graded"]:
        data = (student,
                settings.filename,
                settings.studentdata["graded"][student]["grade"],
                settings.studentdata["graded"][student]["comment"],
                settings.diorthotis.year)
        csv_data.append(list(data))
    write_to_csv(settings, settings.filename, csv_data)


def write_list_to_file(settings, optional_filename_description=''):
    with open(os.path.join(settings.results_path, settings.filename + '_manual_review' + '.csv'), mode='w') as my_file:
        my_file.write('\n'.join(settings.manual_review))
    with open(os.path.join(settings.results_path, settings.filename + '_not_submitted' + '.csv'), mode='w') as my_file:
        my_file.write('\n'.join(settings.exercise_not_found))
    with open(os.path.join(settings.results_path, settings.filename + '_copiers' + '.csv'), mode='w') as my_file:
        my_file.write('\n'.join(','.join(map(str, sl)) for sl in settings.copiers_list))


def parallel_my_compile(my_tuple):
    return my_compile(my_tuple[0], my_tuple[1])


def my_compile(settings, student):
    correct_compile = 0
    error_compile = 0
    warn_compile = 0
    pointer_integer_error = 0
    file_not_found = 0
    graded = None
    comment = ""
    grade = None

    #path = settings.diorthotis.diorthotis_path+"/"+settings.fullpath + "/" + student
    path = settings.fullpath + "/" + student
    src_files = os.listdir(path)
    if settings.filename in src_files:
        failed_tests = False
        dir_to_create = settings.working_dir + "/" + settings.filename + "/" + student + "/"
        os.umask(0)
        os.makedirs(dir_to_create, mode=0o777, exist_ok=False)
        full_file_name_source = os.path.join(path, settings.filename)
        shutil.copy(full_file_name_source, dir_to_create)
        #full_file_name_destination = os.path.join(settings.diorthotis.diorthotis_path,dir_to_create, settings.filename)
        #full_file_name_executable = os.path.join(settings.diorthotis.diorthotis_path,dir_to_create, settings.filename.replace(".c", ".exe"))
        full_file_name_destination = os.path.join(dir_to_create, settings.filename)
        full_file_name_executable = os.path.join(dir_to_create, settings.filename.replace(".c", ".exe"))

        exe_args = settings.diorthotis.create_compilation_command(full_file_name_destination, full_file_name_executable)
        output = subprocess.run(exe_args, stderr=subprocess.PIPE)

        """
        output = subprocess.run(["gcc", full_file_name_destination, "-Wno-format", "-Wno-implicit-int",
                                 "-Wno-implicit-function-declaration", "-lm", "-o", full_file_name_executable],
                                stderr=subprocess.PIPE)
        """
        result = ""
        if output.returncode == 1:  # Error compiling
            error_compile += 1
            failed_tests = True
            try:
                result = output.stderr.decode("utf-8")
            except Exception as e:
                failed_tests = True
                print("Exception!", full_file_name_source, e)
        else:
            try:
                result = output.stderr.decode("utf-8")
            except Exception as e:
                failed_tests = True
                print("Exception!", full_file_name_source, e)
            if len(result) > 0:  # Warning in compilation
                # if "between pointer and integer" in result or "from incompatible pointer type" in result:
                if "pointer" in result:
                    pointer_integer_error += 1
                    failed_tests = True
                else:
                    if ("function is dangerous" in result and result.count('\n') < 3):
                        correct_compile += 1  # Correct compilation.
                        # Just using unsafe function. count assures it is the only warning.
                    else:
                        warn_compile += 1
            else:  # Correct compilation
                correct_compile += 1

        if failed_tests == True:
            graded = True
            comment = "\n\n\n<pre> <b>*** EXERCISE " + settings.filename + " ***</b>\n\n=== COMPILATION === \nFAILED:\n" + result + "\nGRADE:0\n</pre>"
            grade = 0
            # fill_student_data(graded=True, student_id=ids[index], exercise=sf, comment="\n\n\n<pre> <b>*** EXERCISE " + sf + " ***</b>\n\n=== COMPILATION === \nFAILED:\n" + result + "\nGRADE:0\n</pre>", grade=0)
        else:
            if len(result) > 0:
                result = "\n\n\n<pre>*** <b>EXERCISE " + settings.filename + "</b> ***\n\n=== COMPILATION === \nCOMPILED, BUT WITH ISSUES:\n" + result
            else:
                result = "\n\n\n<pre>*** <b>EXERCISE " + settings.filename + "</b> ***\n\n=== COMPILATION === \nSUCCESSFUL"
            graded = False
            comment = result
            grade = -1
            # fill_student_data(graded=False, student_id=ids[index], exercise=sf, comment=result, grade=-1)
    else:
        file_not_found = 1

    return student, (
        path, correct_compile, error_compile, warn_compile, pointer_integer_error, file_not_found), (
        graded, comment, grade)


def evaluate_output(settings, desired_output, produced_output, weights):
    if settings.ordered == "True":
        return evaluate_ordered_output(desired_output, produced_output, weights)
    else:
        return evaluate_unordered_output(desired_output, produced_output, weights)


def evaluate_unordered_output(desired_output, produced_output, weights):
    # print ("Finding UNordered output")
    # print("Desired output-->",desired_output,"<--")
    # print("Produced output-->",produced_output,"<--")
    produced_output = produced_output.lower()
    counter = 0
    for index,k in enumerate(desired_output):
        position = produced_output.find(k.lower())
        if position != -1:
            # print("Matched!",k,produced_output[position-5:position+5])
            counter += weights[index]
    return counter


def evaluate_ordered_output(desired_output, produced_output, weights):
    produced_output = produced_output.lower()
    counter = 0
    found_pos = 0
    prev_pos = 0

    for index,k in enumerate(desired_output):
        prev_pos = found_pos
        found_pos = produced_output.find(k.lower(), max(found_pos, 0))
        # print(k, found_pos, produced_output[found_pos-3:found_pos+3])
        if found_pos != -1:
            counter += weights[index]
            found_pos += len(k)
        else:
            found_pos = prev_pos
    return counter


def get_ungraded_students_list(settings):
    return [key for key in settings.studentdata["ungraded"]]


def parallel_check_keywords_with_input(data):
    return check_keywords_with_input(data[0], data[1])


def check_keywords_with_input(settings, student_id):
    print("-------------- Checking keywords with interaction")

    graded = []
    exercise = settings.filename
    correct_matches = settings.get_weights_sum(settings.cmd_weights)
    #for listElem in settings.cmd_response:
    #    correct_matches += len(listElem)

    # Changes begin - this is for output in files
    # There is an output file
    if len(settings.outfile) > 0:
        #for listElem in settings.outfile_keywords:
            # Keywords to be found in files are added to correct_matches
            #correct_matches += len(listElem)
        correct_matches += settings.get_weights_sum(settings.outfile_weights)
    # Changes end - this is for output in files

    executable_path = settings.base_dir + "/" + settings.working_dir + "/" + exercise + "/" + student_id + "/"
    executable_name = settings.diorthotis.create_executable_name(exercise) # "./" + exercise.replace(".c", ".exe")
    matches = 0
    raised_exceptions = False
    print("Executing for:" + student_id)

    overall_buffer = ''

    for index, _ in enumerate(settings.cmd_input):  # for each input
        buffer_per_execution = ''
        print("=== EXECUTION. TEST #" + str(index + 1))
        overall_buffer += "\n=== EXECUTION. TEST #" + str(index + 1) + " === \n"
        # print("\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n",cmd_input[index], cmd_response[index])

        os.chdir(executable_path)
        executor = pexpect.spawn(executable_name)

        if len(settings.cmd_input[index]) > 0:
            for command_index, command in enumerate(settings.cmd_input[index]):
                # print("==============================================")
                # print("Input command:", command)
                executor.expect(['', '.* ', '.*  \r\n', pexpect.EOF])
                executor.sendline(command)
                tmp_before_buffer = executor.before.decode("latin-1")
                tmp_after_buffer = executor.after.decode("latin-1")
                overall_buffer += tmp_before_buffer
                overall_buffer += tmp_after_buffer
                buffer_per_execution += tmp_before_buffer
                buffer_per_execution += tmp_after_buffer

        final_output = executor.expect([pexpect.TIMEOUT, '.*  \r\n', pexpect.EOF], timeout=settings.diorthotis.timeout)

        if final_output == 2:
            tmp_before_buffer = executor.before.decode("latin-1")
            overall_buffer += tmp_before_buffer
            buffer_per_execution += tmp_before_buffer
        elif final_output == 1:
            tmp_before_buffer = executor.before.decode("latin-1")
            tmp_after_buffer = executor.after.decode("latin-1")
            tmp_unflushed_buffer = executor.buffer.decode("latin-1")

            overall_buffer += tmp_before_buffer
            overall_buffer += tmp_after_buffer
            overall_buffer += tmp_unflushed_buffer

            buffer_per_execution += tmp_before_buffer
            buffer_per_execution += tmp_after_buffer
            buffer_per_execution += tmp_unflushed_buffer
        else:
            print('Exception!')
            tmp_before_buffer = executor.before.decode("latin-1")
            overall_buffer += tmp_before_buffer
            buffer_per_execution += tmp_before_buffer
            raised_exceptions = True
            overall_buffer += "\n<b><u>EXECUTION TIMED OUT --> KILLED</u></b>\n"

        matches += evaluate_output(settings, settings.cmd_response[index], buffer_per_execution, settings.cmd_weights[index])

        # Changes begin
        if len(settings.outfile_keywords) > 0 and len(settings.outfile_keywords[index]) > 0:
            output_file = executable_path + '/' + settings.outfile
            with open(output_file, 'r') as f:
                output_file_content = f.read()
            matches += evaluate_output(settings, settings.outfile_keywords[index], output_file_content, settings.outfile_weights[index])
            overall_buffer += ("\n<b><u>Begin -- File Output </u></b>\n" +
                               output_file_content +
                               "\n<b><u>End -- File Output </u></b>\n")
        # Changes end

    message = None
    grade = settings.diorthotis.grading_scale * round(Decimal(matches / correct_matches), 2)
    if matches == correct_matches:
        message = "CORRECT"
    elif matches == 0:
        message = "NOT_CORRECT"
    elif matches < correct_matches:
        message = "PARTIALLY_CORRECT"
    else:
        print('Something goes very wrong. Found ', matches, ' out of ', correct_matches, ' for student ',
              student_id)

    graded = True
    comment = overall_buffer + "\n=== EXECUTION RESULT: === \n" + message + "\nGRADE:" + str(
        grade) + "\n</pre>"
    return student_id, (matches, correct_matches), (graded, comment, grade, raised_exceptions)

    # graded.append(student_id)


def detect_copy(settings):
    if settings.copy_detection=="basic":
        return basic_copy_detection(settings)
    elif settings.copy_detection=="jplag":
        return jplag_copy_detection(settings)
    elif settings.copy_detection=="none":
        return settings
    else:
        print("Unsupported method")
        return settings


def jplag_copy_detection(settings):
    os.system("java -jar "+settings.diorthotis.diorthotis_path+"jplag-4.3.0-jar-with-dependencies.jar all -l cpp -m 1.0 -t 5")
    os.system("unzip -o "+settings.diorthotis.diorthotis_path+"result.zip -d "+settings.diorthotis.diorthotis_path+"result")

    with open(settings.diorthotis.diorthotis_path+"result/overview.json", "r") as f:
        data = json.load(f)

    ungraded_students = get_ungraded_students_list(settings)
    for i in range(len(data['clusters'])):
        settings.copiers_list.append(data['clusters'][i]['members'])
        for student_id in data['clusters'][i]['members']:
            if student_id in ungraded_students:
                settings.fill_student_data(
                    graded=True,
                    student_id=student_id,
                    comment="\n\n\n<pre>*** <b>EXERCISE " + settings.filename + "</b> ***\n\n PLAGIARISM DETECTED with " +
                            str(data['clusters'][i]['members']) + "\nGRADE:0\n</pre>",
                    grade=0
                )
    return settings
def basic_copy_detection(settings):
    start = time.time()  # Stop measuring time
    ungraded_students = get_ungraded_students_list(settings)
    #results = [calculate_hash(settings, student) for student in ungraded_students]

    parallel_input = [(settings, student) for student in get_ungraded_students_list(settings)]
    with Pool(settings.diorthotis.cpus) as pool:
        results = pool.map(parallel_calculate_hash, parallel_input)  # Compilation

    stop = time.time()  # Stop measuring time
    print(stop - start)

    settings = detect_exact_copies(settings, results)

    # Copy skeleton code to respective directory
    # Run jplag for each exercise in the set
    # There is an option not to check for copying for the case that the provided

    return settings


def parallel_calculate_hash(data):
    return calculate_hash(data[0], data[1])


def calculate_hash(settings, student):
    # Calculate hash
    exercise_path = settings.fullpath + "/" + student + "/" + settings.filename
    if os.path.exists(exercise_path):
        with open(exercise_path, "rb") as f:
            # read entire file as bytes
            file_bytes = f.read()
            return student, hashlib.sha256(file_bytes).hexdigest()
    return student, None


def detect_exact_copies(settings, list_of_hashes):
    hashes = {}
    for (student, readable_hash) in list_of_hashes:

        # If hash does not exist create key
        if readable_hash not in hashes:
            hashes[readable_hash] = []

        # Append sid to dictionary
        if readable_hash is not None:
            hashes[readable_hash].append(student)

        else:
            settings.fill_student_data(
                graded=True,
                student_id=student,
                comment="\n\n\n<pre>*** <b>EXERCISE " + settings.filename + " NOT FOUND</b> *** ",
                grade=0
            )

    for key in hashes.keys():
        if len(hashes[key]) > 1:
            settings.copiers_list.append(hashes[key])
            for student_id in hashes[key]:
                settings.fill_student_data(
                    graded=True,
                    student_id=student_id,
                    comment="\n\n\n<pre>*** <b>EXERCISE " + settings.filename + "</b> ***\n\n FULL_COPY with " +
                            str(hashes[key]) + "\nGRADE:0\n</pre>",
                    grade=0
                )

    return settings


def check_keyword_usages(settings, student):
    less = {}
    keys = settings.strings_to_count.keys()
    graded = False
    comment = ""
    grade = None
    string_file = None

    # Calculate hash
    exercise_path = settings.fullpath + "/" + student + "/" + settings.filename
    with open(exercise_path, "r", encoding="ascii", errors="surrogateescape") as f:
        try:
            string_file = f.read()
        except:
            try:
                string_file = f.read().decode('utf-8')
            except:
                # print("unknown format - need manually:"+student)
                return student, str(less), (graded, comment, grade)

    for k in keys:
        occ_found = string_file.upper().count(k.upper())
        if occ_found < int(settings.strings_to_count[k]):
            less[k] = occ_found

    if len(less) > 0:
        graded = True
        comment = "\n\n\n<pre>*** <b>EXERCISE " + settings.filename + "</b> ***\n\n"+', '.join(list(less.keys()))+" NOT CORRECTLY USED\nGRADE:0\n</pre>"
        grade = 0

    return student, str(less), (graded, comment, grade)


if __name__ == "__main__":
    main(sys.argv[1:])
