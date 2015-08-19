import ast
import commands
import csv
import gzip
import logging as log
import luigi
import math
import os
import random
import re
import sys
import shutil
import string
import textwrap
import time
import datetime
import sciluigi as sl
from ConfigParser import ConfigParser

JAVA_PATH = '/sw/comp/java/x86_64/sun_jdk1.7.0_25/bin/java'

# ====================================================================================================

class JVMHelpers():
    '''Mixin with helper methods for starting and keeping alive a JVM, using jpype'''
    def start_jvm(self, jvm_path, class_path):
        import jpype
        jpype.startJVM(jvm_path,
                       "-ea",
                       "-Djava.class.path=%s" % class_path)

    def close_jvm(self):
        import jpype
        jpype.shutdownJVM()

# ====================================================================================================

class DBHelpers():
    '''Mixin with helper methods for connecting to databases'''
    def connect_mysql(self, db_host, db_name, db_user, db_passwd):
        import MySQLdb as mydb
        connection = mydb.connect(host=db_host,
                         db=db_name,
                         user=db_user,
                         passwd=db_passwd)
        return connection.cursor()

    def connect_sqlite(self, db_filename):
        import pysqlite2.dbapi2 as sqlite
        connection = sqlite.connect(db_filename)
        return connection.cursor()

# ====================================================================================================

class DependencyMetaTask(luigi.Task):
    # METHODS FOR AUTOMATING DEPENDENCY MANAGEMENT
    def get_upstream_targets(self):
        upstream_tasks = []
        for param_val in self.param_args:
            if type(param_val) is dict:
                if 'upstream' in param_val:
                    upstream_tasks.append(param_val['upstream']['task'])
        return upstream_tasks

    def requires(self):
        return self.get_upstream_targets()

    def get_input(self, input_name):
        param = self.param_kwargs[input_name]
        if type(param) is dict and 'upstream' in param:
            return param['upstream']['task'].output()[param['upstream']['port']]
        else:
            return param

    def get_value(self, input_name):
        param = self.param_kwargs[input_name]
        if type(param) is dict and 'upstream' in param:
            input_target = param['upstream']['task'].output()[param['upstream']['port']]
            if os.path.isfile(input_target.path):
                with input_target.open() as infile:
                    csv_reader = csv.reader(infile)
                    for row in csv_reader:
                        if row[0] == param['upstream']['key']:
                            return row[1]
            else:
                return 'NA'
        else:
            return param

# ====================================================================================================

class DependencyMetaTaskExternal(luigi.ExternalTask):
    # METHODS FOR AUTOMATING DEPENDENCY MANAGEMENT
    def requires(self):
        upstream_tasks = []
        for param_val in self.param_args:
            if 'upstream' in param_val:
                upstream_tasks.append(param_val['upstream']['task'])
        return upstream_tasks

    def get_input(self, input_name):
        param = self.param_kwargs[input_name]
        if type(param) is dict and 'upstream' in param:
            return param['upstream']['task'].output()[param['upstream']['port']]
        else:
            return param

# ====================================================================================================

class DatasetNameMixin():
    dataset_name = luigi.Parameter() # This is used here, so must be included

# ====================================================================================================

class TaskHelpers():
    '''Mixin with various convenience methods that most tasks need, such as for executing SLURM commands'''
    accounted_project = luigi.Parameter()

    # Main Execution methods
    def execute_in_configured_mode(self, command):
        '''Execute either locally or via SLURM, depending on config'''

        if self.get_task_config("runmode") == "local":
            self.execute_command(command)

        elif self.get_task_config("runmode") == "nodejob":
            train_size="NA"
            if hasattr(self,'train_size'):
                train_size = self.train_size
            replicate_id="NA"
            if hasattr(self,'replicate_id'):
                replicate_id = self.replicate_id

            self.execute_hpcjob(command,
                    accounted_project = self.accounted_project,
                    time_limit = self.get_task_config("time_limit"),
                    partition  = self.get_task_config("partition"),
                    cores      = self.get_task_config("cores"),
                    jobname    = "".join([train_size,
                                          replicate_id,
                                          self.dataset_name,
                                          self.task_family]),
                    threads    = self.get_task_config("threads"))

        elif self.get_task_config("runmode") == "mpijob":
            self.execute_mpijob(command,
                    accounted_project = self.accounted_project,
                    time_limit = self.get_task_config("time_limit"),
                    partition  = self.get_task_config("partition"),
                    cores      = self.get_task_config("cores"),
                    jobname    = "".join([self.train_size,
                                          self.replicate_id,
                                          self.dataset_name,
                                          self.task_family]))

    def execute_command(self, command):

        if isinstance(command, list):
            command = " ".join(command)

        log.info("Executing command: " + str(command))
        (status, output) = commands.getstatusoutput(command)
        log.info("STATUS: " + str(status))
        log.info("OUTPUT: " + "; ".join(str(output).split("\n")))
        if status != 0:
            log.error("Command failed: {cmd}".format(cmd=command))
            log.error("OUTPUT OF FAILED COMMAND: " + "; \n".join(str(output).split("\n")))
            raise Exception("Command failed: {cmd}\nOutput:\n{output}".format(cmd=command, output=output))
        return (status, output)

    def execute_hpcjob(self, command, accounted_project, time_limit="4:00:00", partition="node", cores=16, jobname="LuigiNodeJob", threads=16):

        slurm_part = "salloc -A {pr} -p {pt} -n {c} -t {t} -J {m} srun -n 1 -c {thr} ".format(
                pr  = accounted_project,
                pt  = partition,
                c   = cores,
                t   = time_limit,
                m   = jobname,
                thr = threads)

        if isinstance(command, list):
            command = " ".join(command)

        (status, output) = self.execute_command(slurm_part + command)
        self.log_slurm_info(output)

        return (status, output)

    def execute_mpijob(self, command, accounted_project, time_limit="4-00:00:00", partition="node", cores=32, jobname="LuigiMPIJob", cores_per_node=16):

        slurm_part = "salloc -A {pr} -p {pt} -n {c} -t {t} -J {m} mpirun -v -np {c} ".format(
                pr = accounted_project,
                pt = partition,
                c  = cores,
                t  = time_limit,
                m  = jobname)

        if isinstance(command, list):
            command = " ".join(command)

        (status, output) = self.execute_command(slurm_part + command)
        self.log_slurm_info(output)

        return (status, output)

    def execute_locally(self, command):
        '''Execute locally only'''
        return self.execute_command(command)

    def x(self, command):
        '''A short-hand alias around the execute_in_configured_mode method'''
        return self.execute_in_configured_mode(command)

    def lx(self, command):
        '''Short-hand alias around the execute_locally method'''
        return self.execute_locally(command)


    # VARIOUS CONVENIENCE METHODS
    def assert_matches_character_class(self, char_class, a_string):
        if not bool(re.match("^{c}+$".format(c=char_class), a_string)):
            raise Exception("String {s} does not match character class {cc}".format(s=a_string, cc=char_class))

    def clean_filename(self, filename):
        return re.sub("[^A-Za-z0-9\_\ ]", '_', str(filename)).replace(' ', '_')

    def get_task_config(self, name):
        return luigi.configuration.get_config().get(self.task_family, name)

    def log_slurm_info(self, command_output):
        matches = re.search('[0-9]+', command_output)
        if matches:
            jobid = matches.group(0)
            with open(self.auditlog_file, 'a') as alog:
                # Write jobid to audit log
                tsv_writer = csv.writer(alog, delimiter='\t')
                tsv_writer.writerow(['slurm_jobid', jobid])
                # Write slurm execution time to audit log
                (jobinfo_status, jobinfo_output) = self.execute_command('/usr/bin/sacct -j {jobid} --noheader --format=elapsed'.format(jobid=jobid))
                last_line = jobinfo_output.split('\n')[-1]
                sacct_matches = re.search('([0-9\:\-]+)',last_line)
                if sacct_matches:
                    slurm_exectime_fmted = sacct_matches.group(1)
                    # Date format needs to be handled differently if the days field is included
                    if '-' in slurm_exectime_fmted:
                        t = time.strptime(slurm_exectime_fmted, '%d-%H:%M:%S')
                        self.slurm_exectime_sec = int(datetime.timedelta(t.tm_mday, t.tm_sec, 0, 0, t.tm_min, t.tm_hour).total_seconds())
                    else:
                        t = time.strptime(slurm_exectime_fmted, '%H:%M:%S')
                        self.slurm_exectime_sec = int(datetime.timedelta(0, t.tm_sec, 0, 0, t.tm_min, t.tm_hour).total_seconds())
                    tsv_writer.writerow(['slurm_exectime_sec', int(self.slurm_exectime_sec)])

# ====================================================================================================

class ExistingSmiles(sl.Task):
    '''External task for getting hand on existing smiles files'''

    # PARAMETERS
    dataset_name = luigi.Parameter()
    replicate_id = luigi.Parameter()

    # TARGETS
    def out_smiles(self):
        datapath = os.path.abspath('./data')
        filename = self.dataset_name + '.smiles'
        outfile_path = os.path.join(datapath, filename)
        smilestgt = sl.TargetInfo(self, outfile_path)
        return smilestgt

# ====================================================================================================

class Concatenate2Files(DependencyMetaTask, TaskHelpers):

    # INPUT TARGETS
    file1_target = luigi.Parameter()
    file2_target = luigi.Parameter()

    # TASK PARAMETERS
    skip_file1_header = luigi.BooleanParameter(default=True)
    skip_file2_header = luigi.BooleanParameter(default=True)

    def output(self):
        return { 'concatenated_file' : luigi.LocalTarget(self.get_input('file1_target').path + '.concat') }

    def run(self):
        inpath1 = self.get_input('file1_target').path
        inpath2 = self.get_input('file2_target').path
        outpath = self.output()['concatenated_file'].path
        with open(inpath1) as infile1, open(inpath2) as infile2, open(outpath, 'w') as outfile:
            # Write file 1, with or without header
            i = 1
            for line in infile1:
                if not (i == 1 and self.skip_file1_header):
                    outfile.write(line)
                i += 1

            # Write file 2, with or without header
            j = 1
            for line in infile2:
                if not (j == 1 and self.skip_file2_header):
                    outfile.write(line)
                j += 1

# ====================================================================================================

class GenerateSignaturesFilterSubstances(DependencyMetaTask, TaskHelpers, DatasetNameMixin):

    # INPUT TARGETS
    smiles_target = luigi.Parameter()

    # TASK PARAMETERS
    replicate_id = luigi.Parameter()
    min_height = luigi.Parameter()
    max_height = luigi.Parameter()

    # DEFINE OUTPUTS
    def output(self):
        return { 'signatures' : luigi.LocalTarget(self.get_input('smiles_target').path + '.h%d_%d.sign' % (self.min_height, self.max_height)) }

    # WHAT THE TASK DOES
    def run(self):
        self.x([JAVA_PATH, '-jar jars/GenerateSignatures.jar -inputfile', self.get_input('smiles_target').path,
                '-threads', self.get_task_config('threads'),
                '-minheight', str(self.min_height),
                '-maxheight', str(self.max_height),
                '-outputfile', self.output()['signatures'].path,
                '-silent'])
        self.lx(['touch', self.output()['signatures'].path])

    '''
    -inputfile <file [inputfile.smiles]>   filename for input SMILES file
    -limit <integer>                       Number of lines to read. Handy for
                                           testing.
    -maxatoms <integer>                    Maximum number of non-hydrogen
                                           atoms. [default: 50]
    -maxheight <integer>                   Maximum signature height.
                                           [default: 3]
    -minatoms <integer>                    Minimum number of non-hydrogen
                                           atoms. [default: 5]
    -minheight <integer>                   Minimum signature height.
                                           [default: 1]
    -outputfile <file [output.sign]>       filename for generated output file
    -silent                                Do not output anything during run
    -threads <integer>                     Number of threads. [default:
    '''

# ====================================================================================================

class CreateUniqueSignaturesCopy(DependencyMetaTask, TaskHelpers, DatasetNameMixin, AuditTrailMixin):

    # INPUT TARGETS
    signatures_target = luigi.Parameter()

    # TASK PARAMETERS
    replicate_id = luigi.Parameter()

    # DEFINE OUTPUTS
    def output(self):
        replicate_id = self.replicate_id
        base_dir = 'data/' + replicate_id + '/'
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        local_part = self.get_input('signatures_target').path.split('/')[-1]
        return { 'signatures' :
                    luigi.LocalTarget(base_dir + local_part) }

    # EXECUTE
    def run(self):
        if self.replicate_id is not None:
            self.x(["cp",
                    self.get_input('signatures_target').path,
                    self.output()['signatures'].path])

# ====================================================================================================

class SampleTrainAndTest(DependencyMetaTask, TaskHelpers, DatasetNameMixin):

    # INPUT TARGETS
    signatures_target = luigi.Parameter()

    # TASK PARAMETERS
    seed = luigi.Parameter(default=None)
    test_size = luigi.Parameter()
    train_size = luigi.Parameter()
    sampling_method = luigi.Parameter()
    replicate_id = luigi.Parameter()

    # DEFINE OUTPUTS
    def output(self):
        base_str = self.get_input('signatures_target').path + ".{test}_{train}_{method}".format(
            test  = self.test_size.replace("%", "proc"),
            train = self.train_size,
            method = self.sampling_method
        )

        return { "train_dataset" : luigi.LocalTarget(base_str + "_train"),
                 "test_dataset"  : luigi.LocalTarget(base_str + "_test"),
                 "log" : luigi.LocalTarget(base_str + "_train.log") }

    # WHAT THE TASK DOES
    def run(self):
        test_temp_path  = self.output()['test_dataset'].path  + '.tmp'
        train_temp_path = self.output()['train_dataset'].path + '.tmp'

        jar_files = { 'random'          : 'SampleTrainingAndTest.jar',
                      'signature_count' : 'SampleTrainingAndTestSizedBased.jar' }
        jar_file = jar_files[self.sampling_method]

        cmd = [JAVA_PATH, "-jar jars/" + jar_file,
                     "-inputfile", self.get_input('signatures_target').path,
                     "-testfile", test_temp_path,
                     "-trainingfile", train_temp_path,
                     "-testsize", self.test_size,
                     "-trainingsize", self.train_size,
                     "-silent"]
        if self.seed != None:
            cmd.extend(["-seed", self.seed])

        self.x(cmd)

        # Restore temporary test and train files to their original file names
        self.lx(["mv",
                test_temp_path,
                self.output()['test_dataset'].path])
        self.lx(["mv",
                train_temp_path,
                self.output()['train_dataset'].path])
        self.lx(["mv",
                self.output()['train_dataset'].path + '.tmp.log',
                self.output()['train_dataset'].path + '.log'])

# ====================================================================================================

class CreateSparseTrainDataset(DependencyMetaTask, TaskHelpers, DatasetNameMixin):

    # INPUT TARGETS
    train_dataset_target = luigi.Parameter()

    # TASK PARAMETERS
    replicate_id = luigi.Parameter()

    # DEFINE OUTPUTS
    def output(self):
        basepath = self.get_input('train_dataset_target').path
        return { "sparse_train_dataset" : luigi.LocalTarget(basepath + ".csr"),
                 "signatures" : luigi.LocalTarget(basepath + ".signatures"),
                 "log" : luigi.LocalTarget(basepath + ".csr.log") }

    # WHAT THE TASK DOES
    def run(self):
        self.x([JAVA_PATH, "-jar jars/CreateSparseDataset.jar",
                "-inputfile", self.get_input('train_dataset_target').path,
                "-datasetfile", self.output()['sparse_train_dataset'].path,
                "-signaturesoutfile", self.output()["signatures"].path,
                "-silent"])

# ====================================================================================================

class CreateSparseTestDataset(DependencyMetaTask, TaskHelpers, DatasetNameMixin):

    # INPUT TARGETS
    test_dataset_target = luigi.Parameter()
    signatures_target = luigi.Parameter()

    # TASK PARAMETERS
    replicate_id = luigi.Parameter()

    # DEFINE OUTPUTS
    def output(self):
        basepath = self.get_input('test_dataset_target').path
        return {"sparse_test_dataset" : luigi.LocalTarget(basepath + ".csr"),
                "signatures" : luigi.LocalTarget(basepath + ".signatures"),
                "log" : luigi.LocalTarget(basepath + ".csr.log") }

    # WHAT THE TASK DOES
    def run(self):
        self.x([JAVA_PATH, "-jar jars/CreateSparseDataset.jar",
                "-inputfile", self.get_input('test_dataset_target').path,
                "-signaturesinfile", self.get_input('signatures_target').path,
                "-datasetfile", self.output()["sparse_test_dataset"].path,
                "-signaturesoutfile", self.output()["signatures"].path,
                "-silent"])

# ====================================================================================================

class TrainSVMModel(DependencyMetaTask, TaskHelpers, DatasetNameMixin):

    # INPUT TARGETS
    train_dataset_target = luigi.Parameter()

    # TASK PARAMETERS
    replicate_id = luigi.Parameter()
    train_size = luigi.Parameter()
    svm_gamma = luigi.Parameter()
    svm_cost = luigi.Parameter()
    svm_type = luigi.Parameter()
    svm_kernel_type = luigi.Parameter()

    # Whether to run svm-train or pisvm-train when training
    train_dataset_gzipped = luigi.BooleanParameter(default=True)
    parallel_train = luigi.BooleanParameter()

    # DEFINE OUTPUTS
    def output(self):
        return { 'svm_model' : luigi.LocalTarget(self.get_input('train_dataset_target').path + ".g{g}_c{c}_s{s}_t{t}.svm".format(
                    g = self.svm_gamma.replace(".", "p"),
                    c = self.svm_cost,
                    s = self.svm_type,
                    t = self.svm_kernel_type)) }

    # WHAT THE TASK DOES
    def run(self):
        '''
        Determine pisvm parameters based on training set size
        Details from Ola and Marcus:

        size         o    q
        -------------------
        <1k:       100  100
        1k-5k:     512  256
        5k-40k    1024 1024
        >40k      2048 2048
        '''

        train_size = self.train_size
        if train_size == 'rest':
            o = 2048
            q = 2048
        else:
            trainsize_num = int(train_size)
            if trainsize_num < 100:
                o = 10
                q = 10
            elif 100 <= trainsize_num < 1000:
                o = 100
                q = 100
            elif 1000 <= trainsize_num < 5000:
                o = 512
                q = 256
            elif 5000 <= trainsize_num < 40000:
                o = 1024
                q = 1024
            elif 40000 <= trainsize_num:
                o = 2048
                q = 2048
            else:
                raise Exception("Trainingsize {s} is not 'rest' nor a valid positive number!".format(s = trainsize_num))

        # Set some file paths
        trainfile = self.get_input('train_dataset_target').path
        trainfile_gunzipped = trainfile + ".ungzipped"
        svmmodel_file = self.output()['svm_model'].path

        if self.train_dataset_gzipped:
            # Unpack the train data file
            self.lx(["gunzip -c",
                     trainfile,
                     ">",
                     trainfile_gunzipped])

            # We want to use the same variable for both the gzipped
            # and ungzipped case, so that the svm-train command will
            # be the same for both, below
            trainfile = trainfile_gunzipped

        # Select train command based on parameter
        if self.parallel_train:
            self.x(['pisvm-train',
                    "-o", str(o),
                    "-q", str(q),
                    "-s", self.svm_type,
                    "-t", self.svm_kernel_type,
                    "-g", self.svm_gamma,
                    "-c", self.svm_cost,
                    "-m", "2000",
                    trainfile,
                    svmmodel_file,
                    ">",
                    "/dev/null"]) # Needed, since there is no quiet mode in pisvm :/
        else:
            self.x(['svm-train',
                "-s", self.svm_type,
                "-t", self.svm_kernel_type,
                "-g", self.svm_gamma,
                "-c", self.svm_cost,
                "-m", "2000",
                "-q", # quiet mode
                trainfile,
                svmmodel_file])

# ====================================================================================================

class TrainLinearModel(DependencyMetaTask, TaskHelpers, DatasetNameMixin):

    # INPUT TARGETS
    train_dataset_target = luigi.Parameter()

    # TASK PARAMETERS
    replicate_id = luigi.Parameter()
    train_size = luigi.Parameter()
    lin_type = luigi.Parameter() # 0 (regression)
    lin_cost = luigi.Parameter() # 100
    # Let's wait with implementing these
    #lin_epsilon = luigi.Parameter()
    #lin_bias = luigi.Parameter()
    #lin_weight = luigi.Parameter()
    #lin_folds = luigi.Parameter()

    # Whether to run normal or distributed lib linear
    train_dataset_gzipped = luigi.BooleanParameter(default=True)
    #parallel_train = luigi.BooleanParameter()

    # DEFINE OUTPUTS
    def output(self):
        return { 'lin_model' : luigi.LocalTarget(self.get_input('train_dataset_target').path + ".s{s}_c{c}.linmodel".format(
                    s = self.lin_type,
                    c = self.lin_cost)) }

    # WHAT THE TASK DOES
    def run(self):
        # Set some file paths
        trainfile = self.get_input('train_dataset_target').path
        trainfile_gunzipped = trainfile + ".ungzipped"
        self.output()['lin_model'].path

        if self.train_dataset_gzipped:
            # Unpack the train data file
            self.lx(["gunzip -c",
                     trainfile,
                     ">",
                     trainfile_gunzipped])

            # We want to use the same variable for both the gzipped
            # and ungzipped case, so that the lin-train command will
            # be the same for both, below
            trainfile = trainfile_gunzipped

        #self.x(['distlin-train',
        self.x(['lin-train',
            "-s", self.lin_type,
            "-c", self.lin_cost,
            "-q", # quiet mode
            trainfile,
            self.output()['lin_model'].path])

# ====================================================================================================

class PredictSVMModel(DependencyMetaTask, TaskHelpers, DatasetNameMixin):
    # INPUT TARGETS
    svmmodel_target = luigi.Parameter()
    sparse_test_dataset_target = luigi.Parameter()
    replicate_id = luigi.Parameter()

    # TASK PARAMETERS
    test_dataset_gzipped = luigi.BooleanParameter(default=True)

    # DEFINE OUTPUTS
    def output(self):
        basepath = self.get_input('svmmodel_target').path
        return {'prediction': luigi.LocalTarget(basepath + ".prediction")}

    # WHAT THE TASK DOES
    def run(self):
        if self.test_dataset_gzipped:
            # Set some file paths
            test_dataset_path = self.get_input('sparse_test_dataset_target').path + ".ungzipped"

            # Un-gzip the csr file
            self.lx(["gunzip -c",
                     self.get_input('sparse_test_dataset_target').path,
                     ">",
                     test_dataset_path])
        else:
            test_dataset_path = self.get_input('sparse_test_dataset_target').path

        # Run prediction
        self.x(["/proj/b2013262/nobackup/src/libsvm-3.17/svm-predict",
                test_dataset_path,
                self.get_input('svmmodel_target').path,
                self.output()["prediction"].path])

# ====================================================================================================

class PredictLinearModel(DependencyMetaTask, TaskHelpers, DatasetNameMixin):
    # INPUT TARGETS
    linmodel_target = luigi.Parameter()
    sparse_test_dataset_target = luigi.Parameter()
    replicate_id = luigi.Parameter()

    # TASK PARAMETERS
    test_dataset_gzipped = luigi.BooleanParameter(default=True)

    # DEFINE OUTPUTS
    def output(self):
        basepath = self.get_input('linmodel_target').path
        return {'prediction': luigi.LocalTarget(basepath + ".prediction")}

    # WHAT THE TASK DOES
    def run(self):
        if self.test_dataset_gzipped:
            # Set some file paths
            test_dataset_path = self.get_input('sparse_test_dataset_target').path + ".ungzipped"

            # Un-gzip the csr file
            self.lx(["gunzip -c",
                     self.get_input('sparse_test_dataset_target').path,
                     ">",
                     test_dataset_path])
        else:
            test_dataset_path = self.get_input('sparse_test_dataset_target').path

        # Run prediction
        #self.x(["/proj/b2013262/nobackup/opt/mpi-liblinear-1.94/predict",
        self.x(["/proj/b2013262/nobackup/workflows/workflows/bin/lin-predict",
                test_dataset_path,
                self.get_input('linmodel_target').path,
                self.output()["prediction"].path])

# ====================================================================================================

class AssessSVMRegression(DependencyMetaTask, TaskHelpers, DatasetNameMixin):

    # INPUT TARGETS
    svmmodel_target = luigi.Parameter()
    sparse_test_dataset_target = luigi.Parameter()
    prediction_target = luigi.Parameter()

    # TASK PARAMETERS
    test_dataset_gzipped = luigi.BooleanParameter(default=True)
    replicate_id = luigi.Parameter()

    # DEFINE OUTPUTS
    def output(self):
        basepath = self.get_input('svmmodel_target').path
        return {'plot' : luigi.LocalTarget(basepath + ".prediction.png"),
                'log' : luigi.LocalTarget(basepath + ".prediction.log") }

    # WHAT THE TASK DOES
    def run(self):
        # Run Assess
        self.x(["/usr/bin/xvfb-run /sw/apps/R/x86_64/3.0.2/bin/Rscript assess/assess.r",
                "-p", self.get_input('prediction_target').path,
                "-t", self.get_input('sparse_test_dataset_target').path])

# ====================================================================================================

class CreateReport(DependencyMetaTask, TaskHelpers, DatasetNameMixin):

    # INPUT TARGETS
    signatures_target = luigi.Parameter()
    sample_traintest_log_target = luigi.Parameter()
    sparse_testdataset_log_target = luigi.Parameter()
    sparse_traindataset_log_target = luigi.Parameter()
    train_dataset_target = luigi.Parameter()
    svmmodel_target = luigi.Parameter()
    assess_svm_log_target = luigi.Parameter()
    assess_svm_plot_target = luigi.Parameter()

    # TASK PARAMETERS
    dataset_name = luigi.Parameter()
    train_size = luigi.Parameter()
    test_size = luigi.Parameter()
    svm_cost = luigi.Parameter()
    svm_gamma = luigi.Parameter()
    replicate_id = luigi.Parameter()
    accounted_project = luigi.Parameter()

    # DEFINE OUTPUTS
    def output(self):
        basepath = self.get_input('assess_svm_log_target').path
        return { 'html_report' : luigi.LocalTarget(basepath + ".report.html") }


    def get_svm_rmsd(self):
        # Figure out svm_rmsd
        with self.get_input('assess_svm_log_target').open() as infile:
            csv_reader = csv.reader(infile)
            for row in csv_reader:
                if row[0] == 'RMSD':
                    svm_rmsd = float(row[1])
        return svm_rmsd


    def get_html_report_content_old(self):
        '''Create and return the content of the HTML report'''
        output_html = "<html><body style=\"font-family: arial, helvetica, sans-serif;\"><h1>Report for dataset {dataset}</h1>\n".format(
                dataset=self.dataset_name)

        # Get hand on some log files that we need to create the HTML report
        log_files = {}
        log_files['Sample_Train_and_Test'] = self.get_input('sample_traintest_log_target').path
        log_files['Create_Sparse_Test_Dataset'] = self.get_input('sparse_testdataset_log_target').path
        log_files['Create_Sparse_Train_Dataset'] = self.get_input('sparse_traindataset_log_target').path
        log_files['Predict_SVM'] = self.get_input('assess_svm_log_target').path

        for name, path in log_files.iteritems():
            output_html += self.tag('h2', name.replace('_',' '))
            output_html += "<ul>\n"
            with open(path, "r") as infile:
                tsv_reader = csv.reader(infile, delimiter=",")
                for row in tsv_reader:
                    output_html += "<li><strong>{k}:</strong> {v}</li>\n".format(
                            k=row[0],
                            v=row[1])
            output_html += "</ul>\n"
        output_html += "<ul>\n"
        output_html += "</body></html>"
        return output_html

    def get_html_report_content(self):
        log_files = {}
        log_files['sample_train_and_test'] = self.get_input('sample_traintest_log_target').path
        log_files['create_sparse_test_dataset'] = self.get_input('sparse_testdataset_log_target').path
        log_files['create_sparse_train_dataset'] = self.get_input('sparse_traindataset_log_target').path
        log_files['predict_svm'] = self.get_input('assess_svm_log_target').path

        # Initialize an empty dict where to store information to show in the report
        report_info = {}

        # Loop over the log files to gather info in to dict structure
        for logfile_name, path in log_files.iteritems():
            report_info[logfile_name] = {}
            with open(path, "r") as infile:
                tsv_reader = csv.reader(infile, delimiter=",")
                for key, val in tsv_reader:
                    report_info[logfile_name][key] = val

        html_head = textwrap.dedent('''
            <!DOCTYPE html PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN">
            <html>
            <head>
              <title>Luigi HTML report / help file</title>
              <style type="text/css">
                ol{margin:0;padding:0}
                body{font-family:"Arial"}
                li{font-size:11pt;}
                p{font-size:11pt;margin:0;}
                h1{font-size:16pt;}
                h2{font-size:13pt;}
                h3{font-size:12pt;}
                h4{font-size:11pt;}
                h5{font-size:11pt;}
                h6{font-style:italic;font-size:11pt;}
                th{background: #efefef;padding: 4px 12px;}
                td{padding: 2px 12px;}
              </style>
            </head>''').strip()
        html_content = textwrap.dedent('''
            <body>
              <h1>QSAR model for $foo</h1>
              <p>The $foo model is a QSAR model for predicting $foo based on a set of
              substances with that property extracted from the ChEMBL database.</p>
              <table cellpadding="0" cellspacing="0">
                <tbody>
                  <tr>
                    <th colspan="2">Dataset</th>
                  </tr>
                  <tr>
                    <td>Training set size</td>
                    <td>{train_size}</td>
                  </tr>
                  <tr>
                    <td>Test set size</td>
                    <td>{test_size}</td>
                  </tr>
                  <tr>
                    <td>Minimum number of non hydrogen atoms</td>
                    <td>{min_nonh_atoms}</td>
                  </tr>
                  <tr>
                    <td>Maximum number of non hydrogen atoms</td>
                    <td>{max_nonh_atoms}</td>
                  </tr>
                  <tr>
                    <td>Number of substances removed during filtering</td>
                    <td>{filtered_substances_count}</td>
                  </tr>
                  <tr>
                    <th colspan="2">Descriptors</th>
                  </tr>
                  <tr>
                    <td>Descriptor type</td>
                    <td>Faulon Signatures</td>
                  </tr>
                  <tr>
                    <td>Faulon signatures heights</td>
                    <td>{sign_height_min}-{sign_height_max}</td>
                  </tr>
                  <tr>
                    <td>Faulon signatures generation running time</td>
                    <td>{sign_gen_runtime}</td>
                  </tr>
                  <tr>
                    <td>Faulon signatures sparse training data set generation running time</td>
                    <td>{sparse_train_gen_runtime}</td>
                  </tr>
                  <tr>
                    <td>Faulon signatures sparse test data set generation running time</td>
                    <td>{sparse_test_gen_runtime}</td>
                  </tr>
                  <tr>
                    <td>Total number of unique Faulon signatures</td>
                    <td>{signatures_count}</td>
                  </tr>
                  <tr>
                    <th colspan="2">Model</th>
                  </tr>
                  <tr>
                    <td>Modelling method</td>
                    <td>RBF-kernel SVM</td>
                  </tr>
                  <tr>
                    <td>RMSD test set</td>
                    <td>{testset_rmsd}</td>
                  </tr>
                  <tr>
                    <td>Model choice</td>
                    <td>Maximal accuracy, with 5-fold cross-validated accuracy (RMSD) as objective function</td>
                  </tr>
                  <tr>
                    <td>Model validation</td>
                    <td>Accuracy measued on an external test set</td>
                  </tr>
                  <tr>
                    <td>Learning parameters</td>
                    <td>kernel=RBF, c={svm_cost}, gamma={svm_gamma}</td>
                  </tr>
                  <tr>
                    <td colspan="2" ><img src=\"{png_image_path}\" style="width:400px;height:400px;margin:2em 7em;border: 1px solid #ccc;"></td>
                  </tr>
                </tbody>
              </table>
            </body>
            </html>
        '''.format(
            train_size=self.train_size,
            test_size=self.test_size,
            min_nonh_atoms='5',
            max_nonh_atoms='50',
            filtered_substances_count='N/A',
            sign_height_min='0',
            sign_height_max='3',
            sign_gen_runtime=report_info['sample_train_and_test']['runningtime'],
            sparse_train_gen_runtime=report_info['create_sparse_train_dataset']['runningtime'],
            sparse_test_gen_runtime=report_info['create_sparse_train_dataset']['runningtime'],
            signatures_count=report_info['create_sparse_train_dataset']['numsign'],
            testset_rmsd=self.get_svm_rmsd(),
            svm_cost=self.svm_cost,
            svm_gamma=self.svm_gamma,
            png_image_path=self.get_input('assess_svm_plot_target').path.split('/')[-1]
        )).strip()

        return html_head + html_content


    # SOME HELPER METHODS
    def tag(self, tagname, content):
        return '<{t}>{c}</{t}>'.format(t=tagname, c=content)


    # WHAT THE TASK DOES
    def run(self):

        # WRITE HTML REPORT
        with open(self.output()['html_report'].path, "w") as html_report_file:
            log.info("Writing HTML report to file: " + self.output()['html_report'].path)
            html_report_file.write(self.get_html_report_content())


# ====================================================================================================

class CreateElasticNetModel(DependencyMetaTask, DatasetNameMixin, TaskHelpers):

    # INPUT TARGETS
    train_dataset_target = luigi.Parameter()

    # TASK PARAMETERS
    l1_value = luigi.Parameter()
    lambda_value = luigi.Parameter()

    # DEFINE OUTPUTS
    def output(self):
        return { 'model': luigi.LocalTarget(self.get_input('train_dataset_target').path + ".model_{l}_{y}".format(l=self.get_value('l1_value'),y=self.get_value('lambda_value'))) }

    def run(self):
        self.x([JAVA_PATH, "-jar", "jars/CreateElasticNetModel.jar",
                "-inputfile", self.get_input('train_dataset_target').path,
                "-l1ratio", str(self.get_value('l1_value')),
                "-lambda", str(self.get_value('lambda_value')),
                "-outputfile", self.output()['model'].path,
                "-silent"])

        #self.lx(["mv",
        #         self.get_input('train_dataset_target').path + ".model",
        #         self.get_input('train_dataset_target').path + ".model_{l}_{y}".format(l=self.get_value('l1_value'),y=self.get_value('lambda_value'))])


# ====================================================================================================

class PredictElasticNetModel(DependencyMetaTask, DatasetNameMixin, TaskHelpers):

    # INPUT TARGETS
    elasticnet_model_target = luigi.Parameter()
    test_dataset_target = luigi.Parameter()

    # TASK PARAMETERS
    l1_value = luigi.Parameter()
    lambda_value = luigi.Parameter()

    def output(self):
        return { 'prediction' : luigi.LocalTarget(self.get_input('elasticnet_model_target').path + ".prediction") }

    def run(self):
        self.x([JAVA_PATH, "-jar", "jars/PredictElasticNetModel.jar",
                "-modelfile", self.get_input('elasticnet_model_target').path,
                "-testset", self.get_input('test_dataset_target').path,
                "-outputfile", self.output()['prediction'].path,
                "-silent"])

# ====================================================================================================

class EvaluateElasticNetPrediction(DependencyMetaTask, DatasetNameMixin, TaskHelpers):
     # INPUT TARGETS
     test_dataset_target = luigi.Parameter()
     prediction_target = luigi.Parameter()

     # TASK PARAMETERS
     l1_value = luigi.Parameter()
     lambda_value = luigi.Parameter()

     # DEFINE OUTPUTS
     def output(self):
         return { 'evaluation' : luigi.LocalTarget( self.get_input('prediction_target').path + '.evaluation' ) }

     # WHAT THE TASK DOES
     def run(self):
         with gzip.open(self.get_input('test_dataset_target').path) as testset_file, self.get_input('prediction_target').open() as prediction_file:
             original_vals = [float(line.split(' ')[0]) for line in testset_file]
             predicted_vals = [float(val.strip('\n')) for val in prediction_file]
         squared = [(pred-orig)**2 for orig, pred in zip(original_vals, predicted_vals)]
         rmsd = math.sqrt( sum(squared) / len(squared) )
         with self.output()['evaluation'].open('w') as outfile:
             csvwriter = csv.writer(outfile)
             csvwriter.writerow(['rmsd', rmsd])
             csvwriter.writerow(['l1ratio', self.get_value('l1_value')])
             csvwriter.writerow(['lambda', self.get_value('lambda_value')])

# ====================================================================================================

class ElasticNetGridSearch(DependencyMetaTask, DatasetNameMixin, TaskHelpers):

    # INPUT TARGETS
    train_dataset_target = luigi.Parameter()
    test_dataset_target = luigi.Parameter()
    replicate_id = luigi.Parameter()

    # TASK PARAMETERS
    l1_steps = luigi.Parameter()
    lambda_steps = luigi.Parameter()

    def grid_step_generator(self):
        for l1 in ast.literal_eval(self.l1_steps):
            for lambda_value in ast.literal_eval(self.lambda_steps):
                create_elasticnet_model = CreateElasticNetModel(
                        train_dataset_target = self.train_dataset_target,
                        l1_value = l1,
                        lambda_value = lambda_value,
                        dataset_name = self.dataset_name,
                        replicate_id = self.replicate_id,
                        accounted_project = self.accounted_project )
                predict_elasticnet_model = PredictElasticNetModel(
                        l1_value = l1,
                        lambda_value = lambda_value,
                        elasticnet_model_target =
                            { 'upstream' : { 'task' : create_elasticnet_model,
                                             'port' : 'model' } },
                        test_dataset_target = self.test_dataset_target,
                        dataset_name = self.dataset_name,
                        replicate_id = self.replicate_id,
                        accounted_project = self.accounted_project )
                eval_elasticnet_prediction = EvaluateElasticNetPrediction(
                        l1_value = l1,
                        lambda_value = lambda_value,
                        test_dataset_target = self.test_dataset_target,
                        prediction_target =
                            { 'upstream' : { 'task' : predict_elasticnet_model,
                                             'port' : 'prediction' } },
                        dataset_name = self.dataset_name,
                        replicate_id = self.replicate_id,
                        accounted_project = self.accounted_project )
                yield eval_elasticnet_prediction

    def requires(self):
        return [x for x in self.grid_step_generator()]

    def output(self):
        inpath = self.input()[0]['evaluation'].path
        return { 'optimal_parameters_info' : luigi.LocalTarget(inpath +
                                                  ".gridsearch_l1_" +
                                                  "_".join([str(x).replace(".","-") for x in ast.literal_eval(self.l1_steps)]) +
                                                  "_lambda_" +
                                                  "_".join([str(x).replace(".","-") for x in ast.literal_eval(self.lambda_steps)]) +
                                                  "_optimal_params" ) }

    def run(self):
        rmsd_infos = []
        for rmsd_task in self.input():
            rmsd_info_obj = ConfigParser()
            rmsd_info_obj.read(rmsd_task['evaluation'].path)
            rmsd_info = dict(rmsd_info_obj.items('evaluation_info'))
            rmsd_infos.append(rmsd_info)

        best_rmsd_info = min(rmsd_infos, key = lambda x: float(x['rmsd']))
        log.info( "BEST RMSD INFO: " + str(best_rmsd_info) )

        with self.output()['optimal_parameters_info'].open("w") as outfile:
            info_obj = ConfigParser()
            info_obj.add_section('optimal_parameters')
            for key in best_rmsd_info.keys():
                info_obj.set('optimal_parameters', key, best_rmsd_info[key])
            info_obj.write(outfile)

# ====================================================================================================

class BuildP2Sites(DependencyMetaTask, TaskHelpers, DatasetNameMixin):

    # INPUT TARGETS
    signatures_target = luigi.Parameter()
    sparse_train_dataset_target = luigi.Parameter()
    svmmodel_target = luigi.Parameter()
    assess_svm_log_target = luigi.Parameter()

    # TASK PARAMETERS
    dataset_name = luigi.Parameter()
    accounted_project = luigi.Parameter()
    test_size = luigi.Parameter()
    svm_cost = luigi.Parameter()
    svm_gamma = luigi.Parameter()

    def output(self):
        return { 'plugin_bundle' : luigi.LocalTarget(self.temp_folder_path() + '/%s_p2_site.zip' % self.dataset_name) }

    def temp_folder_path(self):
        return 'data/' + '/'.join(self.get_input('svmmodel_target').path.split('/')[-2:]) + '.plugin_bundle'

    def temp_folder_abspath(self):
        return os.path.abspath(self.get_input('svmmodel_target').path + '.plugin_bundle')

    def get_svm_rmsd(self):

        with self.get_input('assess_svm_log_target').open() as infile:
            csv_reader = csv.reader(infile)
            for row in csv_reader:
                if row[0] == 'RMSD':
                    svm_rmsd = float(row[1])
        return svm_rmsd

    def get_signatures_count(self):
        signatures_count = 0
        with self.get_input('signatures_target').open() as infile:
            for line in infile:
                signatures_count += 1
        return signatures_count

    def get_properties_file_contents(self, signatures_file, svmmodel_file):
        return textwrap.dedent('''
            <?xml version="1.0" encoding="UTF-8"?>
            <?eclipse version="3.4"?>
            <plugin>
               <extension
                    point="net.bioclipse.decisionsupport">

                <test
                        id="net.bioclipse.mm.{dataset_name}"
                        name="Model predicted {dataset_name}"
                        class="net.bioclipse.ds.libsvm.SignaturesLibSVMPrediction"
                        endpoint="net.bioclipse.mm"
                        informative="false">

                        <parameter name="isClassification" value="false"/>

                        <parameter name="signatures.min.height" value="1"/>
                        <parameter name="signatures.max.height" value="3"/>
                        <resource name="modelfile" path="{svmmodel_file}"/>
                        <resource name="signaturesfile" path="{signatures_file}"/>
                        <parameter name="Model type" value="QSAR"/>
                        <parameter name="Learning model" value="SVM"/>

                        <parameter name="Model performance" value="{svm_rmsd}"/>
                        <parameter name="Model choice" value="External test set of {test_size} observations"/>
                        <parameter name="Learning parameters" value="kernel=RBF, c={svm_cost}, gamme={svm_gamma}"/>
                        <parameter name="Descriptors" value="Signatures (height 1-3)"/>
                        <parameter name="Observations" value="{test_size}"/>
                        <parameter name="Variables" value="{signatures_count}"/>

                        <parameter name="lowPercentile" value="0"/>
                        <parameter name="highPercentile" value="1"/>
                    </test>
               </extension>
            </plugin>
        '''.format(
                dataset_name=self.dataset_name,
                svmmodel_file=svmmodel_file,
                signatures_file=signatures_file,
                svm_rmsd=self.get_svm_rmsd(),
                test_size=self.test_size,
                svm_cost=self.svm_cost,
                svm_gamma=self.svm_gamma,
                signatures_count=self.get_signatures_count()
            )).strip()

    def run(self):
        temp_folder = self.temp_folder_path()
        temp_folder_abspath = self.temp_folder_abspath()
        model_folder = 'model'
        model_folder_abspath = temp_folder_abspath + '/model'

        # Create temp and model folder
        self.lx(['mkdir -p', model_folder_abspath])

        # Copy some files into the newly created model folder
        # (which is in turn inside the newly created temp folder)

        signatures_file = model_folder + '/signatures.txt'
        signatures_file_abspath = model_folder_abspath + '/signatures.txt'
        self.lx(['cp',
                 self.get_input('signatures_target').path,
                 signatures_file_abspath])

        train_dataset_file = model_folder + '/sparse_train_datset.csr'
        train_dataset_file_abspath = model_folder_abspath + '/sparse_train_datset.csr'
        self.lx(['cp',
                 self.get_input('sparse_train_dataset_target').path,
                 train_dataset_file_abspath])

        svmmodel_file = model_folder + '/model.svm'
        svmmodel_file_abspath = model_folder_abspath + '/model.svm'
        self.lx(['cp',
                 self.get_input('svmmodel_target').path,
                 svmmodel_file_abspath])


        # -----------------------------------------------------------------------
        # PROPERTIES FILE

        properties_file_contents = self.get_properties_file_contents(signatures_file, svmmodel_file)
        with open(temp_folder + '/plugin.xml', 'w') as pluginxml_file:
            pluginxml_file.write(properties_file_contents)

        # Zip the files (Has to happen after writing of plugin.xml, in order to include it)
        cmd = ['cd', temp_folder, ';',
               'zip -r ', 'plugin_bundle.zip', './*']
        self.lx(cmd)

        # -----------------------------------------------------------------------
        # ENDPOINT FILE

        # Create Endpoint XML file
        endpoint_xmlfile_content = textwrap.dedent('''
            <?xml version="1.0" encoding="UTF-8"?>
            <?eclipse version="3.4"?>
            <plugin>
               <extension
                     point="net.bioclipse.decisionsupport">
                     <endpoint
                           id="net.bioclipse.mm"
                           description="Predicted logp based on acd logP"
                           icon="biolock.png"
                           name="Predicted Properties">
                     </endpoint>
                </extension>
            </plugin>
        ''').strip()
        with open(temp_folder + '/endpoint_bundle.xml','w') as endpoint_xmlfile:
            endpoint_xmlfile.write(endpoint_xmlfile_content)

        # Create Endpoint BND file
        endpoint_bndfile_content = textwrap.dedent('''
            Bundle-Version:1.0.0
            Bundle-SymbolicName: net.bioclipse.mm.endpoint;singleton:=true
            -includeresource: plugin.xml=endpoint_bundle.xml
            -output source.p2/plugins/${bsn}-${Bundle-Version}.jar
        ''').strip()
        with open(temp_folder + '/endpoint_bundle.bnd','w') as endpoint_bndfile:
            endpoint_bndfile.write(endpoint_bndfile_content)

        # Process Endpoint
        self.lx(['cd', temp_folder, ';',
                JAVA_PATH, '-jar',
                '/proj/b2013262/nobackup/workflows/workflows/jars/bnd-2.3.0.jar',
                'endpoint_bundle.bnd'])


        # -----------------------------------------------------------------------
        # PLUGIN FILE

        time_stamp = time.strftime("%Y%m%d%H%M%S")
        bundle_version = '0.0.0.' + time_stamp
        bundle_id = 'net.bioclipse.mm.' + self.dataset_name

        # Create Plugin BND file
        plugin_bndfile_content = textwrap.dedent('''
            Bundle-Version:{bundle_version}
            Bundle-SymbolicName: {bundle_id};singleton:=true
            -includeresource: @plugin_bundle.zip
            -output source.p2/plugins/${{bsn}}-${{Bundle-Version}}.jar
            Require-Bundle: net.bioclipse.ds.libsvm
        '''.format(
            bundle_version = bundle_version,
            bundle_id = bundle_id
        )).strip()
        with open(temp_folder + '/plugin_bundle.bnd','w') as plugin_bndfile:
            plugin_bndfile.write(plugin_bndfile_content)

        # Process
        self.lx(['cd', temp_folder, ';',
                 JAVA_PATH, '-jar',
                 '/proj/b2013262/nobackup/workflows/workflows/jars/bnd-2.3.0.jar',
                 'plugin_bundle.bnd'])

        # Create feature file

        features_file_content = textwrap.dedent('''
        <?xml version="1.0" encoding="UTF-8"?>
        <feature id="{bundle_id}" label="{feature_label}" version="{bundle_version}" provider-name="Bioclipse" plugin="{bundle_id}">
        <description></description>
        <copyright></copyright>
        <license url=""></license>

        <requires>
        <import plugin="net.bioclipse.ds.libsvm" version="2.6.2" match="greaterOrEqual"/>
        </requires>

        <plugin id="{bundle_id}" download-size="0" install-size="0" version="{bundle_version}" unpack="false"/>
        <plugin id="net.bioclipse.mm.endpoint" download-size="0" install-size="0" version="1.0.0" unpack="false"/>

        </feature>
        '''.format(bundle_id = bundle_id,
                   bundle_version = bundle_version,
                   feature_label = self.dataset_name)).strip()

        # Create a folder for features
        features_folder = temp_folder + '/source.p2/features'
        self.lx(['mkdir -p', features_folder])

        # Write out the content of the feature.xml file
        with open(temp_folder + '/feature.xml', 'w') as features_file:
            features_file.write(features_file_content)

        # Zip the feature.xml file, into a file in the features folder
        self.lx(['pwd; cd', temp_folder, ';',
                 'zip', 'source.p2/features/' + bundle_id + '.jar', 'feature.xml'])

        # -----------------------------------------------------------------------
        # Assemble it all together
        pubcmd = ['cd', temp_folder, ';',
                  '/proj/b2013262/nobackup/eclipse_director/director/director',
                  '-application', 'org.eclipse.equinox.p2.publisher.FeaturesAndBundlesPublisher',
                  '-metadataRepository', 'file:' + temp_folder_abspath + '/site.p2',
                  '-artifactRepository', 'file:' + temp_folder_abspath + '/site.p2',
                  '-metadataRepositoryName', '"MM SVM Model for ' + self.dataset_name + '"',
                  '-source', temp_folder_abspath + '/source.p2',
                  '-publishArtifacts']
        self.lx(pubcmd)

        zipcmd = ['cd', temp_folder + '/site.p2;',
                  'zip -r', '../%s_p2_site.zip' % self.dataset_name,
                  './*']
        self.lx(zipcmd)

# ====================================================================================================

class PushP2SiteToRemoteHost(DependencyMetaTask, TaskHelpers, DatasetNameMixin):
    # INPUT TARGETS
    plugin_bundle_target = luigi.Parameter()

    # TASK PARAMETERS
    remote_host = luigi.Parameter()
    remote_user = luigi.Parameter()
    remote_base_folder = luigi.Parameter()

    def output(self):
        return { 'completion_marker' : luigi.LocalTarget(self.get_input('plugin_bundle_target').path + '.p2site_pushed' ) }

    def run(self):
        remote_folder = self.remote_base_folder + '/' + self.replicate_id
        remote_command = 'mkdir -p ' + remote_folder

        self.lx(['ssh -o PubkeyAuthentication=no',
                 '%s@%s' % (self.remote_user, self.remote_host),
                 '\'' + remote_command + '\''])

        # Copy the p2 site zip file to the remote host via SCP
        self.lx(['scp',
                 self.get_input('plugin_bundle_target').path,
                 '%s@%s:%s/' % (self.remote_user,
                                self.remote_host,
                                remote_folder)])

        # Write some dummy content to the completion marker
        self.lx(['echo',
                 '"p2 site pushed"',
                 '>',
                 self.output()['completion_marker'].path])

# ====================================================================================================

class BuildP2SiteOnRemoteHost(DependencyMetaTask, TaskHelpers, DatasetNameMixin):
    # INPUT TARGETS
    pushp2_completion_target = luigi.Parameter()
    plugin_bundle_target = luigi.Parameter()

    # TASK PARAMETERS
    remote_host = luigi.Parameter()
    remote_user = luigi.Parameter()
    remote_folder = luigi.Parameter()
    eclipse_dir = luigi.Parameter()
    comp_repo_bin_path = luigi.Parameter()
    bundle_name = luigi.Parameter()

    def output(self):
        return { 'completion_marker' : luigi.LocalTarget(self.get_input('pushp2_completion_target').path + '.p2site_built' ) }

    def run(self):
        p2_site_zip_filename = self.get_input('plugin_bundle_target').path.split('/')[-1]

        # Unzip the site zip file
        remote_command = 'cd {basedir}; unzip {dir}/{zipfile} -d {dir}'.format(
                basedir = self.remote_folder,
                dir = self.replicate_id,
                zipfile = p2_site_zip_filename)

        self.lx(['ssh -o PubkeyAuthentication=no',
                 '%s@%s' % (self.remote_user, self.remote_host),
                 '\'' + remote_command + '\''])

        # Build the p2 site previously pushed, on remote host
        #TODO: bundle_zipfile should be replaced with path to unpacked folder (relative to /var/www/armadillo) (no slashes)
        remote_command = 'export ECLIPSE_DIR={eclipse_dir}/;{comp_repo_bin} {repo_folder} --name "{bundle_name}" add {site_folder}'.format(
                eclipse_dir=self.eclipse_dir,
                comp_repo_bin=self.comp_repo_bin_path,
                repo_folder=self.remote_folder,
                bundle_name=self.bundle_name,
                site_folder=self.replicate_id
        )
        self.lx(['ssh -o PubkeyAuthentication=no',
                 '%s@%s' % (self.remote_user, self.remote_host),
                 '\'' + remote_command + '\''])

        # Write some dummy content to the completion marker
        self.lx(['echo',
                 '"p2 site built"',
                 '>',
                 self.output()['completion_marker'].path])

# ====================================================================================================

class ExistingDataFiles(luigi.ExternalTask):
    '''External task for getting hand on existing data files'''

    # PARAMETERS
    dataset_name = luigi.Parameter()

    # DEFINE OUTPUTS
    def output(self):
        test_neg = os.path.join(os.path.abspath('./data'), 'test_neg_' + str(self.dataset_name) + '.data')
        test_pos = os.path.join(os.path.abspath('./data'), 'test_pos_' + str(self.dataset_name) + '.data')
        training = os.path.join(os.path.abspath('./data'), 'training_' + str(self.dataset_name) + '.data')
        log.debug("%s, %s, %s" % (test_neg, test_pos, training))
        return { 'test_neg' : luigi.LocalTarget(test_neg),
                 'test_pos' : luigi.LocalTarget(test_pos),
                 'training' : luigi.LocalTarget(training) }

# ====================================================================================================

class GenerateFingerprint(DependencyMetaTask, DatasetNameMixin, TaskHelpers):
    '''
    Usage of the FingerprintsGenerator Jar file:

    usage: java FingerprintsGenerator
    -fp <String>                           fingerprint name
    -inputfile <file [inputfile.smiles]>   filename for input SMILES file
    -limit <integer>                       Number of lines to read. Handy for
                                           testing.
    -maxatoms <integer>                    Maximum number of non-hydrogen
                                           atoms. [default: 50]
    -minatoms <integer>                    Minimum number of non-hydrogen
                                           atoms. [default: 5]
    -outputfile <file [output.sign]>       filename for generated output file
    -parser <String>                       parser type
    -silent                                Do not output anything during run
    -threads <integer>                     Number of threads. [default:
                                                               number of cores]

    Supported parser modes are:
    1 -- for our ChEMBL datasets (*.data)
    2 -- for our other datasets (*.smi)

    Supported fingerprints are:
    ecfbit
    ecfpcount
    extended
    signbit
    signcount
    '''

    # INPUT TARGETS
    dataset_target = luigi.Parameter()

    # PARAMETERS
    fingerprint_type = luigi.Parameter()

    # DEFINE OUTPUTS
    def output(self):
        return { 'fingerprints' : luigi.LocalTarget(self.get_input('dataset_target').path + '.' + self.fingerprint_type + '.csr') }

    def run(self):
        self.x([JAVA_PATH, '-jar jars/FingerprintsGenerator.jar',
                '-fp', self.fingerprint_type,
                '-inputfile', self.get_input('dataset_target').path,
                '-parser', '1',
                '-outputfile', self.output()['fingerprints'].path])

# ====================================================================================================

class CompactifyFingerprintHashes(DependencyMetaTask, DatasetNameMixin, TaskHelpers):
    '''
    Takes a sparse dataset as input and compacts the values before :'s to integers
    counting from 1 and upwards.
    '''

    train_fingerprints_target = luigi.Parameter()
    test_fingerprints_target = luigi.Parameter()

    def output(self):
        return { 'train_fingerprints' : luigi.LocalTarget(self.get_input('train_fingerprints_target').path + '.compacted'),
                 'test_fingerprints' : luigi.LocalTarget(self.get_input('test_fingerprints_target').path + '.compacted') }

    def run(self):
        inout_filepaths = [(self.get_input('train_fingerprints_target').path,
                            self.output()['train_fingerprints'].path),
                           (self.get_input('test_fingerprints_target').path,
                            self.output()['test_fingerprints'].path)]

        counter = 0
        register = {}

        for infile_path, outfile_path in inout_filepaths:
            with open(infile_path) as infile, open(outfile_path, 'w') as outfile:
                reader = csv.reader(infile, delimiter=' ')
                for row in reader:
                    # Make sure the first columns stays on 1st column
                    newrow = [row[0]]
                    newcols = []
                    for col in row[1:]:
                        if ':' in col:
                            newcol = ''
                            parts = col.split(':')
                            val = int(parts[0])
                            part2 = parts[1]
                            if val in register:
                                newval = register[val]
                            else:
                                counter += 1
                                register[val] = counter
                                newval = counter
                            newcol = str(newval) + ':' + str(part2)
                            newcols.append(newcol)
                        else:
                            newcols.append(col)
                    newcols = sorted(newcols, key=lambda(x): int(x.split(':')[0]))
                    newrow.extend(newcols)
                    outfile.write(' '.join(newrow) + '\n')

# ====================================================================================================

class BCutPreprocess(DependencyMetaTask, TaskHelpers, DatasetNameMixin):

    # INPUT TARGETS
    signatures_target = luigi.Parameter()

    # TASK PARAMETERS
    replicate_id = luigi.Parameter()

    def output(self):
        return { 'bcut_preprocessed' : luigi.LocalTarget(self.get_input('signatures_target').path + '.bcut_preproc'),
                 'bcut_preprocess_log' : luigi.LocalTarget(self.get_input('signatures_target').path + '.bcut_preproc.log') }

    def run(self):
        self.x([JAVA_PATH, '-cp ../../lib/cdk/cdk-1.4.19.jar:jars/bcut.jar bcut',
                self.get_input('signatures_target').path,
                self.output()['bcut_preprocessed'].path])

# ====================================================================================================

class BCutSplitTrainTest(DependencyMetaTask, TaskHelpers, DatasetNameMixin):

    # INPUT TARGETS
    bcut_preprocessed_target = luigi.Parameter()

    # TASK PARAMETERS
    train_size = luigi.Parameter()
    test_size = luigi.Parameter()
    replicate_id = luigi.Parameter()

    def output(self):
        return { 'train_dataset' : luigi.LocalTarget(self.get_input('bcut_preprocessed_target').path + '.{tr}_{te}_bcut_train'.format(
                                        tr=str(self.train_size),
                                        te=str(self.test_size))),
                 'test_dataset' : luigi.LocalTarget(self.get_input('bcut_preprocessed_target').path + '.{tr}_{te}_bcut_test'.format(
                                        tr=str(self.train_size),
                                        te=str(self.test_size))) }

    def run(self):
        self.x(['/usr/bin/xvfb-run /sw/apps/R/x86_64/3.0.2/bin/Rscript r/pick_bcut.r',
                '--input_file=%s' % self.get_input('bcut_preprocessed_target').path,
                '--training_file=%s' % self.output()['train_dataset'].path,
                '--test_file=%s' % self.output()['test_dataset'].path,
                '--training_size=%s' % self.train_size,
                '--test_size=%s' % self.test_size])

        # Documentation of pick_bcut.r commandline flags:

        '''
        Usage: /proj/b2013262/nobackup/workflow_components/SampleTrainingAndTest/src/sample/trainingAndTest/pick_bcut.r [options]


        Options:
        --input_file=INPUT_FILE
                filename for BCUT data [required]

        --test_file=TEST_FILE
                filename for generated test set

        --training_file=TRAINING_FILE
                filename for generated training set

        -x [1..N, X%, OR REST], --test_size=[1..N, X%, OR REST]
                Size of test set [default 20]

        -y [1..N, X%, OR REST], --training_size=[1..N, X%, OR REST]
                Size of training set [default rest]

        -c NUMBER, --centers=NUMBER
                Number of kmeans centers [default 100]

        -i NUMBER, --iterations=NUMBER
                Number of kmeans iterations [default 10]

        -k [0, 1, 2], --keep_na=[0, 1, 2]
                0=>remove NA; 1=>NA OK in test; 2=>NA OK; [default 0]

        -f, --force
                Overwrite output files if found

        -r, --random_seed
                Set a seed for reproducibly random results

        -s, --silent
                Suppress comments to STDOUT

        -h, --help
                Show this help message and exit
        '''
