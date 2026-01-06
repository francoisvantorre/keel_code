## TODO DEBUG :
## 01-004p2 : cov 164 (108-08-7) le script ne va pas chercher la bonne valeur (cov164 dans data_cleaned_numeric_cov_with_code est à 0 ; cov164 dans fichier cutpoint a une valeur...) ; décallage d'indice ?

import sys # args
import re # regular expressions


CSV_SEPARATOR = ";"

class Attribute:

    def __init__(self,id,name,values_orig):
        self.id = id
        self.name = name
        self.values_orig = values_orig

    def __str__(self):
        return "attribute {id: "+str(self.id) +" ; name: " + self.name + " ; values_orig: "+ str(self.values_orig) + "}"

    def __repr__(self):
        return str(self)

    def export(self):
        return "@attribute " + self.name + " {" + ",".join(self.values_orig) + "}"

    def isCOV(self):
        return False

class CutpointCOV:

    def __init__(self, lower_bound, upper_bound):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def __str__(self):
        if(self.lower_bound == None and self.upper_bound == None):
            return "0"
        elif(self.lower_bound == None):
            return "<"+str(self.upper_bound)
        elif(self.upper_bound == None):
            return ">"+str(self.lower_bound)
        else:
            return "["+ str(self.lower_bound) + "-" + str(self.upper_bound) + "["

    def __repr__(self):
        return str(self)


class AttributeCOV(Attribute):

    THRESHOLD = 0.32

    def __init__(self,id,name,cov_id,cas_name,values_orig):
        super().__init__(id,name,values_orig)
        self.cas_name = cas_name
        self.cov_id   = cov_id
        self.cutpoints = []
        self.cutpoints_threshold = []

    def isCOV(self):
        return True

    def set_cutpoints(self,cutpoints):
        previous = None
        # bin cutpoint 0 for threshold cutpoint
        nice_cutpoint_threshold = CutpointCOV(None,None)
        self.cutpoints_threshold.append(nice_cutpoint_threshold)

        for cutpoint in cutpoints:
            # first
            if(previous == None):
                nice_cutpoint = CutpointCOV(None,cutpoint)
                # under the threshold
                if(cutpoint < self.THRESHOLD):
                    nice_cutpoint_threshold = CutpointCOV(None,self.THRESHOLD)
                    self.cutpoints_threshold.append(nice_cutpoint_threshold)
                elif(cutpoint > self.THRESHOLD):
                    nice_cutpoint_threshold = CutpointCOV(None,cutpoint)
                    self.cutpoints_threshold.append(nice_cutpoint_threshold)
            else:
                nice_cutpoint = CutpointCOV(previous,cutpoint)
                # ignore cutpoints under threshold
                if(cutpoint > self.THRESHOLD):
                    if(previous < self.THRESHOLD):
                        nice_cutpoint_threshold = CutpointCOV(self.THRESHOLD,cutpoint)
                    else:
                        nice_cutpoint_threshold = CutpointCOV(previous,cutpoint)
                    self.cutpoints_threshold.append(nice_cutpoint_threshold)

            previous = cutpoint
            self.cutpoints.append(nice_cutpoint)

        if(previous > self.THRESHOLD):
            nice_cutpoint = CutpointCOV(previous,None)
        else:
            nice_cutpoint = CutpointCOV(self.THRESHOLD,None)
        self.cutpoints.append(nice_cutpoint)
        self.cutpoints_threshold.append(nice_cutpoint)

    def export(self):
        return "@attribute " + self.name + "_" + self.cas_name + " <{" + ",".join(str(x) for x in self.cutpoints_threshold) + "}"

    # gives the index of the cutpointthreshold for this individual value
    def get_cpt_index_for_value(self,individual_value):
        index = 0
        if(individual_value == 0):
            return 0
        if(individual_value == self.THRESHOLD):
            individual_value = self.THRESHOLD - 0.01
        for cutpoint in self.cutpoints_threshold:
            if(cutpoint.upper_bound != None and individual_value < cutpoint.upper_bound):
                return index
            index += 1
        return len(self.cutpoints_threshold)-1

    def get_cutpoints(self):
        return self.cutpoints

    def get_cutpoints_threshold(self):
        return self.cutpoints_threshold

    def __str__(self):
        return "attribute {id: "+str(self.id) +" ; name: " + self.name + " ; values_orig: "+ str(self.values_orig) + " cutpoints: " + str(self.cutpoints) + " cutpoints_threshold: " +str(self.cutpoints_threshold) + "}"

    def __repr__(self):
        return str(self)

def load_orig_file(filename):
    f_orig = open(filename,"r")
    for line in f_orig.readlines():
        print(line)
        break

def load_attributes(keel_dat_filename, keel_result_filename, cas_list_filename):
    f_keel_dat = open(keel_dat_filename, "r")
    f_cas_list = open(cas_list_filename, "r")
    attributes_by_id = []
    cas_name_by_cov_id = {}
    id = 0
    # chargement CAS_LIST
    for line in f_cas_list.readlines():
        if("cov" in line):
            result = re.findall(r"[0-9]+,([^,]+),(cov[0-9]+)",line)
            if(len(result) > 0 and len(result[0])==2):
                cas_name = result[0][0]
                cov_id   = result[0][1]
                cas_name_by_cov_id[cov_id] = cas_name

    # chargement attributes fichier dat keel
    for line in f_keel_dat.readlines():
        if(line.startswith('@attribute')):
            result = re.findall(r"@attribute ([^\{]+)\{([^\}]+)",line)
            if(len(result) > 0 and len(result[0])==2):
                name = result[0][0].rstrip()
                values = result[0][1]
                if(name.startswith('cov')):
                    cov_id = int(name[3:])
                    attribute = AttributeCOV(id,name,cov_id, cas_name_by_cov_id[name],values.split("."))
                else:
                    attribute = Attribute(id,name,values.split("."))
                attributes_by_id.append(attribute)
                id+=1
    print(f"{len(attributes_by_id)} attributes loaded")
    # chargement attributes fichier result keel
    f_keel_result = open(keel_result_filename, "r")
    attribute_cutpoints = []
    for line in f_keel_result.readlines():
        if(not line.startswith("Number of cut points of attribute")):
            result = re.findall(r"Cut point [0-9]+ of attribute [0-9]+ : ([0-9]+\.[0-9]+)$",line)
            if(len(result) > 0 and len(result)==1):
                attribute_cutpoints.append(float(result[0]))
            else:
                print("problem loading line "+line)
        else:
            result = re.findall(r"Number of cut points of attribute ([0-9]+) :",line)
            if(len(result) > 0 and len(result)==1):
                attribute_id = int(result[0])
                print(attribute_id)
                attributes_by_id[attribute_id].set_cutpoints(attribute_cutpoints)
                attribute_cutpoints = []
            else:
                print("problem loading line " + line)

    f_keel_dat.close()
    f_keel_result.close()
    f_cas_list.close()

    return attributes_by_id


# exports all the descriptive part of the dataset
def export_file_header(attributes_by_id, keel_dat_filename,orig_filename):
    f_result = open(f"{sys.argv[2][:-4]}.DATA","w")
    f_orig = open(orig_filename,"r")
    individuals_by_id = {}

    # load original values for individuals
    f_orig.seek(1)
    for line in f_orig.readlines():
        values = line.split(CSV_SEPARATOR)
        id = values[0]
        individuals_by_id[id] = values
        print('individual loaded : ' + id)


    f_keel_dat = open(keel_dat_filename, "r")

    for line in f_keel_dat.readlines():
        if(line.startswith('@relation')):
            f_result.write(line)
            break

    # export of attributes
    for attribute in attributes_by_id:
        #print(attribute)
        print(attribute.export(),file=f_result)

    f_keel_dat.seek(0)
    line_number = 1
    for line in f_keel_dat.readlines():

        # inputs & outputs & data
        if(line.startswith('@inputs') or line.startswith('@outputs') or line.startswith('@data')):
            f_result.write(line)

        # convert & write individuals
        if(not line.startswith('@')):
            values = line.split(",")
            new_values = []
            if(len(values) < len(attributes_by_id)):
                print("problem with individual " + str(line_number) + ": invalid number of values")
            attribute_id = 0
            individual_id = values[0]
            for value in values:
                if(attributes_by_id[attribute_id].isCOV()):
                    attribute = attributes_by_id[attribute_id]
                    value = individuals_by_id[individual_id][attribute.cov_id]
                    new_values.append(str(attributes_by_id[attribute_id].get_cpt_index_for_value(float(value))))
                else:
                    new_values.append(value)
                attribute_id += 1
            # write individual
            individual = ",".join(new_values)
            #print("export individual " + individual)
            f_result.write(individual)

        line_number += 1



    f_result.close()
    f_keel_dat.close()
    f_orig.close()

# TODO : vérifier que les 0.32 finissent bien dans le <0.32 et pas dans la tranche d'après
# python3 id3_en_mieux_v3.py Data_cleaned_numeric_cov_with_code.csv ID3-D.data_cleaned_numeric_cov_95_code1tra.dat result1e0.txt
if __name__ == "__main__":
    if(len(sys.argv) < 5):
        print("Usage : python id3_en_mieux.py orig_file_with_ids.csv keel_result.dat keel_result.txt CAS_list.txt")
    else:
        print("Parameters:")
        print("orig file with ids : " + sys.argv[1])
        print("keel .dat file : " + sys.argv[2])
        print("keel result .txt file : " + sys.argv[3])
        print("CASLIST file : " + sys.argv[4])

        attributes_by_id = load_attributes(sys.argv[2], sys.argv[3], sys.argv[4])
        print(str(len(attributes_by_id)) + " attributes loaded !")
        for i in range(5,10):
            print(attributes_by_id[i])
        export_file_header(attributes_by_id, sys.argv[2], sys.argv[1])
        #load_orig_file(sys.argv[1])
