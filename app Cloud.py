import pandas as pd
import numpy as np
from nltk.corpus import wordnet
import csv
import json
import itertools
from spacy.lang.en.stop_words import STOP_WORDS
import spacy
import joblib
from flask import Flask, render_template, request, session



app = Flask(__name__)

nlp = spacy.load('en_core_web_sm')

# save data
data = {"users": []}
with open('DATA.json', 'w') as outfile:
    json.dump(data, outfile)


def write_json(new_data, filename='DATA.json'):
    with open(filename, 'r+') as file:
        # First we load existing data into a dict.
        file_data = json.load(file)
        # Join new_data with file_data inside emp_details
        file_data["users"].append(new_data)
        # Sets file's current position at offset.
        file.seek(0)
        # convert back to json.
        json.dump(file_data, file, indent=4)


df_tr = pd.read_csv('Medical_dataset/Training.csv')
df_tt = pd.read_csv('Medical_dataset/Testing.csv')
with open('Ambulance Numbers.csv', 'r') as f:
    amb_reader = csv.reader(f)
    amb_data = list(amb_reader)
amb_array = np.array(amb_data)

#Palestine Nearby Hospital Trial
import numpy as np
import csv
with open('Hospital.csv', 'r') as f:
    hosp_reader = csv.reader(f)
    hosp_data = list(hosp_reader)
hosp_array = np.array(hosp_data)
hosp_array = np.delete(hosp_array, 0, 0) 
hosp_filt = hosp_array[hosp_array[:, -1] != '0']
hosp=np.unique(hosp_filt[:, 1])
hos = 0
hos_t = ''
for i in hosp:
    hos_i = str(hos) + " for " + i +". "
    hos_t = hos_t + hos_i
    hos = hos+1

def get_city_name(code):
  city_names = {
      0: "Beit Jala", 1: "Bethlehem", 2: "Deir Al-Balah", 3: "Gaza City", 4: "Hebron", 5: "Jenin",
      6: "Jericho", 7: "Jerusalem", 8: "Jerusalem (Grou)", 9: "Khan Younis", 10: "Nablus",
      11: "Rafah", 12: "Ramallah", 13: "Salfit", 14: "Tulkarm"
  }
  return city_names.get(code, "Unknown City") 

symp = []
disease = []
for i in range(len(df_tr)):
    symp.append(df_tr.columns[df_tr.iloc[i] == 1].to_list())
    disease.append(df_tr.iloc[i, -1])

# # I- GET ALL SYMPTOMS

all_symp_col = list(df_tr.columns[:-1])


def clean_symp(sym):
    return sym.replace('_', ' ').replace('.1', '').replace('(typhos)', '').replace('yellowish', 'yellow').replace(
        'yellowing', 'yellow')


all_symp = [clean_symp(sym) for sym in (all_symp_col)]


def preprocess(doc):
    nlp_doc = nlp(doc)
    d = []
    for token in nlp_doc:
        if (not token.text.lower() in STOP_WORDS and token.text.isalpha()):
            d.append(token.lemma_.lower())
    return ' '.join(d)


all_symp_pr = [preprocess(sym) for sym in all_symp]

# associate each processed symp with column name
col_dict = dict(zip(all_symp_pr, all_symp_col))


# II- Syntactic Similarity

# Returns all the subsets of a set. This is a generator.
# {1,2,3}->[{},{1},{2},{3},{1,3},{1,2},..]
def powerset(seq):
    if len(seq) <= 1:
        yield seq
        yield []
    else:
        for item in powerset(seq[1:]):
            yield [seq[0]] + item
            yield item


# Sort list based on length
def sort(a):
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            if len(a[j]) > len(a[i]):
                a[i], a[j] = a[j], a[i]
    a.pop()
    return a


# find all permutations of a list
def permutations(s):
    permutations = list(itertools.permutations(s))
    return ([' '.join(permutation) for permutation in permutations])


# check if a txt and all diferrent combination if it exists in processed symp list
def DoesExist(txt):
    txt = txt.split(' ')
    combinations = [x for x in powerset(txt)]
    sort(combinations)
    for comb in combinations:
        # print(permutations(comb))
        for sym in permutations(comb):
            if sym in all_symp_pr:
                # print(sym)
                return sym
    return False


# Jaccard similarity 2docs
def jaccard_set(str1, str2):
    list1 = str1.split(' ')
    list2 = str2.split(' ')
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union


# apply vanilla jaccard to symp with all corpus
def syntactic_similarity(symp_t, corpus):
    most_sim = []
    poss_sym = []
    for symp in corpus:
        d = jaccard_set(symp_t, symp)
        most_sim.append(d)
    order = np.argsort(most_sim)[::-1].tolist()
    for i in order:
        if DoesExist(symp_t):
            return 1, [corpus[i]]
        if corpus[i] not in poss_sym and most_sim[i] != 0:
            poss_sym.append(corpus[i])
    if len(poss_sym):
        return 1, poss_sym
    else:
        return 0, None


# check a pattern if it exists in processed symp list
def check_pattern(inp, dis_list):
    import re
    pred_list = []
    ptr = 0
    patt = "^" + inp + "$"
    regexp = re.compile(inp)
    for item in dis_list:
        if regexp.search(item):
            pred_list.append(item)
    if (len(pred_list) > 0):
        return 1, pred_list
    else:
        return ptr, None


# III- Semantic Similarity


from nltk.wsd import lesk
from nltk.tokenize import word_tokenize


def WSD(word, context):
    sens = lesk(context, word)
    return sens


# semantic similarity 2docs
def semanticD(doc1, doc2):
    doc1_p = preprocess(doc1).split(' ')
    doc2_p = preprocess(doc2).split(' ')
    score = 0
    for tock1 in doc1_p:
        for tock2 in doc2_p:
            syn1 = WSD(tock1, doc1)
            syn2 = WSD(tock2, doc2)
            if syn1 is not None and syn2 is not None:
                x = syn1.wup_similarity(syn2)
                # x=syn1.path_similarity((syn2))
                if x is not None and x > 0.25:
                    score += x
    return score / (len(doc1_p) * len(doc2_p))


# apply semantic simarity to symp with all corpus
def semantic_similarity(symp_t, corpus):
    max_sim = 0
    most_sim = None
    for symp in corpus:
        d = semanticD(symp_t, symp)
        if d > max_sim:
            most_sim = symp
            max_sim = d
    return max_sim, most_sim


# given a symp suggest possible synonyms
def suggest_syn(sym):
    symp = []
    synonyms = wordnet.synsets(sym)
    lemmas = [word.lemma_names() for word in synonyms]
    lemmas = list(set(itertools.chain(*lemmas)))
    for e in lemmas:
        res, sym1 = semantic_similarity(e, all_symp_pr)
        if res != 0:
            symp.append(sym1)
    return list(set(symp))


# One-Hot-Vector dataframe
def OHV(cl_sym, all_sym):
    l = np.zeros([1, len(all_sym)])
    for sym in cl_sym:
        l[0, all_sym.index(sym)] = 1
    return pd.DataFrame(l, columns=all_symp)


def contains(small, big):
    a = True
    for i in small:
        if i not in big:
            a = False
    return a


# list of symptoms --> possible diseases
def possible_diseases(l):
    poss_dis = []
    for dis in set(disease):
        if contains(l, symVONdisease(df_tr, dis)):
            poss_dis.append(dis)
    return poss_dis


# disease --> all symptoms
def symVONdisease(df, disease):
    ddf = df[df.prognosis == disease]
    m2 = (ddf == 1).any()
    return m2.index[m2].tolist()


# IV- Prediction Model (KNN)
# load model
knn_clf = joblib.load('knn.pkl')

# ##  VI- SEVERITY / DESCRIPTION / PRECAUTION
# get dictionaries for severity-description-precaution for all diseases

severityDictionary = dict()
description_list = dict()
precautionDictionary = dict()


def getDescription():
    global description_list
    with open('Medical_dataset/symptom_Description.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            _description = {row[0]: row[1]}
            description_list.update(_description)


def getSeverityDict():
    global severityDictionary
    with open('Medical_dataset/symptom_severity.csv') as csv_file:

        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        try:
            for row in csv_reader:
                _diction = {row[0]: int(row[1])}
                severityDictionary.update(_diction)
        except:
            pass


def getprecautionDict():
    global precautionDictionary
    with open('Medical_dataset/symptom_precaution.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            _prec = {row[0]: [row[1], row[2], row[3], row[4]]}
            precautionDictionary.update(_prec)


# load dictionaries
getSeverityDict()
getprecautionDict()
getDescription()


# calcul patient condition
def calc_condition(exp, days):
    sum = 0
    for item in exp:
        if item in severityDictionary.keys():
            sum = sum + severityDictionary[item]
    if ((sum * days) / (len(exp)) > 13):
        return 1
        print("You should take the consultation from doctor. ")
    else:
        return 0
        print("It might not be that bad but you should take precautions.")


# print possible symptoms
def related_sym(psym1):
    s = "could you be more specific, <br>"
    i = len(s)
    for num, it in enumerate(psym1):
        s += str(num) + ") " + clean_symp(it) + "<br>"
    if num != 0:
        s += "Select the one you meant."
        return s
    else:
        return 0


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/get")
def get_bot_response():
    from chat import get_response
    from chat import predict_class, intents
    s = request.args.get('msg')
    if "step" in session:
        if session["step"] == "Q_C":
            name = session["name"]
            age = session["age"]
            gender = session["gender"]
            session.clear()
            if s == "q":
                "Thank you for using ower web site Mr/Ms " + name
            else:
                session["step"] = "FS"
                session["name"] = name
                session["age"] = age
                session["gender"] = gender
    if s.upper() == "OK":
        return "What is your name ?"
    if 'name' not in session and 'step' not in session:
        session['name'] = s
        session['step'] = "age"
        return "How old are you? "
    if session["step"] == "age":
        session["age"] = int(s)
        session["step"] = "gender"
        return "Can you specify your gender ?"
    if session["step"] == "gender":
        session["gender"] = s
        session["step"] = "check"
    if session["step"] == "check":
        session["step"] = "check2"
        return "Please type I if you need to treat an injury, H if you want to find nearby hospital and D if you need a disease diagnosis: "
    if session["step"] == "check2":        
        if s.upper() == "I":
            session["step"] = "inj"
        elif s.upper() == "D":
            session["step"] = "Dise"
        elif s.upper() == "H":
            session["step"] = "hos"
        else: 
            return "Please enter either I or D or H: "
        
    if session["step"] == "hos":
        session["step"] = "hos1"
        return "Type " + hos_t + " Choose the city based on your proximity. "
    
    if session["step"] == "hos1":
        session["step"] = "check"
        c = int(s)
        city = get_city_name(c)
        print(f"City name for code {c}: {city}")
        b=''
        d = 1
        for row in hosp_array:
            for column in row:
                if column == city:
                    if row[4] == '1':
                        a = str(d) + ') ' + row[0] + ', ' + row[1] + ', ' + row[2] + ', Contact Number is ' + row[3] + '. '
                        b = b + a
                        d = d + 1
        return "The operational hospitals nearby as of 15 April, 2024 are " + str(b) + " \nPlease call them in advance to confirm if they're currently operational. Type A if you need more assistance."

    if session["step"] == "inj":
        session["step"] = "country"
        return "Which country are you in ?" 
        
    if session["step"] == "country":
        session["country"] = s
    
        f = 0
        for row in amb_array:
            for column in row:
                if column == session["country"]:
                    session["step"] = "inj1"
                    f = 1
                    return "Your country's emergnecy helpline number is "  + row[1] + ".\nIf you've been shot, type 1.\n If you've been injured in a blast, type 2.\n If you were hit or struck by something forcefully, type 3. \nIf you've been stabbed, type 4.\n If none of these, type 5\n"
        if f == 0:
            return "Enter valid country name"
 

    if session["step"] == "inj1":    
        if int(s) == 1:
            #Q is If you've been shot
            session["step"] = "shot"
            return "Is the bullet still inside? (yes or no)"
        if int(s) == 2:
            #Q is If you've been injured in a blast
            session["step"] = "blast"
            return "If you've stepped on landmine, type 1\nIf you've been attacked by air bombardment or artillery shelling, type 2\n "
        if int(s) == 3:
            #Q is If you were hit or struck by something forcefully
            session["step"] = "blunt"
            return "Are you experiencing any of the following: Headache, nausea, vomiting, vision loss, weakness/numbness, vision loss, seizures (yes or no)."
            
        if int(s) == 4:
             #Q is If you've been stabbed
            session["step"] = "stab"
            return "Type 1 if the stab wound is penetrating (knife has just enterred the body on one side) and 2 if the stab wound is perforating (knife has enterred the body on one side and come out from the other side)"
        # if int(s) == 5:
        #    #Q If you've been exposed to some chemical agent
        #     session["step"] = "chem"
        #     return "Is the patient conscious? (yes or no)"
        if int(s) == 5:
            #Q is If none of these
            session["step"] = "inj2"
            return "What sort of injury do you have?"


    if session["step"] == "shot":
        if s.lower() == "yes":
            text = "bullet inside"
        elif s.lower() == "no":
            text = "bullet has left hole in me"
        ints=predict_class(text)
        response = get_response(ints, intents)
        session['step'] = "check"
        return response + "\nYour injury diagnosis is done. Type A if you need more assistance?"

    if session["step"] == "blast":
        if int(s) == 1:
            text = "I stepped on a landmine"
        elif int(s) == 2:
            text = "How do you treat blast injury?"
        ints=predict_class(text)
        response = get_response(ints, intents)
        session['step'] = "check"
        return response + "\nYour injury diagnosis is done. Type A if you need more assistance?"
    
    if session["step"] == "blunt":
        if s.lower() == "yes":
            text = "How do you treat internal bleeding?"
        elif s.lower() == "no":
            text = "How do you treat blunt force trauma?"
        ints=predict_class(text)
        response = get_response(ints, intents)
        session['step'] = "check"
        return response + "\nYour injury diagnosis is done. Type A if you need more assistance?"
    
    if session["step"] == "stab":
        if int(s) == 1:
            text = "How do you treat a penetrating knife stabbing?"
        elif int(s) == 2:
            text = "How do you treat a perforating knife stabbing?"
        ints=predict_class(text)
        response = get_response(ints, intents)
        s = {"answer": response}
        session['step'] = "check"
        return response + "\n Your injury diagnosis is done. Type A if you need more assistance?"
    
    # if session["step"] == "chem":
    #     if s.lower() == "yes":
    #         text = "gas conscious"
    #     elif s.lower() == "no":
    #         text = "gas faint"
    #     ints=predict_class(text)
    #     response = get_response(ints, intents)
    #     session['step'] = "check"
    #     return response + "\nYour injury diagnosis is done. Type A if you need more assistance?"
    
    if session["step"] == "inj2":
        try:
            text = s
            print("Initial value of s is: " + s)
            ints=predict_class(text)
            print("\nInitial value of ints after class prediction is: ")
            print(ints)
            response = get_response(ints, intents)
            print("\nInitial value of response after getting response is: ")
            print(response)
            s = {"answer": response}
            print("\nFinal value of s is: ")
            print(s)
            session['step'] = "check"
            return response + "\nYour injury diagnosis is done. Type A if you need more assistance?"
        except json.decoder.JSONDecodeError:
            print(type(s))
            return "not a valid JSON :(" + s
    if session['step'] == "Dise":
        session['step'] = "BFS"
        return "Okay " + session[
            "name"] + ", now I will be asking some few questions about your symptoms to see what you should do. Tap S to start diagnostic!"
    if session['step'] == "BFS":
        session['step'] = "FS"  # first symp
        return "Can you precisely describe your main symptom " + session["name"] + " ?"
    if session['step'] == "FS":
        sym1 = s
        sym1 = preprocess(sym1)
        sim1, psym1 = syntactic_similarity(sym1, all_symp_pr)
        temp = [sym1, sim1, psym1]
        session['FSY'] = temp  # info du 1er symptome
        session['step'] = "SS"  # second symptomee
        if sim1 == 1:
            session['step'] = "RS1"  # related_sym1
            s = related_sym(psym1)
            if s != 0:
                return s
        else:
            return "You are probably facing another symptom, if so, can you specify it?"
    if session['step'] == "RS1":
        temp = session['FSY']
        psym1 = temp[2]
        psym1 = psym1[int(s)]
        temp[2] = psym1
        session['FSY'] = temp
        session['step'] = 'SS'
        return "You are probably facing another symptom, if so, can you specify it?"
    if session['step'] == "SS":
        sym2 = s
        sym2 = preprocess(sym2)
        sim2 = 0
        psym2 = []
        if len(sym2) != 0:
            sim2, psym2 = syntactic_similarity(sym2, all_symp_pr)
        temp = [sym2, sim2, psym2]
        session['SSY'] = temp  # info du 2eME symptome(sym,sim,psym)
        session['step'] = "semantic"  # face semantic
        if sim2 == 1:
            session['step'] = "RS2"  # related sym2
            s = related_sym(psym2)
            if s != 0:
                return s
    if session['step'] == "RS2":
        temp = session['SSY']
        psym2 = temp[2]
        psym2 = psym2[int(s)]
        temp[2] = psym2
        session['SSY'] = temp
        session['step'] = "semantic"
    if session['step'] == "semantic":
        temp = session["FSY"]  # recuperer info du premier
        sym1 = temp[0]
        sim1 = temp[1]
        temp = session["SSY"]  # recuperer info du 2 eme symptome
        sym2 = temp[0]
        sim2 = temp[1]
        if sim1 == 0 or sim2 == 0:
            session['step'] = "BFsim1=0"
        else:
            session['step'] = 'PD'  # to possible_diseases
    if session['step'] == "BFsim1=0":
        if sim1 == 0 and len(sym1) != 0:
            sim1, psym1 = semantic_similarity(sym1, all_symp_pr)
            temp = []
            temp.append(sym1)
            temp.append(sim1)
            temp.append(psym1)
            session['FSY'] = temp
            session['step'] = "sim1=0"  # process of semantic similarity=1 for first sympt.
        else:
            session['step'] = "BFsim2=0"
    if session['step'] == "sim1=0":  # semantic no => suggestion
        temp = session["FSY"]
        sym1 = temp[0]
        sim1 = temp[1]
        if sim1 == 0:
            if "suggested" in session:
                sugg = session["suggested"]
                if s == "yes":
                    psym1 = sugg[0]
                    sim1 = 1
                    temp = session["FSY"]
                    temp[1] = sim1
                    temp[2] = psym1
                    session["FSY"] = temp
                    sugg = []
                else:
                    del sugg[0]
            if "suggested" not in session:
                session["suggested"] = suggest_syn(sym1)
                sugg = session["suggested"]
            if len(sugg) > 0:
                msg = "are you experiencing any  " + sugg[0] + "?"
                return msg
        if "suggested" in session:
            del session["suggested"]
        session['step'] = "BFsim2=0"
    if session['step'] == "BFsim2=0":
        temp = session["SSY"]  # recuperer info du 2 eme symptome
        sym2 = temp[0]
        sim2 = temp[1]
        if sim2 == 0 and len(sym2) != 0:
            sim2, psym2 = semantic_similarity(sym2, all_symp_pr)
            temp = []
            temp.append(sym2)
            temp.append(sim2)
            temp.append(psym2)
            session['SSY'] = temp
            session['step'] = "sim2=0"
        else:
            session['step'] = "TEST"
    if session['step'] == "sim2=0":
        temp = session["SSY"]
        sym2 = temp[0]
        sim2 = temp[1]
        if sim2 == 0:
            if "suggested_2" in session:
                sugg = session["suggested_2"]
                if s == "yes":
                    psym2 = sugg[0]
                    sim2 = 1
                    temp = session["SSY"]
                    temp[1] = sim2
                    temp[2] = psym2
                    session["SSY"] = temp
                    sugg = []
                else:
                    del sugg[0]
            if "suggested_2" not in session:
                session["suggested_2"] = suggest_syn(sym2)
                sugg = session["suggested_2"]
            if len(sugg) > 0:
                msg = "Are you experiencing " + sugg[0] + "?"
                session["suggested_2"] = sugg
                return msg
        if "suggested_2" in session:
            del session["suggested_2"]
        session['step'] = "TEST"  # test if semantic and syntaxic and suggestion not found
    if session['step'] == "TEST":
        temp = session["FSY"]
        sim1 = temp[1]
        psym1 = temp[2]
        temp = session["SSY"]
        sim2 = temp[1]
        psym2 = temp[2]
        if sim1 == 0 and sim2 == 0:
            # GO TO THE END
            result = None
            session['step'] = "END"
        else:
            if sim1 == 0:
                psym1 = psym2
                temp = session["FSY"]
                temp[2] = psym2
                session["FSY"] = temp
            if sim2 == 0:
                psym2 = psym1
                temp = session["SSY"]
                temp[2] = psym1
                session["SSY"] = temp
            session['step'] = 'PD'  # to possible_diseases
    if session['step'] == 'PD':
        # MAYBE THE LAST STEP
        # create patient symp list
        temp = session["FSY"]
        sim1 = temp[1]
        psym1 = temp[2]
        temp = session["SSY"]
        sim2 = temp[1]
        psym2 = temp[2]
        print("hey2")
        if "all" not in session:
            session["asked"] = []
            session["all"] = [col_dict[psym1], col_dict[psym2]]
            print(session["all"])
        session["diseases"] = possible_diseases(session["all"])
        print(session["diseases"])
        all_sym = session["all"]
        diseases = session["diseases"]
        dis = diseases[0]
        session["dis"] = dis
        session['step'] = "for_dis"
    if session['step'] == "DIS":
        if "symv" in session:
            if len(s) > 0 and len(session["symv"]) > 0:
                symts = session["symv"]
                all_sym = session["all"]
                if s == "yes":
                    all_sym.append(symts[0])
                    session["all"] = all_sym
                    print(possible_diseases(session["all"]))
                del symts[0]
                session["symv"] = symts
        if "symv" not in session:
            session["symv"] = symVONdisease(df_tr, session["dis"])
        if len(session["symv"]) > 0:
            if symts[0] not in session["all"] and symts[0] not in session["asked"]:
                asked = session["asked"]
                asked.append(symts[0])
                session["asked"] = asked
                symts = session["symv"]
                msg = "Are you experiencing " + clean_symp(symts[0]) + "?"
                return msg
            else:
                del symts[0]
                session["symv"] = symts
                s = ""
                print("HANAAA")
                return get_bot_response()
        else:
            PD = possible_diseases(session["all"])
            diseases = session["diseases"]
            if diseases[0] in PD:
                session["testpred"] = diseases[0]
                PD.remove(diseases[0])
            #            diseases=session["diseases"]
            #            del diseases[0]
            session["diseases"] = PD
            session['step'] = "for_dis"
    if session['step'] == "for_dis":
        diseases = session["diseases"]
        if len(diseases) <= 0:
            session['step'] = 'PREDICT'
        else:
            session["dis"] = diseases[0]
            session['step'] = "DIS"
            session["symv"] = symVONdisease(df_tr, session["dis"])
            return get_bot_response()  # turn around sympt of dis
        # predict possible diseases
    if session['step'] == "PREDICT":
        result = knn_clf.predict(OHV(session["all"], all_symp_col))
        session['step'] = "END"
    if session['step'] == "END":
        if result is not None:
            if result[0] != session["testpred"]:
                session['step'] = "Q_C"
                return "as you provide me with few symptoms, I am sorry to announce that I cannot predict your " \
                       "disease for the moment!!! <br> Can you specify more about what you are feeling or Tap q to " \
                       "stop the conversation "
            session['step'] = "Description"
            session["disease"] = result[0]
            return "Well Mr/Ms " + session["name"] + ", you may have " + result[
                0] + ". Tap D to get a description of the disease ."
        else:
            session['step'] = "Q_C"  # test if user want to continue the conversation or not
            return "can you specify more what you feel or Tap q to stop the conversation"
    if session['step'] == "Description":
        y = {"Name": session["name"], "Age": session["age"], "Gender": session["gender"], "Disease": session["disease"],
             "Sympts": session["all"]}
        write_json(y)
        session['step'] = "Severity"
        if session["disease"] in description_list.keys():
            return description_list[session["disease"]] + " \n <br>  How many days have you had symptoms?"
        else:
            if " " in session["disease"]:
                session["disease"] = session["disease"].replace(" ", "_")
            return "please visit <a href='" + "https://en.wikipedia.org/wiki/" + session["disease"] + "'>  here  </a>"
    if session['step'] == "Severity":
        session['step'] = 'FINAL'
        if calc_condition(session["all"], int(s)) == 1:
            return "you should take the consultation from doctor <br> Tap q to exit"
        else:
            msg = 'Nothing to worry about, but you should take the following precautions :<br> '
            i = 1
            for e in precautionDictionary[session["disease"]]:
                msg += '\n ' + str(i) + ' - ' + e + '<br>'
                i += 1
            msg += ' Tap q to end'
            return msg
    if session['step'] == "FINAL":
        session['step'] = "BYE"
        return "Your diagnosis was perfectly completed. Do you need another medical consultation (yes or no)? "
    if session['step'] == "BYE":
        name = session["name"]
        age = session["age"]
        gender = session["gender"]
        session.clear()
        if s.lower() == "yes":
            session["gender"] = gender
            session["name"] = name
            session["age"] = age
            session['step'] = "FS"
            return "HELLO again " + session["name"] + " Please tell me your main symptom. "
        else:
            return "THANKS " + name + " for using me"

if __name__ == "__main__":
    import random  # define the random module
    import string

    S = 10  # number of characters in the string.
    # call random.choices() string module to find the string in Uppercase + numeric data.
    ran = ''.join(random.choices(string.ascii_uppercase + string.digits, k=S))
    # chat_sp()
    app.secret_key = str(ran)
    app.run(host='0.0.0.0', port=8080)

# if __name__ == "__main__":
#     import random  # define the random module
#     import string

#     S = 10  # number of characters in the string.
#     # call random.choices() string module to find the string in Uppercase + numeric data.
#     ran = ''.join(random.choices(string.ascii_uppercase + string.digits, k=S))
#     # chat_sp()
#     app.secret_key = str(ran)
#     app.run()
