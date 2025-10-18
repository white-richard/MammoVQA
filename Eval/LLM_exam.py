import json
import numpy as np
from collections import defaultdict
import difflib
from fuzzywuzzy import process
import re
from scipy.stats import bootstrap, ttest_rel
from sklearn.metrics import f1_score
def uni_option(x, label_options):
    if x in label_options:
        return x
    else:
        for label_option in label_options:
            label, option = label_option.split(': ')
            if x == label or x == option:
                return label_option

def combine_scores(similarities):
    similarities = {tup[0]: tup[1] for tup in similarities}
    new_similarities = {}
    for option, score in similarities.items():
        if ':' in option:
            label, answer = option.split(': ')
            new_similarities[option] = score + similarities[label] + similarities[answer]

    new_similarities = [(key, value) for key, value in new_similarities.items()]
    return new_similarities

def extract_best_option(prediction, qas_question, question_type, target):
    shuffled_options = [
        part.strip() for part in qas_question.split("### Options:")[1].split(". ### Answer:")[0].split(",")
    ]
    labels = [option.split(":")[0].strip() for option in shuffled_options]
    options = [option.split(":")[1].strip() for option in shuffled_options]
    formatted_options = labels + shuffled_options + options

    if question_type != 'multiple choice':
        similarities = []
        for option in formatted_options:
            similarity = difflib.SequenceMatcher(None, prediction.lower(), option.lower()).ratio()
            similarities.append((option, similarity))

        similarities.sort(key=lambda x: x[1], reverse=True)
        max_score = similarities[0][1]
        top_options = [option for option, score in similarities if score == max_score]

        top_options = [uni_option(top_option, shuffled_options) for top_option in top_options]
        if len(top_options) > 1:
            same_option = len(set(top_options)) == 1
            if same_option:
                best_option = top_options[0]
            else:
                combine_score = combine_scores(similarities)
                combine_score.sort(key=lambda x: x[1], reverse=True)
                max_score = combine_score[0][1]
                top_options = [option for option, score in combine_score if score == max_score]
                if len(top_options) > 1:
                    same_option = len(set(top_options)) == 1
                    if same_option:
                        best_option = top_options[0]
                    else:
                        similarities = process.extract(prediction, formatted_options, limit=50)
                        max_score = similarities[0][1]
                        top_options = [option for option, score in similarities if score == max_score]
                        top_options = [uni_option(top_option, shuffled_options) for top_option in top_options]
                        if len(top_options) > 1:
                            same_option = len(set(top_options)) == 1
                            if same_option:
                                best_option = top_options[0]
                            else:
                                combine_score = combine_scores(similarities)
                                combine_score.sort(key=lambda x: x[1], reverse=True)
                                max_score = combine_score[0][1]
                                top_options = [option for option, score in combine_score if score == max_score]
                                if len(top_options) > 1:
                                    same_option = len(set(top_options)) == 1
                                    if same_option:
                                        best_option = top_options[0]
                                    else:
                                        return 0.0,'Make no choice'
                                else:
                                    best_option = top_options[0]
                        else:
                            best_option = top_options[0]
                else:
                    best_option = top_options[0]
        else:
            best_option = top_options[0]
        target = uni_option(target, shuffled_options)
        if best_option == target:
            return 1.0,best_option.split(': ')[-1]
        else:
            return 0.0,best_option.split(': ')[-1]
    else:
        keywords = re.findall(r'\b\w+\b', prediction.lower())
        selected_options = []
        for option in formatted_options:
            option_keywords = re.findall(r'\b\w+\b', option.lower())
            match_count = sum(1 for keyword in keywords if keyword in option_keywords)
            if match_count > 0:
                selected_options.append(option)
        selected_options = set([uni_option(selected_option, shuffled_options) for selected_option in selected_options])
        target = set([uni_option(_, shuffled_options) for _ in target])
        if selected_options == target:
            return 1.0,None
        else:
            return 0.0,None

def calculate_confidence_interval(scores, confidence_level=0.95):

    if len(scores) == 0: 
        return (0.0, 0.0)
    

    res = bootstrap((np.array(scores),), np.mean, confidence_level=confidence_level)
    ci_low = res.confidence_interval.low
    ci_high = res.confidence_interval.high

    ci_low_percent = round(ci_low * 100, 2)
    ci_high_percent = round(ci_high * 100, 2)

    return (ci_low_percent, ci_high_percent)

def calculate_p_value(scores1, scores2):

    if len(scores1) == 0 or len(scores2) == 0: 
        return 1.0
    if len(scores1) != len(scores2): 
        return 1.0
    t_stat, p_value = ttest_rel(scores1, scores2)
    return p_value

def calculate_qas(eval_data, test_data, question_topic_info):
    scores_by_topic = defaultdict(list)
    target_by_topic = defaultdict(list)
    class_counts_by_topic = defaultdict(lambda: defaultdict(int)) 
    unique_categories = defaultdict(dict) 

    scores_by_dataset = defaultdict(list)
    
    pred_labels_by_topic = defaultdict(list)
    gt_labels_by_topic = defaultdict(list)

    pred_labels_by_dataset = defaultdict(list)
    gt_labels_by_dataset = defaultdict(list)

    for topic, info in question_topic_info.items():
        categories = info[1] 
        for idx, category in enumerate(categories):
            unique_categories[topic][category] = idx  
        unique_categories[topic]["Make no choice"] = len(categories)


    for idx, test_sample in test_data.items():
        if idx in eval_data:
            eval_sample = eval_data[idx]
            question_topic = eval_sample['Question topic']
            question_type = eval_sample['Question type']
            category = eval_sample['Answer'] 
            dataset = eval_sample['Dataset'] 

            if isinstance(category, list):
                category = tuple(category) 

     
            prediction = test_sample['qas_answer']
            question = test_sample['qas_question']
            qas_score,pred_label = extract_best_option(prediction, question, question_type, category)

 
            scores_by_topic[question_topic].append(qas_score)
            scores_by_dataset[dataset].append(qas_score)

            if question_topic == "Abnormality":
                continue
            target_by_topic[question_topic].append(unique_categories[question_topic][category])
            
            if question_topic != "Abnormality" and not isinstance(category, list):
                gt_labels_by_topic[question_topic].append(unique_categories[question_topic][category])
                pred_labels_by_topic[question_topic].append(unique_categories[question_topic].get(pred_label, -1))

                gt_labels_by_dataset[dataset].append(unique_categories[question_topic][category])
                pred_labels_by_dataset[dataset].append(unique_categories[question_topic].get(pred_label, -1))

      
            class_counts_by_topic[question_topic][category] += 1
        else:

            scores_by_topic[question_topic].append(0.0)
            scores_by_dataset[dataset].append(0.0)

            if question_topic == "Abnormality":
                continue
            target_by_topic[question_topic].append(-1) 

    weighted_accuracy_by_topic = {}
    simple_accuracy_by_topic = {}
    confidence_intervals_by_topic = {}
    for question_topic, scores in scores_by_topic.items():
        if question_topic == "Abnormality":
            total_score = np.sum(scores)
            total_count = len(scores)
            simple_accuracy = total_score / total_count if total_count > 0 else 0.0
            weighted_accuracy_by_topic[question_topic] = simple_accuracy
            simple_accuracy_by_topic[question_topic] = simple_accuracy
            confidence_intervals_by_topic[question_topic] = calculate_confidence_interval(scores)
            continue

        target = np.array(target_by_topic[question_topic])

        valid_indices = target != -1
        scores = np.array(scores)[valid_indices]
        target = target[valid_indices]

        categories = question_topic_info[question_topic][1]


        class_counts = np.array([class_counts_by_topic[question_topic].get(category, 0) 
                                 for category in categories])

        n = len(scores)
        if n == 0 or np.sum(class_counts) == 0:
            weighted_accuracy_by_topic[question_topic] = 0.0 
            simple_accuracy_by_topic[question_topic] = 0.0
            confidence_intervals_by_topic[question_topic] = (0.0, 0.0)
            continue

        weights = n / class_counts[target]

        weighted_accuracy = np.sum(weights * scores) / np.sum(weights)

        simple_accuracy = np.mean(scores)

        weighted_accuracy_by_topic[question_topic] = weighted_accuracy
        simple_accuracy_by_topic[question_topic] = simple_accuracy
        confidence_intervals_by_topic[question_topic] = calculate_confidence_interval(scores)

    simple_accuracy_by_dataset = {}
    confidence_intervals_by_dataset = {}
    for dataset, scores in scores_by_dataset.items():
        total_score = np.sum(scores)
        total_count = len(scores)
        simple_accuracy = total_score / total_count if total_count > 0 else 0.0
        simple_accuracy_by_dataset[dataset] = simple_accuracy
        confidence_intervals_by_dataset[dataset] = calculate_confidence_interval(scores)

    f1_by_topic = {}
    for topic in gt_labels_by_topic:
        f1_by_topic[topic] = f1_score(gt_labels_by_topic[topic], pred_labels_by_topic[topic], average="macro")

    f1_by_dataset = {}
    for dataset in gt_labels_by_dataset:
        f1_by_dataset[dataset] = f1_score(gt_labels_by_dataset[dataset], pred_labels_by_dataset[dataset], average="macro")

    return (weighted_accuracy_by_topic, simple_accuracy_by_topic, simple_accuracy_by_dataset, 
            confidence_intervals_by_topic, confidence_intervals_by_dataset,
            f1_by_topic, f1_by_dataset) 
    
def format_qas_cs_output(eval_data, test_data, question_topic_info):
    (weighted_accuracy_by_topic, simple_accuracy_by_topic, simple_accuracy_by_dataset, 
     confidence_intervals_by_topic, confidence_intervals_by_dataset,
     f1_by_topic, f1_by_dataset) = calculate_qas(eval_data, test_data, question_topic_info)

    print("\nF1 by Dataset:")
    for dataset, f1 in f1_by_dataset.items():
        print(f"Dataset {dataset}: F1 = {f1*100:.2f}%")

    print("\nF1 by Question Topic:")
    for topic, f1 in f1_by_topic.items():
        print(f"Question Topic {topic}: F1 = {f1*100:.2f}%")

    print("\nQAS/CS by Dataset:")
    for dataset in simple_accuracy_by_dataset:
        qas_score = simple_accuracy_by_dataset[dataset] * 100 
        ci_low, ci_high = confidence_intervals_by_dataset.get(dataset, (0.0, 0.0))
        print(f"Dataset {dataset}: Simple Accuracy = {qas_score:.2f}%, CI = ({ci_low:.2f}%, {ci_high:.2f}%)")

    print("\nQAS/CS by Question Topic:")
    for topic in simple_accuracy_by_topic:
        qas_score = simple_accuracy_by_topic[topic] * 100 
        ci_low, ci_high = confidence_intervals_by_topic.get(topic, (0.0, 0.0))
        print(f"Question Topic {topic}: Simple Accuracy = {qas_score:.2f}%, CI = ({ci_low:.2f}%, {ci_high:.2f}%)")

    print("\nWeighted Accuracy by Question Topic:")
    for topic in weighted_accuracy_by_topic:
        qas_score = weighted_accuracy_by_topic[topic] * 100  
        ci_low, ci_high = confidence_intervals_by_topic.get(topic, (0.0, 0.0))
        print(f"Question Topic {topic}: Weighted Accuracy = {qas_score:.2f}%")


    p_values_by_topic = {}
    for topic in weighted_accuracy_by_topic:
        current_scores = [extract_best_option(test_data[idx]['qas_answer'], test_data[idx]['qas_question'], 
                                             eval_data[idx]['Question type'], eval_data[idx]['Answer'])[0]
                          for idx in test_data if idx in eval_data and eval_data[idx]['Question topic'] == topic]
        
        with open('/home/jiayi/MammoVQA/Result/LLAVA-Mammo-exam.json', 'r') as f:
            llava_data = json.load(f)
        
        llava_scores = [extract_best_option(llava_data[idx]['qas_answer'], llava_data[idx]['qas_question'], 
                                           eval_data[idx]['Question type'], eval_data[idx]['Answer'])[0] 
                        for idx in llava_data if idx in eval_data and eval_data[idx]['Question topic'] == topic]
        
        p_values_by_topic[topic] = calculate_p_value(current_scores, llava_scores)

    print("\nP-values by Question Topic:")
    for topic in p_values_by_topic:
        p_value = p_values_by_topic[topic]
        print(f"Question Topic {topic}: p-value = {p_value:.4f}")

if __name__ == "__main__":
    with open('/home/jiayi/MammoVQA/Benchmark/MammoVQA-Exam-Bench.json', 'r') as f:
        eval_data = json.load(f)
    method='InternVL3-8B-Exam'
    with open(f'/home/jiayi/MammoVQA/Result/{method}.json', 'r') as f:
        test_data = json.load(f)
    print(f'-------{method}--------')
    question_topic_info = {
        "ACR": ["single choice", ["Level A", "Level B", "Level C", "Level D"]],
        "Bi-Rads": ["single choice", ["Bi-Rads 0", "Bi-Rads 1", "Bi-Rads 2", "Bi-Rads 3", "Bi-Rads 4", "Bi-Rads 5"]]
    }
    format_qas_cs_output(eval_data, test_data, question_topic_info)
