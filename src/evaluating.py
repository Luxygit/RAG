""""""
from pathlib import Path
import json

def calculate_iou(start1: int, end1: int, start2: int, end2: int) -> float:
    """"""
    end = min(end1, end2)
    start = max(start1, start2)
    intersection = end - start
    if intersection < 0:
        intersection = 0
    chunk1 = end1 - start1
    chunk2 = end2 - start2
    union = chunk1 + chunk2 - intersection
    if union == 0:
        return 0.0
    iou = intersection / union
    return iou


def evaluate(student_search_results_path: Path, dataset_path: Path) -> float:
    """"""
    student_res = json.load(student_search_results_path)
    ground_res = json.load(dataset_path)
    correct_count = 0
    truth_map = {}
    for question_id, sources in ground_res.items():
        truth_map[question_id] = sources
    for s_question_id, s_sources in student_res.items():
        if truth_map[s_question_id]:
            found_match = False
            for s in s_sources:
                if s["file_path"] == truth_map[s_question_id]
                and calculate_iou() >= 0.05:
                    found_match = True
                    break
            if found_match == True:
                correct_count += 1
    if total_number_of_questions == 0:
        return 0.0
    recall = correct_count / total_number_of_questions
    return recall

